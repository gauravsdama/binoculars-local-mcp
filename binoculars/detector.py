"""Binoculars text detector that loads models only from local directories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import entropy, perplexity
from .offline import configure_offline_environment
from .utils import assert_tokenizer_consistency, require_local_model_path

configure_offline_environment()
torch.set_grad_enabled(False)

BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Binoculars:
    """Score text with a compatible observer/performer model pair."""

    def __init__(
        self,
        observer_name_or_path: str | Path,
        performer_name_or_path: str | Path,
        *,
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        mode: str = "low-fpr",
        device: str | None = None,
        accuracy_threshold: float = BINOCULARS_ACCURACY_THRESHOLD,
        low_fpr_threshold: float = BINOCULARS_FPR_THRESHOLD,
    ) -> None:
        if not 8 <= max_token_observed <= 4096:
            raise ValueError("max_token_observed must be between 8 and 4096")

        observer_path = require_local_model_path(observer_name_or_path, label="observer model")
        performer_path = require_local_model_path(performer_name_or_path, label="performer model")
        self.device = device or _default_device()
        self.accuracy_threshold = accuracy_threshold
        self.low_fpr_threshold = low_fpr_threshold
        self.change_mode(mode)

        # The argument is a validated local directory and remote loading is disabled.
        observer_tokenizer = AutoTokenizer.from_pretrained(
            observer_path, local_files_only=True, trust_remote_code=False
        )
        performer_tokenizer = AutoTokenizer.from_pretrained(
            performer_path, local_files_only=True, trust_remote_code=False
        )
        assert_tokenizer_consistency(observer_tokenizer, performer_tokenizer)
        self.tokenizer = observer_tokenizer
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token is None:
            raise ValueError("Local tokenizer must define either a pad token or an EOS token")

        dtype = torch.bfloat16 if use_bfloat16 and self.device != "cpu" else torch.float32
        load_options = {
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": dtype,
            "use_safetensors": True,
        }
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_path, **load_options
        ).to(self.device)
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_path, **load_options
        ).to(self.device)
        self.observer_model.eval()
        self.performer_model.eval()
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = self.low_fpr_threshold
        elif mode == "accuracy":
            self.threshold = self.accuracy_threshold
        else:
            raise ValueError("mode must be 'low-fpr' or 'accuracy'")
        self.mode = mode

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if len(batch) > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        )
        return encodings.to(self.device)

    @torch.inference_mode()
    def _get_logits(
        self, encodings: transformers.BatchEncoding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(**encodings).logits
        performer_logits = self.performer_model(**encodings).logits
        return observer_logits, performer_logits

    def compute_score(self, input_text: list[str] | str) -> list[float] | float:
        batch = [input_text] if isinstance(input_text, str) else input_text
        if not batch or any(not text.strip() for text in batch):
            raise ValueError("input_text must contain non-empty text")
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits,
            performer_logits,
            encodings,
            self.tokenizer.pad_token_id,
        )
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def predict(self, input_text: list[str] | str) -> list[str] | str:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(
            binoculars_scores < self.threshold,
            "Most likely AI-generated",
            "Most likely human-generated",
        ).tolist()
        return pred
