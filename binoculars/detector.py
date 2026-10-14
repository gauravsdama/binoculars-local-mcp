"""Binoculars text detector that loads models only from local directories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .offline import configure_offline_environment

configure_offline_environment()

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from .metrics import entropy, perplexity  # noqa: E402
from .utils import assert_tokenizer_consistency, require_local_model_path  # noqa: E402

torch.set_grad_enabled(False)

BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527
MIN_ANALYZED_TOKENS = 8


@dataclass(frozen=True)
class Analysis:
    """A finite score plus the exact tokenizer accounting used to produce it."""

    score: float
    input_tokens: int
    tokens_analyzed: int
    truncated: bool


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device(requested: str | None) -> str:
    device = requested or "auto"
    if device == "auto":
        return _default_device()
    if device == "cpu":
        return device
    if device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        return device
    if device == "cuda" or device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        try:
            torch.device(device)
        except RuntimeError as exc:
            raise ValueError(f"Invalid CUDA device {device!r}") from exc
        return device
    raise ValueError("device must be 'auto', 'cpu', 'mps', 'cuda', or 'cuda:<index>'")


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
        if not math.isfinite(accuracy_threshold) or not math.isfinite(low_fpr_threshold):
            raise ValueError("thresholds must be finite")

        observer_path = require_local_model_path(observer_name_or_path, label="observer model")
        performer_path = require_local_model_path(performer_name_or_path, label="performer model")
        self.device = _resolve_device(device)
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
        self.threshold = self.threshold_for_mode(mode)
        self.mode = mode

    def threshold_for_mode(self, mode: str) -> float:
        if mode == "low-fpr":
            return self.low_fpr_threshold
        if mode == "accuracy":
            return self.accuracy_threshold
        raise ValueError("mode must be 'low-fpr' or 'accuracy'")

    def _tokenize(
        self, batch: list[str]
    ) -> tuple[transformers.BatchEncoding, list[int], list[int]]:
        untruncated = self.tokenizer(
            batch,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_token_counts = [len(token_ids) for token_ids in untruncated["input_ids"]]
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if len(batch) > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        )
        analyzed_token_counts = encodings.attention_mask.sum(1).tolist()
        insufficient = [
            index
            for index, count in enumerate(analyzed_token_counts)
            if count < MIN_ANALYZED_TOKENS
        ]
        if insufficient:
            indexes = ", ".join(str(index) for index in insufficient)
            raise ValueError(
                f"Text at batch index {indexes} has fewer than {MIN_ANALYZED_TOKENS} "
                "analyzed tokens; provide a longer sample"
            )
        return encodings.to(self.device), input_token_counts, analyzed_token_counts

    @torch.inference_mode()
    def _get_logits(
        self, encodings: transformers.BatchEncoding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(**encodings).logits
        performer_logits = self.performer_model(**encodings).logits
        return observer_logits, performer_logits

    def analyze(self, input_text: list[str] | str) -> list[Analysis] | Analysis:
        """Analyze text and retain exact token and truncation metadata."""
        batch = [input_text] if isinstance(input_text, str) else input_text
        if not batch or any(not text.strip() for text in batch):
            raise ValueError("input_text must contain non-empty text")
        encodings, input_token_counts, analyzed_token_counts = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits,
            performer_logits,
            encodings,
            self.tokenizer.pad_token_id,
        )
        if torch.any(~torch.isfinite(ppl)) or torch.any(~torch.isfinite(x_ppl)):
            raise ValueError("Analysis produced non-finite intermediate metrics")
        if torch.any(x_ppl <= 0):
            raise ValueError("Analysis produced a non-positive cross-perplexity")
        scores = (ppl / x_ppl).tolist()
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("Analysis produced a non-finite score")
        analyses = [
            Analysis(
                score=score,
                input_tokens=input_tokens,
                tokens_analyzed=tokens_analyzed,
                truncated=input_tokens > tokens_analyzed,
            )
            for score, input_tokens, tokens_analyzed in zip(
                scores, input_token_counts, analyzed_token_counts, strict=True
            )
        ]
        return analyses[0] if isinstance(input_text, str) else analyses

    def compute_score(self, input_text: list[str] | str) -> list[float] | float:
        analyses = self.analyze(input_text)
        if isinstance(analyses, Analysis):
            return analyses.score
        return [analysis.score for analysis in analyses]

    def predict(self, input_text: list[str] | str) -> list[str] | str:
        scores = self.compute_score(input_text)
        if isinstance(scores, float):
            return (
                "Most likely AI-generated"
                if scores < self.threshold
                else "Most likely human-generated"
            )
        return [
            "Most likely AI-generated" if score < self.threshold else "Most likely human-generated"
            for score in scores
        ]
