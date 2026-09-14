from __future__ import annotations

import socket
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast

from binoculars.detector import (
    BINOCULARS_ACCURACY_THRESHOLD,
    BINOCULARS_FPR_THRESHOLD,
    Binoculars,
)
from binoculars.mcp_server import AnalyzeTextInput, binoculars_analyze_text, binoculars_status
from binoculars.profiles import QWEN_05B, get_profile
from binoculars.utils import require_local_model_path


def test_model_identifier_is_rejected() -> None:
    with pytest.raises(FileNotFoundError):
        require_local_model_path("tiiuae/falcon-7b", label="observer model")


def test_network_connections_are_blocked() -> None:
    with pytest.raises(OSError, match="Network access is disabled"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)


def test_status_reports_offline_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BINOCULARS_OBSERVER_MODEL", raising=False)
    monkeypatch.delenv("BINOCULARS_PERFORMER_MODEL", raising=False)
    result = binoculars_status()
    assert result["ready"] is False
    assert result["transport"] == "stdio"
    assert result["network_guard_verified"] is True
    assert result["profile"] == "qwen2.5-0.5b"


def test_compact_profile_is_calibrated_and_pinned() -> None:
    profile = get_profile("qwen2.5-0.5b")
    assert profile is QWEN_05B
    assert profile.max_tokens == 256
    assert len(profile.observer_sha256) == 64
    assert len(profile.performer_sha256) == 64
    assert profile.low_fpr_threshold < profile.accuracy_threshold


def test_analysis_tool_uses_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDetector:
        threshold = BINOCULARS_FPR_THRESHOLD

        def change_mode(self, mode: str) -> None:
            self.threshold = (
                BINOCULARS_ACCURACY_THRESHOLD if mode == "accuracy" else BINOCULARS_FPR_THRESHOLD
            )

        def compute_score(self, text: str) -> float:
            assert text == "A sufficiently clear local test sample."
            return 0.5

    from binoculars import mcp_server

    monkeypatch.setattr(mcp_server, "_DETECTOR", FakeDetector())
    result = binoculars_analyze_text(
        AnalyzeTextInput(text="A sufficiently clear local test sample.", mode="accuracy")
    )
    assert result["label"] == "Most likely AI-generated"
    assert result["threshold"] == BINOCULARS_ACCURACY_THRESHOLD


def _create_tiny_local_model(path: Path, *, seed: int) -> None:
    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[EOS]": 2,
        "local": 3,
        "only": 4,
        "text": 5,
        "analysis": 6,
        "works": 7,
        ".": 8,
    }
    raw_tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(path)

    import torch

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model.save_pretrained(path)


def test_real_scoring_path_with_local_model_files(tmp_path: Path) -> None:
    observer = tmp_path / "observer"
    performer = tmp_path / "performer"
    observer.mkdir()
    performer.mkdir()
    _create_tiny_local_model(observer, seed=1)
    _create_tiny_local_model(performer, seed=2)

    detector = Binoculars(
        observer,
        performer,
        device="cpu",
        max_token_observed=32,
        accuracy_threshold=0.95,
        low_fpr_threshold=0.90,
    )
    score = detector.compute_score("local only text analysis works .")
    assert isinstance(score, float)
    assert score > 0
    detector.change_mode("accuracy")
    assert detector.threshold == 0.95
