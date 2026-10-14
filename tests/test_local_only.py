from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast

from binoculars.config import RuntimeConfig
from binoculars.detector import (
    BINOCULARS_ACCURACY_THRESHOLD,
    Analysis,
    Binoculars,
)
from binoculars.integrity import verify_model_directory
from binoculars.mcp_server import AnalyzeTextInput, binoculars_analyze_text, binoculars_status
from binoculars.profiles import QWEN_05B, ModelFile, get_profile
from binoculars.runtime import DetectorRuntime, RuntimeAnalysis
from binoculars.utils import require_local_model_path


def test_model_identifier_is_rejected() -> None:
    with pytest.raises(FileNotFoundError):
        require_local_model_path("tiiuae/falcon-7b", label="observer model")


def test_network_connections_are_blocked() -> None:
    with pytest.raises(OSError, match="Network access is disabled"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        with pytest.raises(OSError, match="Network access is disabled"):
            udp.sendto(b"blocked", ("127.0.0.1", 9))
    if hasattr(socket.socket, "sendmsg"):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            with pytest.raises(OSError, match="Network access is disabled"):
                udp.sendmsg([b"blocked"], [], 0, ("127.0.0.1", 9))


def test_offline_environment_precedes_transformers_import() -> None:
    repository = Path(__file__).resolve().parents[1]
    script = """
import builtins
import os

original_import = builtins.__import__
import binoculars
assert "torch" not in __import__("sys").modules
assert "transformers" not in __import__("sys").modules
def checked_import(name, *args, **kwargs):
    if name == "transformers" or name.startswith("transformers."):
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from binoculars import Binoculars
"""
    subprocess.run(  # noqa: S603 -- executable is the active test interpreter.
        [sys.executable, "-c", script],
        cwd=repository,
        check=True,
        env={key: value for key, value in os.environ.items() if "OFFLINE" not in key},
    )


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
    assert len(profile.observer_files) == 7
    assert len(profile.performer_files) == 7


def test_runtime_config_is_immutable_and_bounded() -> None:
    config = RuntimeConfig.from_environment(
        {"BINOCULARS_DEVICE": "cpu", "BINOCULARS_LOCK_TIMEOUT_SECONDS": "0.5"}
    )
    assert config.device == "cpu"
    assert config.lock_timeout_seconds == 0.5
    with pytest.raises(AttributeError):
        config.device = "mps"  # type: ignore[misc]
    with pytest.raises(ValueError, match="between 0.1 and 3600"):
        RuntimeConfig.from_environment({"BINOCULARS_LOCK_TIMEOUT_SECONDS": "0"})
    with pytest.raises(ValueError, match="BINOCULARS_DEVICE"):
        RuntimeConfig.from_environment({"BINOCULARS_DEVICE": "metal"})


def test_runtime_lock_wait_is_bounded() -> None:
    runtime = DetectorRuntime(
        RuntimeConfig.from_environment({"BINOCULARS_LOCK_TIMEOUT_SECONDS": "0.1"})
    )
    runtime._lock.acquire()
    try:
        with pytest.raises(TimeoutError, match="remained busy"):
            runtime.analyze("long enough local input text for the detector", "low-fpr")
    finally:
        runtime._lock.release()


def test_manifest_cache_invalidates_after_file_change(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    artifact = model / "config.json"
    artifact.write_bytes(b"first")
    expected = hashlib.sha256(b"first").hexdigest()
    manifest = (ModelFile("config.json", expected),)
    assert verify_model_directory(model, manifest).verified is True

    artifact.write_bytes(b"second-value")
    result = verify_model_directory(model, manifest)
    assert result.verified is False
    assert result.error == "hash mismatch: config.json"


def test_analysis_tool_uses_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRuntime:
        def analyze(self, text: str, mode: str) -> RuntimeAnalysis:
            assert text == "A sufficiently clear local test sample."
            assert mode == "accuracy"
            return RuntimeAnalysis(
                analysis=Analysis(0.5, input_tokens=11, tokens_analyzed=11, truncated=False),
                threshold=BINOCULARS_ACCURACY_THRESHOLD,
                profile=QWEN_05B,
                device="cpu",
                characters_analyzed=len(text),
            )

    from binoculars import mcp_server

    monkeypatch.setattr(mcp_server, "_RUNTIME", FakeRuntime())
    result = binoculars_analyze_text(
        AnalyzeTextInput(text="A sufficiently clear local test sample.", mode="accuracy")
    )
    assert result["label"] == "Most likely AI-generated"
    assert result["threshold"] == BINOCULARS_ACCURACY_THRESHOLD
    assert result["input_tokens"] == 11
    assert result["tokens_analyzed"] == 11
    assert result["truncated"] is False
    assert result["characters_analyzed"] == 39


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
    text = "local only text analysis works . local only text analysis works ."
    score = detector.compute_score(text)
    assert isinstance(score, float)
    assert score > 0
    analysis = detector.analyze(text)
    assert isinstance(analysis, Analysis)
    assert analysis.input_tokens == analysis.tokens_analyzed
    assert analysis.truncated is False
    long_analysis = detector.analyze(" ".join(["local"] * 64))
    assert isinstance(long_analysis, Analysis)
    assert long_analysis.input_tokens > long_analysis.tokens_analyzed
    assert long_analysis.tokens_analyzed == 32
    assert long_analysis.truncated is True
    detector.change_mode("accuracy")
    assert detector.threshold == 0.95


def test_short_text_and_non_finite_scores_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observer = tmp_path / "observer"
    performer = tmp_path / "performer"
    observer.mkdir()
    performer.mkdir()
    _create_tiny_local_model(observer, seed=1)
    _create_tiny_local_model(performer, seed=2)
    detector = Binoculars(observer, performer, device="cpu", max_token_observed=32)

    with pytest.raises(ValueError, match="fewer than 8 analyzed tokens"):
        detector.analyze("local text")

    from binoculars import detector as detector_module

    monkeypatch.setattr(
        detector_module,
        "perplexity",
        lambda encoding, logits: __import__("torch").tensor([float("nan")]),
    )
    with pytest.raises(ValueError, match="non-finite intermediate metrics"):
        detector.analyze("local only text analysis works . local only text analysis works .")


def test_invalid_device_is_rejected_before_model_loading(tmp_path: Path) -> None:
    observer = tmp_path / "observer"
    performer = tmp_path / "performer"
    observer.mkdir()
    performer.mkdir()
    with pytest.raises(ValueError, match="device must be"):
        Binoculars(observer, performer, device="metal")
    with pytest.raises(ValueError, match="thresholds must be finite"):
        Binoculars(observer, performer, device="cpu", accuracy_threshold=float("nan"))


def test_input_limit_is_enforced() -> None:
    with pytest.raises(ValidationError):
        AnalyzeTextInput(text="x" * 100_001)
