import os
import struct
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from whisper.decoding import DecodingOptions, DecodingResult

# Support running this file directly without installing the package.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from whisper_streaming import StreamConfig, StreamTranscriber
from whisper_streaming.rescoring import RescoreCandidate, RescoreSelectionMeta, select_best_candidate


def _write_silence_wav(path: str, seconds: float = 1.2, sr: int = 16000) -> None:
    n = int(seconds * sr)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = b"".join(struct.pack("<h", 0) for _ in range(n))
        wf.writeframes(frames)


class FakeBasicModel:
    def __init__(self, scripted_results: List[List[Dict[str, Any]]]) -> None:
        self.scripted_results = scripted_results
        self.calls = 0

    def transcribe(self, *_args, **_kwargs) -> Dict[str, Any]:
        if self.calls >= len(self.scripted_results):
            raise AssertionError("Unexpected extra transcribe call.")
        segments = self.scripted_results[self.calls]
        self.calls += 1
        return {"segments": segments}


class FakeDecodeModel:
    def __init__(self) -> None:
        self.decode_calls = 0

    def decode(self, _mel, options: DecodingOptions):
        self.decode_calls += 1
        return DecodingResult(
            audio_features=torch.zeros((1, 1), dtype=torch.float32),
            language=options.language or "en",
            tokens=[1, 2],
            text="baseline",
            avg_logprob=-0.5,
            no_speech_prob=0.0,
            temperature=float(options.temperature),
            compression_ratio=1.0,
        )

    def transcribe(self, _chunk, **kwargs) -> Dict[str, Any]:
        options = DecodingOptions(
            task=kwargs.get("task", "transcribe"),
            language=kwargs.get("language"),
            temperature=float(kwargs.get("temperature", 0.0)),
            beam_size=kwargs.get("beam_size"),
            patience=kwargs.get("patience"),
            length_penalty=kwargs.get("length_penalty"),
            fp16=False,
        )
        decode_result = self.decode(torch.zeros((80, 3000), dtype=torch.float32), options)
        return {"segments": [{"start": 0.0, "end": 1.0, "text": decode_result.text}]}


class StubScorer:
    def score(self, text: str, normalize_text: bool = True) -> float:
        _ = normalize_text
        if "preferred" in text:
            return 10.0
        return -1.0


def test_select_best_candidate_prefers_lm_hypothesis():
    candidates = [
        RescoreCandidate(text="weak baseline", tokens=[1, 2], sum_logprob=-0.2, avg_logprob=-0.1, baseline_score=-0.2),
        RescoreCandidate(text="preferred phrase", tokens=[3, 4], sum_logprob=-0.25, avg_logprob=-0.12, baseline_score=-0.25),
    ]

    selected, meta = select_best_candidate(
        candidates=candidates,
        scorer=StubScorer(),
        alpha=0.1,
        beta=0.0,
        normalize_text=True,
    )

    assert selected.text == "preferred phrase"
    assert meta.applied
    assert meta.changed_from_baseline
    assert meta.candidate_count == 2


def test_rescore_disabled_keeps_stats_zero():
    scripted = [
        [{"start": 0.0, "end": 1.0, "text": " Hello"}],
        [{"start": 0.0, "end": 2.0, "text": " Hello world"}],
    ]
    model = FakeBasicModel(scripted)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "short.wav")
        _write_silence_wav(wav_path, seconds=2.0)

        cfg = StreamConfig(step_sec=1.0, window_sec=2.0, commit_lag_sec=0.0, sleep_to_simulate_realtime=False)
        st = StreamTranscriber(model, cfg)

        done_evt = None
        for evt in st.stream_file(wav_path):
            if evt["type"] == "done":
                done_evt = evt
                break

        assert done_evt is not None
        assert done_evt["rescore_decode_calls"] == 0
        assert done_evt["rescore_applied"] == 0
        assert done_evt["rescore_changed_hypothesis"] == 0
        assert done_evt["rescore_candidate_total"] == 0
        assert done_evt["rescore_candidate_avg"] == 0.0


def test_rescore_hard_fails_when_lm_path_missing():
    model = FakeBasicModel([[{"start": 0.0, "end": 1.0, "text": " Hello"}]])
    cfg = StreamConfig(
        step_sec=1.0,
        window_sec=2.0,
        commit_lag_sec=0.0,
        lm_rescore_enabled=True,
        task="transcribe",
        sleep_to_simulate_realtime=False,
    )
    st = StreamTranscriber(model, cfg)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "short.wav")
        _write_silence_wav(wav_path, seconds=1.0)
        with pytest.raises(RuntimeError, match="LM path|lm_path"):
            list(st.stream_file(wav_path))


def test_rescore_hard_fails_when_lm_dependency_load_fails(monkeypatch):
    model = FakeBasicModel([[{"start": 0.0, "end": 1.0, "text": " Hello"}]])

    with tempfile.TemporaryDirectory() as td:
        lm_path = os.path.join(td, "fake.arpa")
        Path(lm_path).write_text("\\data\\\n")
        wav_path = os.path.join(td, "short.wav")
        _write_silence_wav(wav_path, seconds=1.0)

        cfg = StreamConfig(
            step_sec=1.0,
            window_sec=2.0,
            commit_lag_sec=0.0,
            lm_rescore_enabled=True,
            lm_path=lm_path,
            task="transcribe",
            sleep_to_simulate_realtime=False,
        )
        st = StreamTranscriber(model, cfg)
        monkeypatch.setattr(
            st,
            "_ensure_lm_scorer",
            lambda: (_ for _ in ()).throw(RuntimeError("KenLM dependency is missing")),
        )

        with pytest.raises(RuntimeError, match="KenLM dependency is missing"):
            list(st.stream_file(wav_path))


def test_rescore_is_bypassed_for_translate_task():
    scripted = [
        [{"start": 0.0, "end": 1.0, "text": " Hello"}],
        [{"start": 0.0, "end": 2.0, "text": " Hello world"}],
    ]
    model = FakeBasicModel(scripted)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "short.wav")
        _write_silence_wav(wav_path, seconds=2.0)

        cfg = StreamConfig(
            step_sec=1.0,
            window_sec=2.0,
            commit_lag_sec=0.0,
            task="translate",
            lm_rescore_enabled=True,
            lm_path=None,
            sleep_to_simulate_realtime=False,
        )
        st = StreamTranscriber(model, cfg)

        done_evt = None
        for evt in st.stream_file(wav_path):
            if evt["type"] == "done":
                done_evt = evt
                break

        assert done_evt is not None
        assert done_evt["rescore_applied"] == 0
        assert done_evt["rescore_decode_calls"] == 0


def test_rescore_stats_increment_and_reported(monkeypatch):
    model = FakeDecodeModel()

    with tempfile.TemporaryDirectory() as td:
        lm_path = os.path.join(td, "fake.arpa")
        Path(lm_path).write_text("\\data\\\n")
        wav_path = os.path.join(td, "short.wav")
        _write_silence_wav(wav_path, seconds=2.0)

        cfg = StreamConfig(
            step_sec=1.0,
            window_sec=2.0,
            commit_lag_sec=0.0,
            lm_rescore_enabled=True,
            lm_path=lm_path,
            task="transcribe",
            sleep_to_simulate_realtime=False,
        )
        st = StreamTranscriber(model, cfg)

        monkeypatch.setattr(st, "_ensure_lm_scorer", lambda: object())

        def _fake_rescore(model, mel, options, original_decode, scorer):
            _ = (model, mel, original_decode, scorer)
            result = DecodingResult(
                audio_features=torch.zeros((1, 1), dtype=torch.float32),
                language=options.language or "en",
                tokens=[9, 9],
                text="rescored",
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                temperature=float(options.temperature),
                compression_ratio=1.0,
            )
            return result, RescoreSelectionMeta(
                applied=True,
                changed_from_baseline=True,
                candidate_count=5,
                baseline_index=0,
                selected_index=1,
            )

        monkeypatch.setattr(st, "_rescore_decode_result", _fake_rescore)

        done_evt = None
        for evt in st.stream_file(wav_path):
            if evt["type"] == "done":
                done_evt = evt
                break

        assert done_evt is not None
        assert done_evt["rescore_decode_calls"] > 0
        assert done_evt["rescore_applied"] == done_evt["rescore_decode_calls"]
        assert done_evt["rescore_changed_hypothesis"] == done_evt["rescore_applied"]
        assert done_evt["rescore_candidate_total"] == done_evt["rescore_applied"] * 5
        assert done_evt["rescore_candidate_avg"] == pytest.approx(5.0)
