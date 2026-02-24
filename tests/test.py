import wave
import struct
import tempfile
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import whisper

# Support running this file directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from whisper_streaming import StreamConfig, StreamTranscriber
from whisper_streaming.core import _trim_text_overlap
from whisper_streaming.sources import AudioSource, FileAudioSource, RecordingAudioSource


def _write_silence_wav(path: str, seconds: float = 1.2, sr: int = 16000) -> None:
    n = int(seconds * sr)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sr)
        # silence samples
        frames = b"".join(struct.pack("<h", 0) for _ in range(n))
        wf.writeframes(frames)


class FakeModel:
    def __init__(self, scripted_results: List[List[Dict[str, Any]]]) -> None:
        self.scripted_results = scripted_results
        self.calls = 0
        self.kwargs_history: List[Dict[str, Any]] = []

    def transcribe(self, *_args, **kwargs) -> Dict[str, Any]:
        if self.calls >= len(self.scripted_results):
            raise AssertionError("Unexpected extra transcribe call; stream did not terminate as expected.")
        self.kwargs_history.append(dict(kwargs))
        segments = self.scripted_results[self.calls]
        self.calls += 1
        return {"segments": segments}


class FakeAudioSource(AudioSource):
    def __init__(self, chunks: List[np.ndarray], sample_rate: int = 16000) -> None:
        super().__init__(sample_rate=sample_rate, chunk_sec=1.0)
        self._chunks = chunks

    def chunks(self):
        for chunk in self._chunks:
            yield chunk


class FakeVAD:
    def __init__(self, scripted_spans: List[List[Dict[str, Any]]]) -> None:
        self.scripted_spans = scripted_spans
        self.calls = 0

    def speech_timestamps(self, _audio: np.ndarray, _sample_rate: int) -> List[Dict[str, Any]]:
        idx = min(self.calls, max(0, len(self.scripted_spans) - 1))
        spans = self.scripted_spans[idx] if self.scripted_spans else []
        self.calls += 1
        return spans


@pytest.mark.integration
def test_stream_file_emits_done_and_monotonic_commits():
    if os.getenv("WHISPER_INTEGRATION") != "1":
        pytest.skip("Set WHISPER_INTEGRATION=1 to run integration tests.")

    model = whisper.load_model("tiny.en")

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "silence.wav")
        _write_silence_wav(wav_path)

        cfg = StreamConfig(step_sec=0.4, window_sec=1.0, commit_lag_sec=0.3, sleep_to_simulate_realtime=False)
        st = StreamTranscriber(model, cfg)

        last_committed = ""
        got_done = False

        for evt in st.stream_file(wav_path):
            if evt["type"] == "final":
                committed = evt["committed_text"]
                # monotonic: committed text should not shrink
                assert len(committed) >= len(last_committed)
                last_committed = committed
            if evt["type"] == "done":
                got_done = True

        assert got_done


def test_stream_file_trims_overlap_in_commits_and_terminates():
    scripted = [
        [{"start": 0.0, "end": 1.0, "text": " Hello world."}],
        [{"start": 0.0, "end": 2.0, "text": " Hello world. More text."}],
    ]
    model = FakeModel(scripted)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "short.wav")
        _write_silence_wav(wav_path, seconds=2.0)

        cfg = StreamConfig(step_sec=1.0, window_sec=2.0, commit_lag_sec=0.0, sleep_to_simulate_realtime=False)
        st = StreamTranscriber(model, cfg)

        final_text = None
        for evt in st.stream_file(wav_path):
            if evt["type"] == "done":
                final_text = evt["text"]
                break

        assert final_text == "Hello world. More text."
        assert "Hello world. Hello world." not in final_text
        assert model.calls == 2


def test_stream_file_flushes_tail_on_done():
    scripted = [
        [{"start": 0.0, "end": 1.0, "text": " Hello"}],
        [{"start": 0.0, "end": 2.0, "text": " Hello world"}],
        [{"start": 0.0, "end": 2.0, "text": " Hello world"}],
    ]
    model = FakeModel(scripted)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "short.wav")
        _write_silence_wav(wav_path, seconds=2.0)

        cfg = StreamConfig(step_sec=1.0, window_sec=2.0, commit_lag_sec=0.5, sleep_to_simulate_realtime=False)
        st = StreamTranscriber(model, cfg)

        done_evt = None
        for evt in st.stream_file(wav_path):
            if evt["type"] == "done":
                done_evt = evt
                break

        assert done_evt is not None
        assert done_evt["text"] == "Hello world"
        assert model.calls == 3


def test_stream_file_does_not_advance_time_when_trim_is_empty():
    # If a segment is fully overlapped text, it should not advance committed_time;
    # otherwise later segments with slightly earlier end times can get skipped.
    scripted = [
        [{"start": 0.0, "end": 1.0, "text": " A"}],
        [{"start": 0.0, "end": 2.0, "text": " A"}],  # overlap-only duplicate
        [{"start": 0.0, "end": 1.5, "text": " A B"}],  # arrives in final flush window
    ]
    model = FakeModel(scripted)

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
        assert done_evt["text"] == "A B"
        assert model.calls == 3


def test_stream_file_skips_small_time_jitter_commits():
    scripted = [
        [{"start": 0.0, "end": 1.0, "text": " Hello"}],
        [{"start": 0.0, "end": 1.1, "text": " Yellow"}],  # tiny time advance; should be ignored
        [{"start": 0.0, "end": 2.0, "text": " Hello world"}],
    ]
    model = FakeModel(scripted)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "short.wav")
        _write_silence_wav(wav_path, seconds=2.0)

        cfg = StreamConfig(
            step_sec=1.0,
            window_sec=2.0,
            commit_lag_sec=0.0,
            min_commit_advance_sec=0.35,
            sleep_to_simulate_realtime=False,
        )
        st = StreamTranscriber(model, cfg)

        done_evt = None
        for evt in st.stream_file(wav_path):
            if evt["type"] == "done":
                done_evt = evt
                break

        assert done_evt is not None
        assert done_evt["text"] == "Hello world"
        assert model.calls == 3


def test_stream_file_forwards_beam_decode_options():
    scripted = [
        [{"start": 0.0, "end": 1.0, "text": " Hello"}],
        [{"start": 0.0, "end": 2.0, "text": " Hello world"}],
        [{"start": 0.0, "end": 2.0, "text": " Hello world"}],
    ]
    model = FakeModel(scripted)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "short.wav")
        _write_silence_wav(wav_path, seconds=2.0)

        cfg = StreamConfig(
            step_sec=1.0,
            window_sec=2.0,
            commit_lag_sec=0.0,
            beam_size=8,
            beam_patience=1.2,
            beam_length_penalty=0.4,
            sleep_to_simulate_realtime=False,
        )
        st = StreamTranscriber(model, cfg)

        done_evt = None
        for evt in st.stream_file(wav_path):
            if evt["type"] == "done":
                done_evt = evt
                break

        assert done_evt is not None
        assert model.kwargs_history
        assert all(kwargs.get("beam_size") == 8 for kwargs in model.kwargs_history)
        assert all(kwargs.get("patience") == 1.2 for kwargs in model.kwargs_history)
        assert all(kwargs.get("length_penalty") == 0.4 for kwargs in model.kwargs_history)


def test_stream_file_silence_rms_gating_skips_decode():
    model = FakeModel([])

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "silence.wav")
        _write_silence_wav(wav_path, seconds=2.0)

        cfg = StreamConfig(
            step_sec=1.0,
            window_sec=2.0,
            commit_lag_sec=0.0,
            silence_rms_threshold=0.01,
            sleep_to_simulate_realtime=False,
        )
        st = StreamTranscriber(model, cfg)

        done_evt = None
        for evt in st.stream_file(wav_path):
            if evt["type"] == "done":
                done_evt = evt
                break

        assert done_evt is not None
        assert done_evt["text"] == ""
        assert model.calls == 0
        assert done_evt["decode_performed"] == 0
        assert done_evt["decode_skipped_silence"] >= 1


def test_stream_file_vad_gating_skips_no_speech_ticks():
    scripted = [
        [{"start": 0.0, "end": 2.0, "text": " Hello there."}],
    ]
    model = FakeModel(scripted)
    vad = FakeVAD(
        scripted_spans=[
            [],
            [{"start": 4000, "end": 28000}],
        ]
    )

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "short.wav")
        _write_silence_wav(wav_path, seconds=2.0)

        cfg = StreamConfig(
            step_sec=1.0,
            window_sec=2.0,
            commit_lag_sec=0.0,
            vad_enabled=True,
            sleep_to_simulate_realtime=False,
        )
        st = StreamTranscriber(model, cfg, vad_detector=vad)

        done_evt = None
        for evt in st.stream_file(wav_path):
            if evt["type"] == "done":
                done_evt = evt
                break

        assert done_evt is not None
        assert done_evt["text"] == "Hello there."
        assert model.calls == 1
        assert done_evt["decode_performed"] == 1
        assert done_evt["decode_skipped_vad"] >= 1


def test_stream_file_vad_adaptive_bypass_reduces_vad_calls():
    scripted = [[], [], [], [], []]
    model = FakeModel(scripted)
    vad = FakeVAD(
        scripted_spans=[
            [{"start": 1000, "end": 32000}],
        ]
    )

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "short.wav")
        _write_silence_wav(wav_path, seconds=4.0)

        cfg = StreamConfig(
            step_sec=1.0,
            window_sec=2.0,
            commit_lag_sec=0.0,
            vad_enabled=True,
            vad_adaptive_disable_ticks=2,
            vad_recheck_interval_ticks=3,
            sleep_to_simulate_realtime=False,
        )
        st = StreamTranscriber(model, cfg, vad_detector=vad)

        done_evt = None
        for evt in st.stream_file(wav_path):
            if evt["type"] == "done":
                done_evt = evt
                break

        assert done_evt is not None
        assert model.calls == 5
        assert vad.calls == 2
        assert done_evt["decode_vad_bypassed"] == 3
        assert done_evt["decode_skipped_vad"] == 0


def test_trim_overlap_handles_punctuation_drift():
    existing = "What happens if an immovable object meets an unstoppable force?"
    candidate = "What happens if an immovable object meets an unstoppable force, is a popular question on the internet."
    trimmed = _trim_text_overlap(existing, candidate)
    assert trimmed == "is a popular question on the internet."


def test_stream_dispatch_uses_existing_file_path_logic():
    scripted = [
        [{"start": 0.0, "end": 1.0, "text": " Hello world."}],
        [{"start": 0.0, "end": 2.0, "text": " Hello world. More text."}],
    ]
    model = FakeModel(scripted)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "short.wav")
        _write_silence_wav(wav_path, seconds=2.0)
        source = FileAudioSource(wav_path, chunk_sec=0.2)

        cfg = StreamConfig(step_sec=1.0, window_sec=2.0, commit_lag_sec=0.0, sleep_to_simulate_realtime=False)
        st = StreamTranscriber(model, cfg)

        done_evt = None
        for evt in st.stream(source):
            if evt["type"] == "done":
                done_evt = evt
                break

        assert done_evt is not None
        assert done_evt["text"] == "Hello world. More text."
        assert model.calls == 2


def test_stream_source_accepts_incremental_audio_chunks():
    scripted = [
        [{"start": 0.0, "end": 1.0, "text": " Hello world."}],
        [{"start": 0.0, "end": 2.0, "text": " Hello world. More text."}],
    ]
    model = FakeModel(scripted)
    source = FakeAudioSource(
        chunks=[
            np.zeros(16000, dtype=np.float32),
            np.zeros(16000, dtype=np.float32),
        ]
    )

    cfg = StreamConfig(step_sec=1.0, window_sec=2.0, commit_lag_sec=0.0, sleep_to_simulate_realtime=False)
    st = StreamTranscriber(model, cfg)

    done_evt = None
    for evt in st.stream_source(source):
        if evt["type"] == "done":
            done_evt = evt
            break

    assert done_evt is not None
    assert done_evt["text"] == "Hello world. More text."
    assert model.calls == 2


def test_recording_audio_source_captures_and_writes_wav():
    source = FakeAudioSource(
        chunks=[
            np.ones(16000, dtype=np.float32) * 0.25,
            np.ones(16000, dtype=np.float32) * -0.25,
        ]
    )
    recorder = RecordingAudioSource(source)

    for _ in recorder.chunks():
        pass

    assert recorder.has_audio
    assert recorder.recorded_seconds == 2.0
    assert recorder.recorded_peak() > 0.0
    assert recorder.recorded_rms() > 0.0

    with tempfile.TemporaryDirectory() as td:
        out_wav = os.path.join(td, "capture.wav")
        recorder.save_wav(out_wav, normalize=True)

        with wave.open(out_wav, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 32000
