from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional
import time
import warnings
import contextlib
import os
import sys
import numpy as np
import whisper

from .sources import AudioSource, FileAudioSource

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")
_EPS = 1e-3


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old

@dataclass(frozen=True)
class StreamConfig:
    # Streaming behavior
    step_sec: float = 0.5
    window_sec: float = 8.0
    commit_lag_sec: float = 1.0
    min_chunk_sec: float = 0.25
    min_commit_advance_sec: float = 0.35

    # Whisper decode options
    fp16: bool = False
    language: Optional[str] = None
    task: str = "transcribe"  # "translate" also valid
    temperature: float = 0.0
    condition_on_previous_text: bool = True
    use_initial_prompt: bool = False

    # For demo/simulation
    sleep_to_simulate_realtime: bool = False


def _join_clean(parts: List[str]) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return " ".join(cleaned)


def _normalize_token(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalnum() or ch == "'")


def _trim_text_overlap(existing: str, candidate: str, lookback_chars: int = 400) -> str:
    """
    Remove overlap where `candidate` starts with text already at the tail of `existing`.
    """
    existing_clean = _join_clean([existing])
    candidate_clean = _join_clean([candidate])

    if not candidate_clean:
        return ""
    if not existing_clean:
        return candidate_clean

    existing_words = [w for w in existing_clean.split() if _normalize_token(w)]
    candidate_words = [w for w in candidate_clean.split() if _normalize_token(w)]
    if not candidate_words:
        return ""

    existing_norm = [_normalize_token(w) for w in existing_words]
    candidate_norm = [_normalize_token(w) for w in candidate_words]

    lookback_tokens = min(max(1, lookback_chars // 4), len(existing_norm))
    existing_tail = existing_norm[-lookback_tokens:]

    # Candidate entirely repeated near the tail.
    if len(candidate_norm) <= len(existing_tail):
        for i in range(0, len(existing_tail) - len(candidate_norm) + 1):
            if existing_tail[i : i + len(candidate_norm)] == candidate_norm:
                return ""

    # Token-based suffix/prefix overlap is robust to punctuation drift.
    max_overlap = min(len(existing_tail), len(candidate_norm))
    for n in range(max_overlap, 0, -1):
        if existing_tail[-n:] == candidate_norm[:n]:
            return " ".join(candidate_words[n:]).strip()

    return " ".join(candidate_words).strip()


class StreamTranscriber:
    """
    Rolling-window streamer for Whisper.

    Yields events:
      - type="partial": display text may change (committed + draft)
      - type="final": committed text advanced (monotonic)
      - type="done": final committed transcript
    """

    def __init__(
        self,
        model: "whisper.Whisper",
        config: StreamConfig = StreamConfig(),
        on_partial: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_final: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.on_partial = on_partial
        self.on_final = on_final
        self._stop_flag = False

    def stop(self) -> None:
        """Request a graceful stop (returns committed transcript in 'done')."""
        self._stop_flag = True

    def reset(self) -> None:
        self._stop_flag = False

    def stream(self, source: AudioSource) -> Iterator[Dict[str, Any]]:
        """
        Stream transcription from an AudioSource.

        File sources reuse stream_file() so existing file behavior stays unchanged.
        """
        if isinstance(source, FileAudioSource):
            return self.stream_file(source.audio_path)
        return self.stream_source(source)

    def stream_file(self, audio_path: str) -> Iterator[Dict[str, Any]]:
        cfg = self.config
        self._stop_flag = False

        audio = whisper.load_audio(audio_path)
        sr = whisper.audio.SAMPLE_RATE
        total_sec = len(audio) / sr

        committed_time = 0.0
        committed_parts: List[str] = []

        last_display = ""
        t = 0.0

        min_chunk_samples = int(cfg.min_chunk_sec * sr)

        def _decode_at(end_t: float, prompt: str) -> tuple[float, List[Dict[str, Any]]]:
            window_start = max(0.0, end_t - cfg.window_sec)
            i0 = int(window_start * sr)
            i1 = int(end_t * sr)
            chunk = audio[i0:i1]

            if len(chunk) < min_chunk_samples:
                return window_start, []

            with suppress_stderr():
                result = self.model.transcribe(
                    chunk,
                    fp16=cfg.fp16,
                    verbose=False,
                    language=cfg.language,
                    task=cfg.task,
                    temperature=cfg.temperature,
                    initial_prompt=prompt if (cfg.use_initial_prompt and prompt) else None,
                    condition_on_previous_text=cfg.condition_on_previous_text,
                )

            return window_start, result.get("segments", [])

        try:
            while t < total_sec - _EPS and not self._stop_flag:
                t = min(total_sec, t + cfg.step_sec)
                safe_until = max(0.0, t - cfg.commit_lag_sec)
                prompt = _join_clean(committed_parts)
                window_start, segments = _decode_at(t, prompt)

                new_commits: List[str] = []
                draft_parts: List[str] = []

                for seg in segments:
                    # seg times are relative to chunk; convert to absolute
                    abs_end = window_start + float(seg["end"])
                    seg_text = seg.get("text", "")

                    # Ignore anything already committed in time.
                    if abs_end <= committed_time + _EPS:
                        continue

                    commit_base = _join_clean(committed_parts + new_commits)
                    if abs_end <= safe_until + _EPS:
                        trimmed = _trim_text_overlap(commit_base, seg_text)
                        if trimmed:
                            if committed_time > 0.0 and (abs_end - committed_time) < cfg.min_commit_advance_sec:
                                continue
                            new_commits.append(trimmed)
                            committed_time = max(committed_time, abs_end)
                    else:
                        if abs_end > committed_time + _EPS:
                            draft_base = _join_clean(committed_parts + new_commits + draft_parts)
                            trimmed = _trim_text_overlap(draft_base, seg_text)
                            if trimmed:
                                draft_parts.append(trimmed)

                committed_added = False
                if new_commits:
                    committed_parts.extend(new_commits)
                    committed_added = True

                committed_text = _join_clean(committed_parts)
                draft_text = _join_clean(draft_parts)
                display_text = _join_clean([committed_text, draft_text])

                if display_text != last_display:
                    evt = {
                        "type": "partial",
                        "t": t,
                        "text": display_text,
                        "committed_text": committed_text,
                        "draft_text": draft_text,
                        "committed_until": committed_time,
                    }
                    yield evt
                    if self.on_partial:
                        self.on_partial(evt)
                    last_display = display_text

                if committed_added:
                    evt = {
                        "type": "final",
                        "t": t,
                        "text": committed_text,
                        "committed_text": committed_text,
                        "draft_text": draft_text,
                        "committed_until": committed_time,
                    }
                    yield evt
                    if self.on_final:
                        self.on_final(evt)

                if cfg.sleep_to_simulate_realtime:
                    time.sleep(cfg.step_sec)

        except KeyboardInterrupt:
            # Graceful Ctrl+C: fall through and emit done()
            pass

        # Flush remaining tail so done() includes the latest draft portion.
        flush_t = min(t, total_sec)
        if flush_t > 0.0 and committed_time < flush_t - _EPS:
            prompt = _join_clean(committed_parts)
            window_start, segments = _decode_at(flush_t, prompt)

            new_commits: List[str] = []
            for seg in segments:
                abs_end = window_start + float(seg["end"])
                if abs_end <= committed_time + _EPS:
                    continue
                trimmed = _trim_text_overlap(_join_clean(committed_parts + new_commits), seg.get("text", ""))
                if trimmed:
                    new_commits.append(trimmed)
                    committed_time = max(committed_time, abs_end)

            if new_commits:
                committed_parts.extend(new_commits)
                committed_text = _join_clean(committed_parts)
                evt = {
                    "type": "final",
                    "t": flush_t,
                    "text": committed_text,
                    "committed_text": committed_text,
                    "draft_text": "",
                    "committed_until": committed_time,
                }
                yield evt
                if self.on_final:
                    self.on_final(evt)

        final_text = _join_clean(committed_parts)
        yield {
            "type": "done",
            "t": min(t, total_sec),
            "text": final_text,
            "committed_text": final_text,
            "draft_text": "",
            "committed_until": committed_time,
        }

    def stream_source(self, source: AudioSource) -> Iterator[Dict[str, Any]]:
        """
        Stream transcription from a live/incremental source (mic, websocket, RTSP, etc.).
        """
        cfg = self.config
        self._stop_flag = False

        sr = source.sample_rate
        if sr != whisper.audio.SAMPLE_RATE:
            raise ValueError(
                f"AudioSource sample_rate={sr} is unsupported; expected {whisper.audio.SAMPLE_RATE}."
            )

        committed_time = 0.0
        committed_parts: List[str] = []
        last_display = ""

        t = 0.0
        next_tick = cfg.step_sec
        audio = np.zeros(0, dtype=np.float32)

        min_chunk_samples = int(cfg.min_chunk_sec * sr)

        def _decode_at(end_t: float, prompt: str) -> tuple[float, List[Dict[str, Any]]]:
            window_start = max(0.0, end_t - cfg.window_sec)
            i0 = int(window_start * sr)
            i1 = int(end_t * sr)
            chunk = audio[i0:i1]

            if len(chunk) < min_chunk_samples:
                return window_start, []

            with suppress_stderr():
                result = self.model.transcribe(
                    chunk,
                    fp16=cfg.fp16,
                    verbose=False,
                    language=cfg.language,
                    task=cfg.task,
                    temperature=cfg.temperature,
                    initial_prompt=prompt if (cfg.use_initial_prompt and prompt) else None,
                    condition_on_previous_text=cfg.condition_on_previous_text,
                )

            return window_start, result.get("segments", [])

        try:
            with source:
                for incoming in source.chunks():
                    if self._stop_flag:
                        break
                    if incoming is None:
                        continue

                    chunk = np.asarray(incoming, dtype=np.float32).reshape(-1)
                    if chunk.size == 0:
                        continue

                    audio = np.concatenate((audio, chunk))
                    total_sec = len(audio) / sr

                    while next_tick <= total_sec + _EPS and not self._stop_flag:
                        t = next_tick
                        safe_until = max(0.0, t - cfg.commit_lag_sec)
                        prompt = _join_clean(committed_parts)
                        window_start, segments = _decode_at(t, prompt)

                        new_commits: List[str] = []
                        draft_parts: List[str] = []

                        for seg in segments:
                            abs_end = window_start + float(seg["end"])
                            seg_text = seg.get("text", "")

                            if abs_end <= committed_time + _EPS:
                                continue

                            commit_base = _join_clean(committed_parts + new_commits)
                            if abs_end <= safe_until + _EPS:
                                trimmed = _trim_text_overlap(commit_base, seg_text)
                                if trimmed:
                                    if committed_time > 0.0 and (abs_end - committed_time) < cfg.min_commit_advance_sec:
                                        continue
                                    new_commits.append(trimmed)
                                    committed_time = max(committed_time, abs_end)
                            else:
                                if abs_end > committed_time + _EPS:
                                    draft_base = _join_clean(committed_parts + new_commits + draft_parts)
                                    trimmed = _trim_text_overlap(draft_base, seg_text)
                                    if trimmed:
                                        draft_parts.append(trimmed)

                        committed_added = False
                        if new_commits:
                            committed_parts.extend(new_commits)
                            committed_added = True

                        committed_text = _join_clean(committed_parts)
                        draft_text = _join_clean(draft_parts)
                        display_text = _join_clean([committed_text, draft_text])

                        if display_text != last_display:
                            evt = {
                                "type": "partial",
                                "t": t,
                                "text": display_text,
                                "committed_text": committed_text,
                                "draft_text": draft_text,
                                "committed_until": committed_time,
                            }
                            yield evt
                            if self.on_partial:
                                self.on_partial(evt)
                            last_display = display_text

                        if committed_added:
                            evt = {
                                "type": "final",
                                "t": t,
                                "text": committed_text,
                                "committed_text": committed_text,
                                "draft_text": draft_text,
                                "committed_until": committed_time,
                            }
                            yield evt
                            if self.on_final:
                                self.on_final(evt)

                        if cfg.sleep_to_simulate_realtime:
                            time.sleep(cfg.step_sec)

                        next_tick += cfg.step_sec

        except KeyboardInterrupt:
            # Graceful Ctrl+C: fall through and emit done()
            pass

        total_sec = len(audio) / sr

        # Flush remaining tail so done() includes the latest draft portion.
        flush_t = total_sec
        if flush_t > 0.0 and committed_time < flush_t - _EPS:
            prompt = _join_clean(committed_parts)
            window_start, segments = _decode_at(flush_t, prompt)

            new_commits: List[str] = []
            for seg in segments:
                abs_end = window_start + float(seg["end"])
                if abs_end <= committed_time + _EPS:
                    continue
                trimmed = _trim_text_overlap(_join_clean(committed_parts + new_commits), seg.get("text", ""))
                if trimmed:
                    new_commits.append(trimmed)
                    committed_time = max(committed_time, abs_end)

            if new_commits:
                committed_parts.extend(new_commits)
                committed_text = _join_clean(committed_parts)
                evt = {
                    "type": "final",
                    "t": flush_t,
                    "text": committed_text,
                    "committed_text": committed_text,
                    "draft_text": "",
                    "committed_until": committed_time,
                }
                yield evt
                if self.on_final:
                    self.on_final(evt)

        final_text = _join_clean(committed_parts)
        yield {
            "type": "done",
            "t": flush_t,
            "text": final_text,
            "committed_text": final_text,
            "draft_text": "",
            "committed_until": committed_time,
        }
