from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol
import time
import warnings
import contextlib
import os
import sys
import types
import numpy as np
import whisper

from .rescoring import KenLMScorer, RescoreSelectionMeta, rescore_decode_result
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
    step_sec: float = 3.0
    window_sec: float = 9.0
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

    # Beam search / LM rescoring (opt-in)
    beam_size: Optional[int] = None
    beam_patience: Optional[float] = None
    beam_length_penalty: Optional[float] = None
    lm_rescore_enabled: bool = False
    lm_path: Optional[str] = None
    lm_alpha: float = 0.25
    lm_beta: float = 0.0
    lm_top_n: Optional[int] = None
    lm_normalize_text: bool = True

    # Decode gating (speed/quality tradeoff)
    vad_enabled: bool = False
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 100
    vad_speech_pad_ms: int = 30
    vad_backend: str = "onnx"  # "onnx" or "torch"
    vad_adaptive_disable_ticks: int = 8
    vad_recheck_interval_ticks: int = 12
    silence_rms_threshold: float = 0.0

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


class VoiceActivityDetector(Protocol):
    def speech_timestamps(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        ...


class SileroVADDetector:
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        speech_pad_ms: int = 30,
        backend: str = "onnx",
    ) -> None:
        try:
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD is not installed. Install with: "
                "python3 -m pip install 'whisper-streaming[vad]' (or pip install silero-vad)."
            ) from exc

        self._torch = torch
        self._get_speech_timestamps = get_speech_timestamps
        self._model = load_silero_vad(onnx=(backend == "onnx"))
        self.threshold = float(threshold)
        self.min_speech_ms = int(min_speech_ms)
        self.min_silence_ms = int(min_silence_ms)
        self.speech_pad_ms = int(speech_pad_ms)
        self.backend = backend

    def speech_timestamps(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        if audio.size == 0:
            return []

        tensor = self._torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        with self._torch.no_grad():
            spans = self._get_speech_timestamps(
                tensor,
                self._model,
                threshold=self.threshold,
                sampling_rate=int(sample_rate),
                min_speech_duration_ms=self.min_speech_ms,
                min_silence_duration_ms=self.min_silence_ms,
                speech_pad_ms=self.speech_pad_ms,
                return_seconds=False,
            )
        return list(spans)


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
        vad_detector: Optional[VoiceActivityDetector] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.on_partial = on_partial
        self.on_final = on_final
        self._vad_detector = vad_detector
        self._lm_scorer: Optional[KenLMScorer] = None
        self._stop_flag = False
        self._decode_stats: Dict[str, Any] = {}
        self._vad_no_skip_streak = 0
        self._vad_bypass_ticks_left = 0
        self._reset_decode_stats()

    def stop(self) -> None:
        """Request a graceful stop (returns committed transcript in 'done')."""
        self._stop_flag = True

    def reset(self) -> None:
        self._stop_flag = False

    def _reset_decode_stats(self) -> None:
        self._decode_stats = {
            "decode_ticks": 0,
            "decode_performed": 0,
            "decode_skipped_short_chunk": 0,
            "decode_skipped_silence": 0,
            "decode_skipped_vad": 0,
            "decode_vad_bypassed": 0,
            "rescore_decode_calls": 0,
            "rescore_applied": 0,
            "rescore_changed_hypothesis": 0,
            "rescore_candidate_total": 0,
        }
        self._vad_no_skip_streak = 0
        self._vad_bypass_ticks_left = 0

    def _final_decode_stats(self) -> Dict[str, Any]:
        stats = dict(self._decode_stats)
        applied = int(stats.get("rescore_applied", 0))
        total = int(stats.get("rescore_candidate_total", 0))
        stats["rescore_candidate_avg"] = (float(total) / float(applied)) if applied > 0 else 0.0
        return stats

    def _ensure_vad_detector(self) -> Optional[VoiceActivityDetector]:
        if not self.config.vad_enabled:
            return None
        if self._vad_detector is None:
            self._vad_detector = SileroVADDetector(
                threshold=self.config.vad_threshold,
                min_speech_ms=self.config.vad_min_speech_ms,
                min_silence_ms=self.config.vad_min_silence_ms,
                speech_pad_ms=self.config.vad_speech_pad_ms,
                backend=self.config.vad_backend,
            )
        return self._vad_detector

    def _effective_beam_size(self) -> Optional[int]:
        if self.config.beam_size is not None:
            return int(self.config.beam_size)
        if self.config.lm_rescore_enabled and self.config.task == "transcribe":
            return 8
        return None

    def _effective_lm_top_n(self) -> int:
        if self.config.lm_top_n is not None:
            return max(1, int(self.config.lm_top_n))
        beam_size = self._effective_beam_size()
        if beam_size is not None:
            return max(1, int(beam_size))
        return 1

    def _lm_rescore_active(self) -> bool:
        return bool(self.config.lm_rescore_enabled and self.config.task == "transcribe")

    def _ensure_lm_scorer(self) -> KenLMScorer:
        if self._lm_scorer is not None:
            return self._lm_scorer

        lm_path = self.config.lm_path
        if not lm_path:
            raise RuntimeError(
                "LM rescoring is enabled but no LM path was provided. "
                "Set --lm-path (CLI) or StreamConfig.lm_path."
            )

        lm_file = Path(lm_path).expanduser()
        if not lm_file.is_file():
            raise RuntimeError(f"LM rescoring path does not exist or is not a file: {lm_file}")

        self._lm_scorer = KenLMScorer(str(lm_file))
        return self._lm_scorer

    def _prepare_rescoring(self) -> None:
        if not self._lm_rescore_active():
            return
        self._ensure_lm_scorer()

    def _rescore_decode_result(
        self,
        model: "whisper.Whisper",
        mel: Any,
        options: Any,
        original_decode: Callable[[Any, Any], Any],
        scorer: KenLMScorer,
    ) -> tuple[Any, RescoreSelectionMeta]:
        if getattr(options, "task", "transcribe") != "transcribe":
            result = original_decode(mel, options)
            return (
                result,
                RescoreSelectionMeta(
                    applied=False,
                    changed_from_baseline=False,
                    candidate_count=0,
                    baseline_index=0,
                    selected_index=0,
                ),
            )

        return rescore_decode_result(
            model=model,
            mel=mel,
            options=options,
            scorer=scorer,
            alpha=self.config.lm_alpha,
            beta=self.config.lm_beta,
            top_n=self._effective_lm_top_n(),
            normalize_text=self.config.lm_normalize_text,
        )

    @contextlib.contextmanager
    def _patched_decode_for_rescoring(self):
        if not self._lm_rescore_active():
            yield
            return

        scorer = self._ensure_lm_scorer()
        original_decode = self.model.decode

        def _patched_decode(model_self, mel, options):
            self._decode_stats["rescore_decode_calls"] += 1
            result, meta = self._rescore_decode_result(
                model=model_self,
                mel=mel,
                options=options,
                original_decode=original_decode,
                scorer=scorer,
            )
            if meta.applied:
                self._decode_stats["rescore_applied"] += 1
                self._decode_stats["rescore_candidate_total"] += int(meta.candidate_count)
                if meta.changed_from_baseline:
                    self._decode_stats["rescore_changed_hypothesis"] += 1
            return result

        self.model.decode = types.MethodType(_patched_decode, self.model)
        try:
            yield
        finally:
            self.model.decode = original_decode

    @staticmethod
    def _speech_end_seconds(end_value: Any, sample_rate: int, chunk_samples: int) -> float:
        end_f = max(0.0, float(end_value))
        if sample_rate <= 0:
            return 0.0
        chunk_sec = float(chunk_samples) / float(sample_rate)
        # Silero returns sample indices with return_seconds=False.
        # If a custom detector returns seconds, keep them as-is.
        if end_f <= chunk_sec + 1.0:
            return end_f
        return end_f / float(sample_rate)

    def _window_has_new_speech(
        self,
        speech_timestamps: List[Dict[str, Any]],
        window_start: float,
        sample_rate: int,
        chunk_samples: int,
        committed_time: float,
    ) -> bool:
        for span in speech_timestamps:
            end_sec = self._speech_end_seconds(span.get("end", 0.0), sample_rate, chunk_samples)
            abs_end = window_start + end_sec
            if abs_end > committed_time + _EPS:
                return True
        return False

    def _decode_window(
        self,
        audio: np.ndarray,
        sr: int,
        end_t: float,
        committed_time: float,
        prompt: str,
    ) -> tuple[float, List[Dict[str, Any]]]:
        cfg = self.config
        self._decode_stats["decode_ticks"] += 1

        window_start = max(0.0, end_t - cfg.window_sec)
        i0 = int(window_start * sr)
        i1 = int(end_t * sr)
        chunk = np.asarray(audio[i0:i1], dtype=np.float32).reshape(-1)

        min_chunk_samples = max(1, int(cfg.min_chunk_sec * sr))
        if len(chunk) < min_chunk_samples:
            self._decode_stats["decode_skipped_short_chunk"] += 1
            return window_start, []

        if cfg.silence_rms_threshold > 0.0:
            rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
            if rms < cfg.silence_rms_threshold:
                self._decode_stats["decode_skipped_silence"] += 1
                return window_start, []

        vad_detector = self._ensure_vad_detector()
        if vad_detector is not None:
            if self._vad_bypass_ticks_left > 0:
                self._decode_stats["decode_vad_bypassed"] += 1
                self._vad_bypass_ticks_left -= 1
            else:
                speech_timestamps = vad_detector.speech_timestamps(chunk, sr)
                has_new_speech = self._window_has_new_speech(
                    speech_timestamps=speech_timestamps,
                    window_start=window_start,
                    sample_rate=sr,
                    chunk_samples=int(chunk.size),
                    committed_time=committed_time,
                )
                if not has_new_speech:
                    self._decode_stats["decode_skipped_vad"] += 1
                    self._vad_no_skip_streak = 0
                    return window_start, []
                self._vad_no_skip_streak += 1
                if (
                    cfg.vad_adaptive_disable_ticks > 0
                    and self._vad_no_skip_streak >= cfg.vad_adaptive_disable_ticks
                ):
                    self._vad_no_skip_streak = 0
                    self._vad_bypass_ticks_left = max(1, int(cfg.vad_recheck_interval_ticks))

        decode_kwargs: Dict[str, Any] = {}
        beam_size = self._effective_beam_size()
        if beam_size is not None:
            decode_kwargs["beam_size"] = beam_size
        if cfg.beam_patience is not None:
            decode_kwargs["patience"] = cfg.beam_patience
        if cfg.beam_length_penalty is not None:
            decode_kwargs["length_penalty"] = cfg.beam_length_penalty

        with self._patched_decode_for_rescoring():
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
                    **decode_kwargs,
                )

        self._decode_stats["decode_performed"] += 1
        return window_start, result.get("segments", [])

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
        self._reset_decode_stats()
        self._prepare_rescoring()

        audio = whisper.load_audio(audio_path)
        sr = whisper.audio.SAMPLE_RATE
        total_sec = len(audio) / sr

        committed_time = 0.0
        committed_parts: List[str] = []

        last_display = ""
        t = 0.0

        def _decode_at(end_t: float, prompt: str) -> tuple[float, List[Dict[str, Any]]]:
            return self._decode_window(
                audio=audio,
                sr=sr,
                end_t=end_t,
                committed_time=committed_time,
                prompt=prompt,
            )

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
            **self._final_decode_stats(),
        }

    def stream_source(self, source: AudioSource) -> Iterator[Dict[str, Any]]:
        """
        Stream transcription from a live/incremental source (mic, websocket, RTSP, etc.).
        """
        cfg = self.config
        self._stop_flag = False
        self._reset_decode_stats()
        self._prepare_rescoring()

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

        def _decode_at(end_t: float, prompt: str) -> tuple[float, List[Dict[str, Any]]]:
            return self._decode_window(
                audio=audio,
                sr=sr,
                end_t=end_t,
                committed_time=committed_time,
                prompt=prompt,
            )

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
            **self._final_decode_stats(),
        }
