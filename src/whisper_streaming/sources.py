from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional
import shutil
import subprocess
import wave

import numpy as np
import whisper


class AudioSource(ABC):
    """Source of mono float32 audio chunks at Whisper's sample rate."""

    def __init__(self, sample_rate: int = whisper.audio.SAMPLE_RATE, chunk_sec: float = 0.5) -> None:
        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec

    def __enter__(self) -> "AudioSource":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    @abstractmethod
    def chunks(self) -> Iterator[np.ndarray]:
        """Yield 1D float32 PCM chunks in range [-1.0, 1.0]."""

    def close(self) -> None:
        """Release resources held by the source."""
        return


class FileAudioSource(AudioSource):
    def __init__(
        self,
        audio_path: str,
        sample_rate: int = whisper.audio.SAMPLE_RATE,
        chunk_sec: float = 0.5,
    ) -> None:
        super().__init__(sample_rate=sample_rate, chunk_sec=chunk_sec)
        self.audio_path = audio_path

    def chunks(self) -> Iterator[np.ndarray]:
        if self.sample_rate != whisper.audio.SAMPLE_RATE:
            raise ValueError(
                f"FileAudioSource sample_rate={self.sample_rate} is unsupported; "
                f"use {whisper.audio.SAMPLE_RATE}Hz for Whisper compatibility."
            )

        audio = whisper.load_audio(self.audio_path)
        frames = max(1, int(self.chunk_sec * self.sample_rate))
        for i in range(0, len(audio), frames):
            yield np.asarray(audio[i : i + frames], dtype=np.float32)


class RecordingAudioSource(AudioSource):
    """
    Wrap another AudioSource and keep a copy of streamed audio for debugging.
    """

    def __init__(self, source: AudioSource) -> None:
        super().__init__(sample_rate=source.sample_rate, chunk_sec=source.chunk_sec)
        self.source = source
        self._captured: list[np.ndarray] = []
        self._captured_samples = 0

    def chunks(self) -> Iterator[np.ndarray]:
        for chunk in self.source.chunks():
            captured = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if captured.size:
                self._captured.append(captured.copy())
                self._captured_samples += int(captured.size)
            yield captured

    def close(self) -> None:
        self.source.close()

    @property
    def has_audio(self) -> bool:
        return self._captured_samples > 0

    @property
    def recorded_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(self._captured_samples) / float(self.sample_rate)

    def recorded_audio(self) -> np.ndarray:
        if not self._captured:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._captured)

    def recorded_peak(self) -> float:
        audio = self.recorded_audio()
        if audio.size == 0:
            return 0.0
        return float(np.max(np.abs(audio)))

    def recorded_rms(self) -> float:
        audio = self.recorded_audio()
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio * audio)))

    def save_wav(self, path: str, normalize: bool = False, target_peak: float = 0.95) -> str:
        audio = self.recorded_audio()
        if normalize and audio.size:
            peak = float(np.max(np.abs(audio)))
            if peak > 0.0:
                gain = min(float(target_peak) / peak, 20.0)
                audio = audio * gain
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm16.tobytes())
        return path


class MicrophoneAudioSource(AudioSource):
    def __init__(
        self,
        sample_rate: int = whisper.audio.SAMPLE_RATE,
        chunk_sec: float = 0.5,
        backend: str = "auto",
        device: Optional[int] = None,
    ) -> None:
        super().__init__(sample_rate=sample_rate, chunk_sec=chunk_sec)
        if backend not in {"auto", "sounddevice", "pyaudio"}:
            raise ValueError("backend must be one of: auto, sounddevice, pyaudio")
        self.backend = backend
        self.device = device

        self._active_backend: Optional[str] = None
        self._pa = None
        self._stream = None

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend

        try:
            import sounddevice  # noqa: F401

            return "sounddevice"
        except ImportError:
            pass

        try:
            import pyaudio  # noqa: F401

            return "pyaudio"
        except ImportError as exc:
            raise RuntimeError(
                "No microphone backend available. Install sounddevice or pyaudio."
            ) from exc

    def chunks(self) -> Iterator[np.ndarray]:
        self._active_backend = self._resolve_backend()
        if self._active_backend == "sounddevice":
            yield from self._chunks_sounddevice()
            return
        yield from self._chunks_pyaudio()

    def _chunks_sounddevice(self) -> Iterator[np.ndarray]:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice backend requested but not installed.") from exc

        frames = max(1, int(self.chunk_sec * self.sample_rate))
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            # Keep capture format deterministic; convert to float ourselves.
            dtype="int16",
            blocksize=frames,
            device=self.device,
        )
        self._stream.start()
        try:
            while True:
                data, _overflowed = self._stream.read(frames)
                chunk = np.asarray(data, dtype=np.int16).reshape(-1).astype(np.float32) / 32768.0
                yield chunk
        finally:
            self.close()

    def _chunks_pyaudio(self) -> Iterator[np.ndarray]:
        try:
            import pyaudio
        except ImportError as exc:
            raise RuntimeError("pyaudio backend requested but not installed.") from exc

        frames = max(1, int(self.chunk_sec * self.sample_rate))
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device,
            frames_per_buffer=frames,
        )
        try:
            while True:
                raw = self._stream.read(frames, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                yield chunk
        finally:
            self.close()

    def close(self) -> None:
        stream = self._stream
        self._stream = None

        if stream is not None:
            if self._active_backend == "sounddevice":
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            else:
                try:
                    stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None


class FFmpegAudioSource(AudioSource):
    def __init__(
        self,
        input_url: str,
        sample_rate: int = whisper.audio.SAMPLE_RATE,
        chunk_sec: float = 0.5,
        ffmpeg_bin: str = "ffmpeg",
        input_args: Optional[list[str]] = None,
    ) -> None:
        super().__init__(sample_rate=sample_rate, chunk_sec=chunk_sec)
        self.input_url = input_url
        self.ffmpeg_bin = ffmpeg_bin
        self.input_args = input_args or []
        self._proc: Optional[subprocess.Popen] = None

    def _build_cmd(self) -> list[str]:
        return [
            self.ffmpeg_bin,
            "-nostdin",
            "-loglevel",
            "error",
            *self.input_args,
            "-i",
            self.input_url,
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ]

    def chunks(self) -> Iterator[np.ndarray]:
        if shutil.which(self.ffmpeg_bin) is None:
            raise RuntimeError(
                f"ffmpeg executable '{self.ffmpeg_bin}' not found in PATH. "
                "Install ffmpeg to use websocket/RTSP sources."
            )

        frames = max(1, int(self.chunk_sec * self.sample_rate))
        bytes_per_chunk = frames * 2  # int16 mono
        self._proc = subprocess.Popen(
            self._build_cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        if self._proc.stdout is None:
            raise RuntimeError("Failed to start ffmpeg audio pipeline.")

        try:
            while True:
                raw = self._proc.stdout.read(bytes_per_chunk)
                if not raw:
                    break

                # Keep int16 framing aligned if a short odd-byte read occurs.
                if len(raw) % 2:
                    raw = raw[:-1]
                    if not raw:
                        continue

                chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                yield chunk
        finally:
            self.close()

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return

        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()


class WebSocketAudioSource(FFmpegAudioSource):
    def __init__(
        self,
        input_url: str,
        sample_rate: int = whisper.audio.SAMPLE_RATE,
        chunk_sec: float = 0.5,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        if not input_url.startswith(("ws://", "wss://")):
            raise ValueError("WebSocketAudioSource expects a ws:// or wss:// URL.")
        super().__init__(
            input_url=input_url,
            sample_rate=sample_rate,
            chunk_sec=chunk_sec,
            ffmpeg_bin=ffmpeg_bin,
            input_args=["-fflags", "+nobuffer"],
        )


class RTSPAudioSource(FFmpegAudioSource):
    def __init__(
        self,
        input_url: str,
        sample_rate: int = whisper.audio.SAMPLE_RATE,
        chunk_sec: float = 0.5,
        ffmpeg_bin: str = "ffmpeg",
        rtsp_transport: str = "tcp",
    ) -> None:
        if not input_url.startswith("rtsp://"):
            raise ValueError("RTSPAudioSource expects an rtsp:// URL.")
        if rtsp_transport not in {"tcp", "udp"}:
            raise ValueError("rtsp_transport must be 'tcp' or 'udp'.")
        super().__init__(
            input_url=input_url,
            sample_rate=sample_rate,
            chunk_sec=chunk_sec,
            ffmpeg_bin=ffmpeg_bin,
            input_args=["-rtsp_transport", rtsp_transport, "-fflags", "+nobuffer"],
        )
