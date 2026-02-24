import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import time
import wave

import numpy as np
import whisper

from .core import StreamConfig, StreamTranscriber
from .sources import (
    FileAudioSource,
    MicrophoneAudioSource,
    RTSPAudioSource,
    RecordingAudioSource,
    WebSocketAudioSource,
)


def _default_debug_wav_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"mic_debug_capture_{stamp}.wav"


def _play_wav(path: Path) -> bool:
    try:
        import sounddevice as sd

        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels)
        sd.play(audio, sample_rate)
        sd.wait()
        return True
    except Exception:
        pass

    ffplay = shutil.which("ffplay")
    if ffplay:
        try:
            subprocess.run(
                [ffplay, "-nodisp", "-autoexit", "-loglevel", "error", str(path)],
                check=True,
            )
            return True
        except Exception:
            return False

    return False


def _list_mic_devices() -> int:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        default_input = None
        try:
            default_input = int(sd.default.device[0])
        except Exception:
            default_input = None

        print("MIC INPUT DEVICES (sounddevice):")
        found = False
        for idx, dev in enumerate(devices):
            max_inputs = int(dev.get("max_input_channels", 0))
            if max_inputs <= 0:
                continue
            found = True
            mark = " [default]" if default_input is not None and idx == default_input else ""
            print(
                f"- {idx}: {dev.get('name', 'unknown')} "
                f"(inputs={max_inputs}, default_sr={dev.get('default_samplerate', 'unknown')}){mark}"
            )
        if not found:
            print("- no input devices found")
        return 0
    except Exception:
        pass

    try:
        import pyaudio

        pa = pyaudio.PyAudio()
        print("MIC INPUT DEVICES (pyaudio):")
        found = False
        for idx in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(idx)
            max_inputs = int(dev.get("maxInputChannels", 0))
            if max_inputs <= 0:
                continue
            found = True
            print(
                f"- {idx}: {dev.get('name', 'unknown')} "
                f"(inputs={max_inputs}, default_sr={dev.get('defaultSampleRate', 'unknown')})"
            )
        if not found:
            print("- no input devices found")
        pa.terminate()
        return 0
    except Exception:
        print("Unable to list mic devices (install sounddevice or pyaudio).")
        return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="whisper-stream",
        description="Streaming-style Whisper transcription from file, mic, websocket, or RTSP.",
    )
    p.add_argument("audio_path", nargs="?", help="Path to audio file (required for --source=file)")
    p.add_argument(
        "--source",
        default="file",
        choices=["file", "mic", "websocket", "rtsp"],
        help="Audio input source type.",
    )
    p.add_argument("--url", default=None, help="Input URL for websocket/RTSP sources")
    p.add_argument("--chunk", type=float, default=0.5, help="Input chunk size in seconds for live sources")
    p.add_argument("--mic-backend", default="auto", choices=["auto", "sounddevice", "pyaudio"])
    p.add_argument("--mic-device", type=int, default=None, help="Optional microphone device index")
    p.add_argument("--list-mic-devices", action="store_true", help="List available microphone input devices and exit.")
    p.add_argument(
        "--debug-mic-capture",
        action="store_true",
        help="Record raw mic input and play it back after transcription exits.",
    )
    p.add_argument(
        "--debug-mic-wav",
        default=None,
        help="Path to save mic debug capture WAV (default: ./mic_debug_capture_<timestamp>.wav)",
    )
    p.add_argument(
        "--debug-mic-normalize",
        action="store_true",
        help="Normalize saved mic debug WAV for easier listening.",
    )
    p.add_argument("--rtsp-transport", default="tcp", choices=["tcp", "udp"])
    p.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name/path for websocket/RTSP")
    p.add_argument("--model", default="tiny.en", help="Whisper model name (e.g., tiny.en, base, small)")
    p.add_argument("--step", type=float, default=0.5, help="Tick step seconds")
    p.add_argument("--window", type=float, default=8.0, help="Rolling window seconds")
    p.add_argument("--lag", type=float, default=1.0, help="Commit lag seconds (stability delay)")
    p.add_argument(
        "--min-commit-advance",
        type=float,
        default=0.35,
        help="Minimum segment end-time advance (seconds) before accepting a new commit.",
    )
    p.add_argument("--realtime", action="store_true", help="Sleep to simulate realtime playback")
    p.add_argument("--language", default=None, help="Force language code (e.g., en), otherwise auto")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument(
        "--use-initial-prompt",
        action="store_true",
        help="Pass committed text back as Whisper initial_prompt (can improve continuity, can also bias output).",
    )
    p.add_argument(
        "--show-partials",
        action="store_true",
        help="Show volatile partial hypotheses inline (may be revised frequently).",
    )
    args = p.parse_args(argv)

    if args.list_mic_devices:
        return _list_mic_devices()

    if args.source == "file" and not args.audio_path:
        p.error("audio_path is required when --source=file")
    if args.source in {"websocket", "rtsp"} and not args.url:
        p.error("--url is required when --source=websocket or --source=rtsp")
    if args.debug_mic_capture and args.source != "mic":
        p.error("--debug-mic-capture is only supported with --source=mic")

    model = whisper.load_model(args.model)

    cfg = StreamConfig(
        step_sec=args.step,
        window_sec=args.window,
        commit_lag_sec=args.lag,
        min_commit_advance_sec=args.min_commit_advance,
        sleep_to_simulate_realtime=args.realtime,
        language=args.language,
        task=args.task,
        use_initial_prompt=args.use_initial_prompt,
    )
    st = StreamTranscriber(model, cfg)

    recorder = None
    if args.source == "file":
        source = FileAudioSource(args.audio_path, chunk_sec=args.chunk)
    elif args.source == "mic":
        mic_source = MicrophoneAudioSource(
            chunk_sec=args.chunk,
            backend=args.mic_backend,
            device=args.mic_device,
        )
        if args.debug_mic_capture:
            recorder = RecordingAudioSource(mic_source)
            source = recorder
        else:
            source = mic_source
    elif args.source == "websocket":
        source = WebSocketAudioSource(args.url, chunk_sec=args.chunk, ffmpeg_bin=args.ffmpeg_bin)
    else:
        source = RTSPAudioSource(
            args.url,
            chunk_sec=args.chunk,
            ffmpeg_bin=args.ffmpeg_bin,
            rtsp_transport=args.rtsp_transport,
        )

    started_at = time.perf_counter()

    last_partial = ""
    last_final = ""
    last_partial_len = 0
    for evt in st.stream(source):
        if evt["type"] == "partial" and args.show_partials:
            text = evt["text"]
            if text != last_partial:
                # Clear stale tail when the new inline text is shorter than the previous one.
                pad = " " * max(0, last_partial_len - len(text))
                print("\r" + text + pad, end="", flush=True)
                last_partial = text
                last_partial_len = len(text)
        elif evt["type"] == "final":
            text = evt["text"]
            if text != last_final:
                if args.show_partials:
                    print("\r" + text + " " * max(0, last_partial_len - len(text)), end="", flush=True)
                    last_partial_len = len(text)
                else:
                    print(text, flush=True)
                last_final = text
        elif evt["type"] == "done":
            if args.show_partials:
                print()
            print("\n\nFINAL:\n" + evt["text"])
            wall_sec = time.perf_counter() - started_at
            clip_sec = float(evt.get("t", 0.0))
            committed_sec = min(float(evt.get("committed_until", 0.0)), clip_sec)
            speed = (clip_sec / wall_sec) if wall_sec > 0 else 0.0
            print(
                "\nDEBUG:\n"
                f"- clip_duration_sec: {clip_sec:.2f}\n"
                f"- committed_until_sec: {committed_sec:.2f}\n"
                f"- transcription_wall_sec: {wall_sec:.2f}\n"
                f"- speed_vs_realtime: {speed:.2f}x"
            )

    if recorder is not None:
        debug_path = Path(args.debug_mic_wav).expanduser() if args.debug_mic_wav else _default_debug_wav_path()
        debug_path = debug_path.resolve()
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        if recorder.has_audio:
            peak = recorder.recorded_peak()
            rms = recorder.recorded_rms()
            recorder.save_wav(str(debug_path), normalize=args.debug_mic_normalize)
            print(
                "\nMIC DEBUG:\n"
                f"- captured_audio_sec: {recorder.recorded_seconds:.2f}\n"
                f"- input_peak: {peak:.4f}\n"
                f"- input_rms: {rms:.4f}\n"
                f"- saved_wav: {debug_path}\n"
                f"- normalize_wav: {str(args.debug_mic_normalize).lower()}"
            )
            if peak < 0.02:
                print("- warning: mic level looks very low; try a different --mic-device or OS input device.")
            if _play_wav(debug_path):
                print("- playback: success")
            else:
                print("- playback: unavailable (install sounddevice or ffplay)")
        else:
            print("\nMIC DEBUG:\n- captured_audio_sec: 0.00\n- saved_wav: not written (no audio captured)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
