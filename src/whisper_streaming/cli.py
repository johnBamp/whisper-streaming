import argparse
import sys
import time
import whisper

from .core import StreamConfig, StreamTranscriber


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="whisper-stream", description="Streaming-style Whisper transcription for a file.")
    p.add_argument("audio_path", help="Path to audio file (wav/mp3/m4a/...)")
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
    started_at = time.perf_counter()

    last_partial = ""
    last_final = ""
    last_partial_len = 0
    for evt in st.stream_file(args.audio_path):
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
