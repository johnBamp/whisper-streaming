# whisper-streaming

Rolling-window, streaming-style transcription utilities built on OpenAI Whisper.

This package gives you:
- A library API for stable partial/final transcript events.
- An `AudioSource` abstraction (file, microphone, websocket, RTSP).
- A CLI (`whisper-stream`) for quick local use.

## Install

```bash
python3 -m pip install whisper-streaming
```

For microphone backends:

```bash
python3 -m pip install 'whisper-streaming[audio]'
```

For Silero VAD decode gating:

```bash
python3 -m pip install 'whisper-streaming[vad]'
```

For KenLM beam rescoring:

```bash
python3 -m pip install 'whisper-streaming[lm]'
```

## Quickstart (Library)

```python
import whisper
from whisper_streaming import StreamConfig, StreamTranscriber, FileAudioSource

model = whisper.load_model("tiny.en")
cfg = StreamConfig(step_sec=0.5, window_sec=8.0, commit_lag_sec=1.0)
st = StreamTranscriber(model, cfg)

source = FileAudioSource("sample.wav")
for evt in st.stream(source):
    if evt["type"] == "final":
        print(evt["text"])
    elif evt["type"] == "done":
        print("FINAL:", evt["text"])
```

## CLI Examples

File transcription:

```bash
whisper-stream sample.wav --model tiny.en
```

File transcription with VAD + silence gating (skip no-speech decode windows):

```bash
whisper-stream sample.wav --model tiny.en --vad --silence-rms-threshold 0.002
```

Beam-only decoding:

```bash
whisper-stream sample.wav --model tiny.en --beam-size 8
```

Beam + KenLM rescoring:

```bash
whisper-stream sample.wav --model tiny.en --beam-size 8 --lm-rescore --lm-path ./model.arpa --lm-alpha 0.25 --lm-beta 0.0
```

Max speed profile (lower update cadence, keeps quality high on continuous speech clips):

```bash
whisper-stream sample.wav --model tiny.en --step 3.0 --window 9.0 --lag 1.0 --no-vad
```

Live microphone transcription:

```bash
whisper-stream --source mic --mic-backend sounddevice --show-partials
```

List input devices and pick one explicitly:

```bash
whisper-stream --list-mic-devices
whisper-stream --source mic --mic-backend sounddevice --mic-device 2 --show-partials
```

Mic debug capture (save + playback after Ctrl+C):

```bash
whisper-stream --source mic --mic-backend sounddevice --show-partials --debug-mic-capture --debug-mic-normalize --debug-mic-wav ./mic-check.wav
```

WebSocket / RTSP input (requires `ffmpeg`):

```bash
whisper-stream --source websocket --url ws://localhost:9000/audio
whisper-stream --source rtsp --url rtsp://camera.local/stream --rtsp-transport tcp
```

Tune Silero VAD behavior:

```bash
whisper-stream sample.wav --vad --vad-backend onnx --vad-threshold 0.45 --vad-min-speech-ms 180 --vad-min-silence-ms 120 --vad-speech-pad-ms 40
```

For long streams, adaptive VAD bypass avoids VAD overhead when every window already has speech:

```bash
whisper-stream sample.wav --vad --vad-adaptive-disable-ticks 8 --vad-recheck-interval-ticks 12
```

LM rescoring notes:
- Rescoring is opt-in (`--lm-rescore`) and only applied for `--task transcribe`.
- If `--lm-rescore` is enabled in transcribe mode and LM setup is invalid, the CLI fails fast with an actionable error.

## API Notes

Main classes:
- `StreamTranscriber`
- `StreamConfig`
- `AudioSource`
- `FileAudioSource`
- `MicrophoneAudioSource`
- `WebSocketAudioSource`
- `RTSPAudioSource`
- `RecordingAudioSource`

Event types yielded by streaming methods:
- `partial`: volatile text (committed + draft)
- `final`: committed text advanced
- `done`: final committed transcript

## Development

Create a virtual environment and install dev tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev,audio]'
```

Run lint/tests:

```bash
python3 -m ruff check src tests
python3 -m pytest -q -m 'not integration'
```

Run integration tests (optional):

```bash
WHISPER_INTEGRATION=1 python3 -m pytest -q -m integration
```

## Build and Publish

Build artifacts:

```bash
python3 -m build
python3 -m twine check dist/*
```

Upload to TestPyPI first:

```bash
python3 -m twine upload --repository testpypi dist/*
```

Then upload to PyPI:

```bash
python3 -m twine upload dist/*
```

## Push to GitHub

From the repository root (if not already initialized, run `git init` first):

```bash
git add .
git commit -m "Initial whisper-streaming library"
git branch -M main
git remote add origin git@github.com:johnreidbampfield/whisper-streaming.git
git push -u origin main
```

## License

MIT
