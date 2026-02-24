Most examples of Real Time Audio in Whisper work by dividing the audio into cuts, transcribing each cut one by one, and appending the text.

The main problem with this approach is if I were to say "Hello, my name is John B" it may divide it into something like [hello my na][me is john B] and because [...na] and [me..] are in two different cuts, the meaning is ambiguous and whisper cannot figure out those two sounds combine into a larger word.


The other approach I have seen is to record a cut of audio, [hello my na], then transcribe it, then concatenate more of the audio each time so it can rerun transcriptions on previously messed-up audio cuts. This is effective and better than having no real-time audio, but you end up transcribing over and over each time on a longer input.

As the audio grows like Δ, 2Δ, 3Δ, ..., kΔ, then total work is O(Δ) + O(2Δ) + O(3Δ) + ... + O(kΔ) that sum is O(Δ * (1 + 2 + 3 + ... + k)) = O(Δ * k²) and since N = kΔ the total complexity of the algorithm is quadratic, O(N²) but full real time transcription would be ideally O(1). True O(1) inference per tick would mean caching transformer key/value states and decoding incrementally, and I didn't feel like doing that, so I choose the second best way, which was to do a streaming style transcription with is to use a rolling window and then doing a stable commit. The idea is threefold:

-Every "STEP_SEC", take the last "WINDOW_SEC" seconds of audio (implicit overlap).
- Transcribe that window.
- Only commit segments that end at least "COMMIT_LAG_SEC" seconds before the current time. (Segments near the end change on the next tick.)
- keep a strict commit cursor and only advance it when new committed text is actually added
- flush the tail at the end so "done" includes the lagged region
- trim overlap using token-level matching (not just raw chars), so punctuation drift is handled better
- reject tiny end-time jitter commits with "min_commit_advance_sec" so near-duplicate paraphrase fragments are not committed
- default "use_initial_prompt" to "False" to reduce continuation bias in this streaming approach
- keep partial display optional in CLI ("--show-partials"), while default output shows stable commit

This way I was able to get much closer to amortized O(1) per tick and O(N) total over the entire audio stream, though it is not true streaming decoding. 

Specs I got using a mac M4 pro with 24gb ram:
-tiny.en model: 3.82x speed vs realtime / 1 second of audio processed in ~0.26 seconds / ~74% headroom
-base.en model: 1.88x speed vs realtime 
-small.en model: 0.94x speed vs realtime