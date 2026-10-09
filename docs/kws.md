# Streaming keyword spotting

`OnnxZipformerKws` loads a local streaming encoder, stateless decoder, joiner
and `config.json`. It returns acoustic phrase detections only. Capture,
downloads, tokenization, model integrity, command authorization and actions
belong to the host.

Feed normalized mono 16 kHz Float32 PCM in arrival order on one serial worker.
Each block is limited to two seconds. `push` returns zero or more hits;
`finish` pads the tail and closes the stream. After finish or a backend failure,
reset explicitly or construct a fresh instance. Backend errors propagate;
there is no alternate model or transcription fallback.

Register `KwsKeyword` entries with a phrase, tokenizer IDs, acoustic threshold
and context boost. Blank tokens and duplicate token sequences are rejected.
Shared trie prefixes use one boost. Modified beam expansion selects acoustic
candidates before applying context scores; context boost cannot bypass the
independent average acoustic threshold or trailing-blank requirement.

Idle reset counts frames since the last nonblank token rather than session
age. Keyword hits reset search state without rewinding `stream_frame`.
Explicit reset restarts the recording clock. Frame positions advance by
40 ms; `audio_end_seconds` names the accepted PCM available to the decoder
for that call, which lets the host reject stale asynchronous delivery.

The frontend is separate from speaker embedding's fbank/CMVN contract:
80 Kaldi log-mel bins, normalized samples, 400/160 frame/shift, Povey window,
DC removal, 0.97 preemphasis, no dither, mirrored nonsnipping boundaries and
high frequency -400 Hz. Its streaming behavior and numeric values are checked
against the existing frozen Kaldi fixture.

Model tensor names, ranks, types and configured/actual dimensions must match.
Float outputs, PCM and logits must be finite. Audio, state and decoder-history
budgets are bounded. Complete token history affects normalized beam ranking,
so exhausting its budget throws instead of silently truncating it.

Build with `SPEECH_CORE_WITH_ONNX=ON`. Core decoder/frontend unit tests need
no model. `test_onnx_kws_model` runs when `SPEECH_KWS_MODEL_DIR` names the
local bundle; without it the model test reports a skip. It can also replay a
caller-owned vocabulary and normalized Float32 audio:

```sh
test_onnx_kws_model /models/kws phrases.json clip.f32
```

`phrases.json` is an array of `{"phrase":"alpha beta","tokens":[1,2]}`
entries; the token IDs are examples and must match the actual tokenizer.
Replay prints phrase names and stream times for offline comparison. This does
not authorize commands or establish acoustic accuracy.
