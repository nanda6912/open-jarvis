# Offline Model Setup

VoiceOS uses three offline model families, all via
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Apache-2.0
code; each model keeps its own upstream license — see each model's
`LICENSE`/`README` inside its directory). **Model binaries are not
committed to this repository.**

Machine-readable requirements: `config/model_manifest.json`.
Verification: `scripts/verify_installation.py`.

## Target layout

```
%LOCALAPPDATA%\VoiceOS\models\
├── kws\sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01\
├── stt\sherpa-onnx-moonshine-base-en-int8\
└── tts\vits-piper-en_US-lessac-medium-int8\
    └── (plus lexicon-en-us.txt at models\tts\ or inside the model dir)
```

Directories can be relocated with `VOICEOS_KWS_MODEL_DIR`,
`VOICEOS_STT_MODEL_DIR`, `VOICEOS_TTS_MODEL_DIR`, or per-engine
`voice.stt_model_dir` / `voice.tts_model_dir` settings.

## 1. Wake-word (KWS)

- **Model:** `sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01`
- **Source:** sherpa-onnx release tag `kws-models`,
  `https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2`
- **Extract to:** `models\kws\`
- **Required files** (searched recursively under `models\kws\`):
  `encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`, `tokens.txt`,
  and `keywords_jarvis.txt` (preferred) or `keywords.txt`.
- **Notes:** the shipped keyword file contains the BPE-tokenized
  "hello jarvis" pronunciation variants tuned on physical microphones.
  Custom wake phrases need their own tokenized keyword file.

## 2. Speech-to-text (STT)

- **Model:** `sherpa-onnx-moonshine-base-en-int8` (production
  default; `moonshine-tiny-en-int8` also works via `voice.stt_engine`)
- **Source:** sherpa-onnx release tag `asr-models`,
  `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-en-int8.tar.bz2`
  (Hugging Face mirror: `csukuangfj/sherpa-onnx-moonshine-base-en-int8`)
- **Upstream:** [usefulsensors/moonshine](https://github.com/usefulsensors/moonshine)
- **Extract to:** `models\stt\`
- **Required files:** `preprocess.onnx`, `encode*.onnx`,
  `uncached_decode*.onnx`, `cached_decode*.onnx`, `tokens.txt`.
- **Notes:** model type `moonshine` is auto-detected from
  `preprocess.onnx` + `encode*.onnx`; int8 variants are preferred
  automatically.

## 3. Text-to-speech (TTS)

- **Model:** `vits-piper-en_US-lessac-medium-int8` (Piper voice,
  single male speaker, 22050 Hz)
- **Source:** sherpa-onnx release tag `tts-models` — pick the
  `-int8` asset for `vits-piper-en_US-lessac-medium`
  (`https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models`);
  upstream voice: `rhasspy/piper-voices` `en/en_US/lessac/medium`.
- **Extract to:** `models\tts\`
- **Required files:** `*.onnx` (model), `tokens.txt`, and a lexicon —
  `lexicon-en-us.txt`
  (`https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt`)
  placed in the model dir or `models\tts\`. `espeak-ng-data/` must
  remain inside the model directory.
- **Notes:** `voice.tts_speed` adjusts speech rate; `ack_mode`
  controls acknowledgement verbosity.

## Verification

```powershell
.\.venv\Scripts\python.exe scripts\verify_installation.py
```

Reports PASS/WARNING/FAIL per model with the exact missing glob
patterns and the env var to override. When models are absent the
engines degrade to null implementations — the app runs but wake-word,
recognition and speech are inactive (warnings in `voiceos.log`).

## License note

sherpa-onnx is Apache-2.0. Moonshine models derive from
usefulsensors/moonshine (see its LICENSE, shipped in the model dir).
Piper voices are distributed under their own terms — keep the `LICENSE`
files that ship inside each model archive.
