# Testing Guide

## Layout

The current repository contains 55 `tests/test_*.py` modules. The
parameterized test-case count can change as tests are added; use
`--collect-only` for the exact count on the checkout being tested.

```
tests/
├── fixtures/
│   ├── commands/             # command WAVs + manifest.json
│   ├── live/                 # live-capture samples
│   ├── physical/intel/       # Intel-array microphone recordings
│   ├── command_corpus.json   # intent corpus
│   ├── eval_manifest.json    # STT evaluation set
│   ├── eval_physical_*.json  # physical microphone evaluation sets
│   └── kws_benchmark_manifest*.json
└── test_*.py                 # 55 modules on the published baseline
```

## Running locally

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest tests/test_voice.py -q
.\.venv\Scripts\python.exe -m pytest --collect-only -q
```

`pytest.ini`: `pythonpath = .`, `testpaths = tests`, marker `slow`.

## GitHub Actions

`.github/workflows/windows-tests.yml` runs on Windows with Python 3.13.
It splits the modules into four independent jobs:

- Core, command routing and Windows controllers.
- Audio capture, microphone and voice pipeline.
- Session lifecycle and diagnostics.
- HUD, Qt and tray tests.

The shards run in separate processes. This is intentional: the release
readiness investigation recorded native Qt/offscreen-GL access
violations after long single-process test runs. A passing shard is not
physical hardware validation; CI does not prove microphone, speaker,
GPU, monitor or real Windows power behavior.

The workflow's first run is the authority for current CI status. Do not
describe CI as passing until GitHub reports all four jobs green.

## What is covered

| Area | Modules (examples) |
|---|---|
| Core/configuration | `test_application`, `test_configuration`, `test_database`, `test_single_instance`, `test_startup_manager` |
| Voice pipeline | `test_voice`, `test_mic_pipeline`, `test_dynamic_mic`, `test_detector_ownership`, `test_audio_recovery`, `test_post_standby_recognition` |
| Recognition | `test_live_reliability`, `test_intent_accuracy`, `test_speech_accuracy_v105`, `test_voice_command_accuracy`, `test_command_corpus` |
| Windows controllers | `test_monitor_manager`, `test_monitor_commands`, `test_app_identity`, `test_dynamic_discovery`, `test_windows_system_control` |
| HUD and tray | `test_hud_v107`, `test_hud_v108_overlay`, `test_hud_v108_appearance`, `test_hud_v108_topmost`, `test_tray` |
| Diagnostics | `test_v108_async_rescore`, `test_wake_diagnostics_v106`, `test_v108_task1_diagnostics`, `test_v108_task2_kws_benchmark`, `test_v108_task4_stt_diagnostics` |

## Fixture policy

- **Never modify reference WAVs** to make a test pass.
- Physical recordings under `tests/fixtures/physical/` are hardware-
  specific evaluation data. Preserve them when they are needed for
  repeatable model measurements.
- The repository contains recorded speech WAV fixtures. Confirm
  permission and remove identifying background content before adding
  new recordings or distributing additional samples; see
  [DATA_PRIVACY.md](DATA_PRIVACY.md).
- Negative samples (`silence.wav`, `background_speech.wav`,
  `open_nonexistent.wav`) guard against false activations.
- New command coverage should add corpus entries and fixture audio
  where practical and appropriately consented.

## Known issues and historical test evidence

- `test_live_reliability.py::TestRecordedCommandPipeline::test_fixture[hey_jarvis.wav]`
  has a known failure on the development model: Moonshine Base INT8
  transcribes the fixture as `Hage-arvis`. Do not skip or weaken this
  test to conceal the recognition limitation.
- The release-readiness report recorded native Qt/offscreen-GL access
  violations during repeated full-suite, single-process runs. It also
  recorded that the modules passed when executed in three sequential
  batches. The current Windows CI workflow isolates the test areas into
  four processes; the first CI run must verify whether that is sufficient.
- `test_v108_task1_diagnostics` had a timing-sensitive failure that
  was not reproduced in 45+ follow-up attempts in the recorded
  investigation. Keep its assertions intact.

The cleanup validation report's 459 passed / 0 failed result covered
16 selected modules. It is focused regression evidence, not a new
full-suite result.

## Benchmarks (manual)

Offline model benchmarks live in `scripts/` and write JSON/Markdown
reports under the runtime `reports/` directory or an explicit
`--output` path:

- `benchmark_kws.py --manifest tests/fixtures/kws_benchmark_manifest.json`
- `benchmark_stt.py --model-dir <dir> --eval <manifest>`
- `benchmark_v105_models.py` — sweeps STT model directories and eval sets.
- `compare_live_kws_replay.py` — replays captured audio through KWS.
- `record_eval_samples.py` — explicitly records microphone evaluation
  samples; it writes WAV files under `tests/fixtures/physical/`.
