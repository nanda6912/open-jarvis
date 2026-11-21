# Testing Guide

## Layout

```
tests/
├── fixtures/
│   ├── commands/            # command WAVs + manifest.json
│   ├── live/                # live-capture samples
│   ├── physical/intel/      # Intel-array mic recordings (r1-r3)
│   ├── command_corpus.json  # intent corpus
│   ├── eval_manifest.json   # STT eval set
│   ├── eval_physical_intel.json / eval_physical_realtek.json
│   └── kws_benchmark_manifest.example.json
└── test_*.py                # ~50 modules, ~1200 tests
```

## Running

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest tests/test_voice.py -q
.\.venv\Scripts\python.exe -m pytest --collect-only -q   # count only
```

`pytest.ini`: `pythonpath = .`, `testpaths = tests`, marker `slow`.

## What is covered

| Area | Modules (examples) |
|---|---|
| Core/config | `test_application`, `test_configuration`, `test_database`, `test_single_instance`, `test_startup_manager`, `test_state` via others |
| Voice pipeline | `test_voice`, `test_mic_pipeline`, `test_dynamic_mic`, `test_intel_array_wake_detection`, `test_post_standby_recognition`, `test_standby_watchdog`, `test_vad_recovery`, `test_wake_cycles`, `test_wake_word_after_session`, `test_wake_word_latency` |
| Recognition | `test_live_reliability`, `test_intent_accuracy`, `test_speech_accuracy_v105`, `test_voice_command_accuracy`, `test_voice_reliability`, `test_command_corpus`, `test_vad_segmentation_v105` |
| Controllers | `test_monitor_manager`, `test_monitor_commands`, `test_monitor_command_recognition`, `test_monitor_command_accuracy_v105`, `test_app_identity`, `test_app_catalog`, `test_dynamic_discovery` |
| HUD | `test_hud_v107`, `test_hud_v108_overlay`, `test_hud_v108_appearance`, `test_hud_v108_topmost`, `test_v1071_hotfix`, `test_v1075_reliability`, `test_v10751_production_trace` |
| Diagnostics/bench | `test_v108_task1_diagnostics`, `test_v108_task2_kws_benchmark`, `test_v108_task4_stt_diagnostics`, `test_kws_parity_diagnostic_v10741`, `test_wake_diagnostics_v106`, `test_v92`, `test_task4/5/6`, `test_tray`, `test_quick_acknowledgement`, `test_conversation_response_latency`, `test_p0_p1_system_ops`, `test_p2_awareness`, `test_live_fixes` |

## Fixture policy

- **Never modify reference WAVs** to make a test pass.
- Physical recordings (`tests/fixtures/physical/intel/`,
  `eval_physical_*.json`) are hardware-specific regression data —
  keep them even though they came from one laptop; they measure real
  array-microphone performance.
- Negative samples (`silence.wav`, `background_speech.wav`,
  `open_nonexistent.wav`) guard against false activations.
- New commands should add corpus entries + fixture audio where
  practical.

## Known issues

- `test_live_reliability.py::TestRecordedCommandPipeline::test_fixture[hey_jarvis.wav]`
  fails: Moonshine Base INT8 transcribes the fixture as "Hage-arvis".
  Documented model limitation — do not skip or weaken the test.
- Qt HUD tests create real (offscreen) windows; a rare native access
  violation has been observed once in a full-suite run
  (`test_v1071_hotfix.py`) that does not reproduce in isolation —
  treat as flaky infrastructure, re-run to confirm.

## Benchmarks (manual)

Offline model benchmarks live in `scripts/` and write JSON/Markdown
reports under the runtime `reports/` dir or an explicit `--output`:

- `benchmark_kws.py --manifest tests/fixtures/kws_benchmark_manifest.json`
- `benchmark_stt.py --model-dir <dir> --eval <manifest>`
- `benchmark_v105_models.py` — sweeps all STT model dirs × eval sets
- `compare_live_kws_replay.py` — replay captured audio through KWS
- `record_eval_samples.py` — record new eval WAVs per device
