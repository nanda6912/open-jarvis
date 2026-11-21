# Configuration Reference

## Layers (in increasing precedence)

1. `config/default_settings.json` — shipped defaults (edit carefully;
   this file is version-controlled).
2. `config/default_settings.json` values overridden by **user
   settings** persisted in the `user_settings` table of `voiceos.db`
   (monitor choice, microphone selection, HUD position/appearance,
   auto-start preference).
3. **Environment variables** (see below).
4. CLI flags (`run.py --startup ...`).

`ConfigurationManager` (`src/core/configuration.py`) validates every
value against a schema; missing files, malformed JSON and wrong types
fall back to defaults with logged warnings — the app never fails to
start because of a bad settings file.

## `default_settings.json` sections

### `application`

| Key | Default | Meaning |
|---|---|---|
| `name` | `VoiceOS` | App display name |
| `assistant_name` | `Jarvis` | Assistant display name |
| `version` | `1.0.0` | Version string |

### `voice`

| Key | Default | Meaning |
|---|---|---|
| `wake_word` | `Hello Jarvis` | Wake phrase (matched by KWS keyword list) |
| `activation_response` | `At your service` | Spoken on wake |
| `exit_phrase` | `Goodbye Jarvis` | Ends the session |
| `session_timeout_seconds` | `60` | Inactivity timeout |
| `listening_enabled` | `true` | Master voice enable |
| `microphone_enabled` | `true` | Mic capture enable |
| `wake_word_enabled` | `true` | KWS enable |
| `wake_word_threshold` | `0.25` | KWS confidence threshold |
| `stt_enabled` / `stt_engine` | `true` / `auto` | STT enable + model type (`auto`, `moonshine`, `transducer`, `whisper`, `nemo_ctc`) |
| `stt_model_dir` | `""` | Override STT model dir (empty = runtime default / env) |
| `stt_num_threads` | `2` | STT inference threads |
| `stt_speech_threshold` | `0.01` | VAD speech threshold |
| `tts_enabled` / `tts_engine` | `true` / `sherpa-onnx-vits` | TTS enable + engine |
| `tts_model_dir` | `""` | Override TTS model dir |
| `tts_num_threads` / `tts_speed` | `2` / `1.0` | TTS threads / speech rate |
| `ack_mode` | `full` | `full` / `quick` / `silent` acknowledgement |
| `quick_ack_response` | `Yes.` | Quick-mode acknowledgement text |

### `startup`

`auto_start` (`true`), `start_minimized` (`true`).

### `interface`

`theme` (`dark`), `show_notifications` (`true`).

### `hud`

`enabled`, `monitor` (`auto` — persisted per-user choice wins),
`size` (320), `position`, `position_screen`, `locked`,
`always_on_top`, `fullscreen`, `fps` (30), and `appearance` —
`preset` (`balanced`) plus `brightness`, `glow`, `opacity`, `ring`,
`particle`, `waveform`, `speed` (percentages).

### `storage`

`database_name` (`voiceos.db`) — file name only; path traversal is
rejected.

## Environment variables

Resolved by `src/core/paths.py` (and `.env` via python-dotenv —
existing process env always wins over `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `VOICEOS_HOME` | `%LOCALAPPDATA%\VoiceOS` | Runtime root |
| `VOICEOS_DATA_DIR` | — | Legacy alias for `VOICEOS_HOME` |
| `VOICEOS_KWS_MODEL_DIR` | `<home>\models\kws` | KWS models |
| `VOICEOS_STT_MODEL_DIR` | `<home>\models\stt` | STT models |
| `VOICEOS_TTS_MODEL_DIR` | `<home>\models\tts` | TTS models |
| `VOICEOS_DB_PATH` | `<home>\voiceos.db` | Main database |
| `VOICEOS_FILE_INDEX_DB_PATH` | `<home>\file_index.db` | File index |
| `VOICEOS_LOG_DIR` | `<home>\logs` | Logs |

Other runtime flags used by the code: `VOICEOS_BUILD_ID`
(speech-recognition diagnostics), `VOICEOS_KWS_PARITY_DIAGNOSTIC=1`
(enables the KWS parity recorder writing
`<home>\reports\v1074_production_kws_parity.json`).

## Runtime directories

See `docs/PORTABLE_INSTALLATION.md` for the full layout and migration
behavior.
