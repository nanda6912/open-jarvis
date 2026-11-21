# Known Limitations & Verification Status

This file records what is **verified** versus **partially validated**.
Do not upgrade a status without physical evidence.

## Verified on physical hardware

- Dell monitor detection and automatic HUD fallback — physically
  verified.
- Wake-word pipeline on Intel array + Realtek headset microphones —
  physical recordings under `tests/fixtures/physical/intel/` and
  `commands/phys_*.wav`.
- Voice sessions end-to-end (wake → command → TTS → standby) on the
  development laptop.
- Single-instance lock, tray behavior, Startup-folder registration.
- File/folder discovery and safe open flows.

## Partially validated — pending physical verification

- **Balanced HUD appearance preset** — automated tests plus a
  real-GPU smoke test reported; complete physical visual validation
  is still pending.
- **HUD staying above Chrome** — previously reported as *not working*;
  no successful physical re-test is on record, so it is not claimed
  fixed.
- **HUD click-through and dragging** — implemented and unit-tested;
  full desktop interaction validation pending.
- **STT accuracy** — known transcription limitations remain (see the
  `hey_jarvis.wav` → "Hage-arvis" fixture failure). Fuzzy recovery
  mitigates mishearings but does not eliminate them.

## Platform limits

- Windows 10/11 x64 only. No Linux/macOS support; several controllers
  are Win32-specific (Toolhelp32, EnumWindows, power broadcasts,
  pycaw endpoint audio).
- English-only models.
- "Laptop screen" monitor selection is a heuristic — Windows does not
  expose panel type; it resolves best-effort to the primary display.
- Models are not distributed (~1.5 GB manual install).

## Operational notes

- If no STT/TTS/KWS model is present, the app runs with null engines —
  wake-word, recognition and speech are inactive (logged warnings).
- The legacy project-local `data\file_index.db` is migrated once to the
  runtime dir on first run after upgrade; the old file is left in
  place (never deleted by the app).
- `VOICEOS_KWS_PARITY_DIAGNOSTIC=1` enables extra KWS recording —
  diagnostic only, off by default.
