# VoiceOS Data and Audio Privacy

This document distinguishes normal application behavior from the development
and evaluation assets distributed in this repository.

## Normal application runtime

- Microphone frames are processed in memory by the audio capture, wake-word,
  and speech-recognition pipeline. Normal voice operation does not
  automatically write raw microphone audio to disk.
- The local SQLite command-history store records recognized transcripts,
  command types, outcomes, and timestamps. Treat this history as potentially
  sensitive user data.
- The wake-word diagnostic ring buffer is an in-memory diagnostic facility;
  it is not a replacement for an explicit recording workflow.
- VoiceOS performs speech inference locally with the configured KWS, STT, and
  TTS models. The optional `SEARCH_WEB` command opens a search URL in the
  user's browser and therefore requires an internet connection.

## Explicit evaluation recordings

The repository includes WAV fixtures used for command recognition, negative
wake-word checks, and physical microphone evaluation. These files are
committed test data and are publicly downloadable from this repository.

The script `scripts/record_eval_samples.py` is an explicit, user-invoked
recording utility. Its documented output is written under
`tests/fixtures/physical/<device>/`; it is not part of continuous application
operation. Existing files are not overwritten without `--force` and an
interactive confirmation.

Before adding or redistributing a recording:

1. Confirm that the speaker and anyone else whose speech is captured has
   agreed to the recording being shared publicly.
2. Check for names, notifications, conversations, account details, or other
   identifying information in the audio.
3. Keep only recordings that are needed for a reproducible test.
4. Update the relevant manifest and document the purpose of the sample.

A text manifest does not prove that an audio recording is safe to publish.
The existing public WAV fixtures should be reviewed by the repository owner;
this document does not assert that every voice in every fixture has been
independently consent-verified.

## Local files that must not be committed

Do not commit:

- `.env` files, credentials, API keys, or access tokens.
- Runtime databases or command-history databases.
- Runtime logs or diagnostic captures containing private information.
- Personal screenshots, microphone recordings, or temporary evaluation data.
- Installed model directories unless redistribution rights and upstream
  licensing have been checked.

The repository's `.gitignore` excludes the primary runtime and secret paths,
but every proposed commit still needs a staged-diff review. A file ignored
today may already exist in Git history from an earlier commit.

## Removal from Git history

Deleting an audio file in a later commit stops it appearing in the current
tree, but does not remove earlier copies from Git history or existing clones.
If a recording is found to be published without permission, stop further
distribution and assess whether history removal and any required notification
are appropriate. Do not rewrite public history casually.
