# Meta MMS Transcription and Diarization API Design

## Purpose

Build a standalone HTTP service that transcribes audio with Meta MMS and assigns
speaker labels with the diarization engine used by `whisper-diarization`. Its
public job API and completed-result structure match that service so existing
clients can use it without changing their job lifecycle handling.

## Scope

The service provides asynchronous audio jobs, persistent on-disk job state,
optional bearer-token authentication, MMS word timestamps, and speaker
diarization. It supports the same audio extensions, job states, routes,
polling behaviour, deletion behaviour, queueing, `no_stem`, `batch_size`, and
millisecond timestamp convention as `whisper-diarization`.

It does not implement Whisper, Whisper language detection, Whisper forced
alignment, or `suppress_numerals`.

## Public API

- `POST /jobs` accepts multipart fields `file`, required `language`, optional
  `no_stem` (default `true`), and optional `batch_size` (default `8`).
- `GET /jobs/{job_id}` returns the existing queued, processing, failed, or
  completed job envelopes.
- `DELETE /jobs/{job_id}` removes all persisted job files and returns `204`.

`language` is normalized to lowercase and must be a two-letter ISO 639-1 code
or a bare three-letter ISO 639-3 code. Two-letter codes are mapped to a
supported MMS adapter before job creation. Invalid, unmappable, and unsupported
codes return `400` immediately.

Two-letter requests include only `language_code` in their completed result.
Three-letter requests include only `language_code_3`. The result otherwise
contains the established `id`, `status`, `audio_duration`, `text`, and
`utterances` schema with integer-millisecond word and utterance timestamps.

`suppress_numerals` is not parsed, persisted, or used.

## Processing

The API process validates and persists jobs without importing model modules.
Each worker converts input to mono 16 kHz, optionally separates vocals with
Demucs, switches the MMS CTC adapter for the validated target language, obtains
word timestamps, runs MSDD or Sortformer diarization on the same waveform, maps
words to speakers, and persists the completed result. Long audio uses CTC
chunking with overlap. A processing error persists a failed status without
terminating the worker process.

## Verification

Tests cover API lifecycle and authentication, language validation, response-key
selection, atomic job persistence, timestamp and speaker mapping, worker
success/failure isolation, and dependency-light API imports. Model and NeMo
boundaries are mocked for deterministic tests.
