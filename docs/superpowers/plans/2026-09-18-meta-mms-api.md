# Meta MMS API Implementation Plan

**Goal:** Build an asynchronous Meta MMS transcription API with speaker
diarization and the established Whisper service job contract.

**Spec:** `docs/superpowers/specs/2026-09-18-meta-mms-api-design.md`

## Tasks

1. Configure Python dependencies, maintain the supplied MMS adapter inventory,
   and normalize two- and three-letter language codes before job creation.
2. Implement atomic job storage and the response schema with conditional
   `language_code` or `language_code_3` fields.
3. Convert MMS CTC chunks to integer-millisecond words and map each word to
   MSDD/Sortformer speaker spans, including an empty-diarization fallback.
4. Reuse the reference MSDD and Sortformer components through a lazy factory so
   importing the API does not import NeMo.
5. Load mono 16 kHz audio, retain a single MMS model per worker, switch its
   adapter per job, request word timestamps, and use CTC overlap chunking.
6. Orchestrate persistent worker state changes, optional Demucs separation,
   inference, diarization, schema construction, and failure isolation.
7. Implement the FastAPI routes, bearer authentication, immediate `400`
   language validation, multiprocessing queue, and executable server options.
8. Cover API, language, persistence, mapping, pipeline, worker, and
   documentation contracts with Pytest; run the complete suite and Ruff.
