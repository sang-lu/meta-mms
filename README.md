# Meta MMS Transcription API

This service exposes an asynchronous, speaker-diarized transcription API backed
by `facebook/mms-1b-all` and the MSDD or Sortformer diarizer.

## Installation

Use Python 3.10 or newer. Install the project and its runtime dependencies in a
virtual environment. PyTorch and NeMo may require platform-specific installation
instructions before installing this project.

```bash
python -m pip install -e .[test]
```

The first worker start downloads the MMS model and the selected diarizer models.
Workers need a working PyTorch, Torchaudio, FFmpeg/Demucs, Transformers, and
NeMo installation. Audio is converted to mono 16 kHz before inference.

## Server

### Automatic setup scripts

On Debian or Ubuntu, the scripts create `.venv`, install missing FFmpeg and
libsndfile packages through `sudo` when necessary, install project dependencies,
and start the server. Any arguments after the script name are passed to
`api_server.py`.

```bash
./run_cpu.sh --port 8000 --token change-me
```

For an NVIDIA GPU, install a compatible NVIDIA driver first. The GPU script
checks `nvidia-smi`, installs the CUDA 12.8 PyTorch wheel, verifies that PyTorch
can use CUDA, and starts the server with `--device cuda`.

```bash
./run_gpu.sh --port 8000 --diarizer msdd --token change-me
```

Preview either script without changing the system or creating a virtual
environment:

```bash
./run_cpu.sh --dry-run
./run_gpu.sh --dry-run
```

```bash
python api_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model facebook/mms-1b-all \
  --device cpu \
  --diarizer msdd \
  --max-parallel 1 \
  --jobs-dir ./jobs \
  --token change-me
```

The server also accepts `--device cuda`, `--diarizer sortformer`, and the
operational options `--host`, `--port`, `--max-parallel`, `--jobs-dir`, and
`--token`. When `--token` is configured, every request must include
`Authorization: Bearer <token>`.

## Jobs

`POST /jobs` requires a multipart `file` and `language`. The supported audio
extensions are `.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.mp4`, and
`.webm`. `no_stem` defaults to `true`; set it to `false` to run Demucs vocal
separation. `batch_size` defaults to `8`.

Language input may be a two-letter ISO 639-1 code such as `vi`, or a bare,
three-letter ISO 639-3 code such as `vie`. Input is normalized to lowercase and
validated against the static MMS adapter inventory before a job is created.
Invalid, unmappable, or unsupported language values return `400` immediately.

```bash
curl -X POST http://localhost:8000/jobs \
  -H 'Authorization: Bearer change-me' \
  -F 'file=@recording.wav' \
  -F 'language=vi' \
  -F 'no_stem=true' \
  -F 'batch_size=8'
```

The response is queued immediately:

```json
{"job_id":"<job-id>","status":"queued"}
```

Poll the job:

```bash
curl -H 'Authorization: Bearer change-me' \
  http://localhost:8000/jobs/<job-id>
```

Completed results contain integer-millisecond timestamps. A two-letter request
uses `language_code`:

```json
{
  "job_id": "<job-id>",
  "status": "completed",
  "result": {
    "id": "<job-id>",
    "status": "completed",
    "language_code": "vi",
    "audio_duration": 12,
    "text": "xin chao",
    "utterances": []
  }
}
```

A three-letter request such as `vie` uses `language_code_3` instead. Only the
field matching the submitted language form is present. Each utterance contains
`speaker`, `start`, `end`, `text`, and `words`; each word contains `text`,
`start`, `end`, and `speaker`.

Delete a job and its persisted files:

```bash
curl -X DELETE \
  -H 'Authorization: Bearer change-me' \
  http://localhost:8000/jobs/<job-id>
```

`suppress_numerals` is not part of this API. If sent as an extra multipart field,
it has no effect and is not persisted.
