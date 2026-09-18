import queue

import numpy as np

import worker
from jobstore import JobStore


class FakeDiarizer:
    def __init__(self, spans=None):
        self.spans = spans or [(0, 1000, 2)]

    def diarize(self, _audio):
        return self.spans


def _create_job(store, *, no_stem=True):
    return store.create_job(
        b"audio",
        ".wav",
        {
            "language": {
                "request_code": "vi",
                "adapter_code": "vie",
                "response_key": "language_code",
                "response_code": "vi",
            },
            "no_stem": no_stem,
            "batch_size": 8,
        },
    )


def test_worker_writes_a_completed_mms_result(monkeypatch, tmp_path):
    store = JobStore(str(tmp_path / "jobs"))
    job_id = _create_job(store)
    monkeypatch.setattr(worker.mms_pipeline, "load_audio", lambda _: (np.zeros(16_000), 16_000))
    monkeypatch.setattr(
        worker.mms_pipeline,
        "transcribe",
        lambda *args: ("xin chao", [{"text": "xin", "timestamp": (0.0, 0.2)}]),
    )
    monkeypatch.setattr(worker, "_diarize", lambda diarizer, waveform: diarizer.diarize(waveform))

    worker.process_one_job(object(), FakeDiarizer(), store, job_id, temp_dir=str(tmp_path / "tmp"))

    assert store.get_status(job_id) == {"status": "completed"}
    assert store.get_result(job_id)["language_code"] == "vi"
    assert store.get_result(job_id)["utterances"][0]["speaker"] == "A"


def test_worker_no_stem_false_uses_demucs_output(monkeypatch, tmp_path):
    store = JobStore(str(tmp_path / "jobs"))
    job_id = _create_job(store, no_stem=False)
    selected = []
    monkeypatch.setattr(
        worker.mms_pipeline,
        "separate_vocals",
        lambda audio, output_dir, device: (
            selected.append((audio, output_dir, device)) or "vocals.wav"
        ),
    )
    monkeypatch.setattr(
        worker.mms_pipeline,
        "load_audio",
        lambda path: (selected.append(path) or np.zeros(16_000), 16_000),
    )
    monkeypatch.setattr(worker.mms_pipeline, "transcribe", lambda *args: ("", []))
    monkeypatch.setattr(worker, "_diarize", lambda *_: [])

    worker.process_one_job(
        type("Models", (), {"device": "cpu"})(),
        FakeDiarizer(),
        store,
        job_id,
        temp_dir="tmp",
    )

    assert selected[0][0].endswith("audio.wav")
    assert selected[1] == "vocals.wav"


def test_worker_empty_diarization_uses_fallback_speaker(monkeypatch, tmp_path):
    store = JobStore(str(tmp_path / "jobs"))
    job_id = _create_job(store)
    monkeypatch.setattr(worker.mms_pipeline, "load_audio", lambda _: (np.zeros(16_000), 16_000))
    monkeypatch.setattr(
        worker.mms_pipeline,
        "transcribe",
        lambda *args: ("hello", [{"text": "hello", "timestamp": (0.0, 0.1)}]),
    )
    monkeypatch.setattr(worker, "_diarize", lambda *_: [])

    worker.process_one_job(object(), FakeDiarizer(), store, job_id, temp_dir=str(tmp_path / "tmp"))

    assert store.get_result(job_id)["utterances"][0]["speaker"] == "A"


def test_worker_persists_inference_exception_as_failed(monkeypatch, tmp_path):
    store = JobStore(str(tmp_path / "jobs"))
    job_id = _create_job(store)
    def raise_error(_path):
        raise RuntimeError("boom")

    monkeypatch.setattr(worker.mms_pipeline, "load_audio", raise_error)

    worker.process_one_job(object(), FakeDiarizer(), store, job_id, temp_dir=str(tmp_path / "tmp"))

    assert store.get_status(job_id) == {"status": "failed", "error": "boom"}


def test_worker_loop_stops_on_none_without_processing_another_job(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(worker.mms_pipeline, "load_models", lambda *args: object())
    monkeypatch.setattr(worker, "create_diarizer", lambda *args: object())
    monkeypatch.setattr(worker, "process_one_job", lambda *args, **kwargs: calls.append(args[3]))
    jobs = queue.Queue()
    jobs.put(None)

    worker.worker_loop(jobs, str(tmp_path), "model", "cpu", "msdd")

    assert calls == []
