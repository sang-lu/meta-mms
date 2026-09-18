import json

import pytest

from jobstore import ALLOWED_AUDIO_EXTENSIONS, JobStore


def test_create_job_writes_audio_params_and_queued_status(tmp_path):
    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"fake-audio-bytes", ".wav", {"language": "en"})

    job_dir = tmp_path / job_id
    assert (job_dir / "audio.wav").read_bytes() == b"fake-audio-bytes"
    assert json.loads((job_dir / "params.json").read_text()) == {"language": "en"}
    assert json.loads((job_dir / "status.json").read_text()) == {"status": "queued"}


def test_create_job_rejects_disallowed_extension(tmp_path):
    with pytest.raises(ValueError):
        JobStore(str(tmp_path)).create_job(b"data", ".exe", {})


def test_audio_path_returns_saved_file(tmp_path):
    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"data", ".mp3", {})
    assert store.audio_path(job_id) == str(tmp_path / job_id / "audio.mp3")


def test_missing_job_operations_raise_file_not_found(tmp_path):
    store = JobStore(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        store.audio_path("does-not-exist")
    with pytest.raises(FileNotFoundError):
        store.delete_job("does-not-exist")


def test_status_and_result_round_trip(tmp_path):
    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"data", ".wav", {})
    store.set_status(job_id, {"status": "processing"})
    result = {"id": job_id, "text": "hello world"}
    store.set_result(job_id, result)
    assert store.get_status(job_id) == {"status": "processing"}
    assert store.get_result(job_id) == result


def test_status_write_is_atomic(tmp_path):
    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"data", ".wav", {})
    store.set_status(job_id, {"status": "completed"})
    job_dir = tmp_path / job_id
    assert not (job_dir / "status.json.tmp").exists()
    assert (job_dir / "status.json").exists()


def test_job_exists_and_recursive_delete(tmp_path):
    store = JobStore(str(tmp_path))
    job_id = store.create_job(b"data", ".wav", {})
    (tmp_path / job_id / "nested").mkdir()
    (tmp_path / job_id / "nested" / "file").write_text("data")
    assert store.job_exists(job_id)
    store.delete_job(job_id)
    assert not store.job_exists(job_id)


def test_allowed_extensions_match_reference_contract():
    assert ALLOWED_AUDIO_EXTENSIONS == {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".opus",
        ".m4a",
        ".mp4",
        ".webm",
    }
