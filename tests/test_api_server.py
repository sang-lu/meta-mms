import queue
import subprocess
import sys

from fastapi.testclient import TestClient

from api_server import create_app
from jobstore import JobStore


def _client(tmp_path, token=None):
    store = JobStore(str(tmp_path))
    jobs = queue.Queue()
    return TestClient(create_app(store, jobs, token=token)), store, jobs


def _file():
    return {"file": ("audio.wav", b"audio", "audio/wav")}


def test_post_requires_language_and_does_not_create_a_job(tmp_path):
    client, _store, _queue = _client(tmp_path)
    response = client.post("/jobs", files=_file())
    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_post_rejects_unsupported_language_before_queueing(tmp_path):
    client, _store, queue_ = _client(tmp_path)
    response = client.post("/jobs", files=_file(), data={"language": "zzz"})
    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []
    assert queue_.empty()


def test_post_stores_normalized_language_and_omits_suppress_numerals(tmp_path):
    client, store, _queue = _client(tmp_path)
    response = client.post(
        "/jobs",
        files=_file(),
        data={"language": "VI", "suppress_numerals": "true"},
    )
    assert response.status_code == 202
    params = store.params(response.json()["job_id"])
    assert params["language"]["adapter_code"] == "vie"
    assert "suppress_numerals" not in params


def test_post_accepts_three_letter_language_and_defaults_options(tmp_path):
    client, store, _queue = _client(tmp_path)
    response = client.post("/jobs", files=_file(), data={"language": "ENG"})
    params = store.params(response.json()["job_id"])
    assert params == {
        "language": {
            "request_code": "eng",
            "adapter_code": "eng",
            "response_key": "language_code_3",
            "response_code": "eng",
        },
        "no_stem": True,
        "batch_size": 8,
    }


def test_post_rejects_invalid_extension_before_creating_a_job(tmp_path):
    client, _store, jobs = _client(tmp_path)
    response = client.post(
        "/jobs",
        files={"file": ("audio.exe", b"x", "application/octet-stream")},
        data={"language": "vi"},
    )
    assert response.status_code == 400
    assert "Unsupported audio extension" in response.json()["detail"]
    assert jobs.empty()


def test_all_language_validation_errors_are_bad_request(tmp_path):
    for value in ["english", "zz", "zzz"]:
        client, _store, jobs = _client(tmp_path / value)
        response = client.post("/jobs", files=_file(), data={"language": value})
        assert response.status_code == 400
        assert response.json()["detail"] == "Unsupported language code"
        assert jobs.empty()


def test_status_completed_envelope_and_unknown_job(tmp_path):
    client, store, _queue = _client(tmp_path)
    response = client.post("/jobs", files=_file(), data={"language": "eng"})
    job_id = response.json()["job_id"]
    store.set_result(job_id, {"id": job_id, "status": "completed"})
    store.set_status(job_id, {"status": "completed"})
    assert client.get(f"/jobs/{job_id}").json() == {
        "job_id": job_id,
        "status": "completed",
        "result": {"id": job_id, "status": "completed"},
    }
    assert client.get("/jobs/missing").status_code == 404


def test_delete_removes_job_and_unknown_delete_is_not_found(tmp_path):
    client, _store, _queue = _client(tmp_path)
    job_id = client.post("/jobs", files=_file(), data={"language": "eng"}).json()["job_id"]
    assert client.delete(f"/jobs/{job_id}").status_code == 204
    assert client.get(f"/jobs/{job_id}").status_code == 404
    assert client.delete(f"/jobs/{job_id}").status_code == 404


def test_configured_bearer_token_is_required(tmp_path):
    client, _store, _queue = _client(tmp_path, token="secret")
    assert client.get("/jobs/missing").status_code == 401
    assert client.get("/jobs/missing", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert (
        client.get("/jobs/missing", headers={"Authorization": "Bearer secret"}).status_code
        == 404
    )


def test_importing_api_server_does_not_load_model_modules():
    script = (
        "import api_server, sys; "
        "leaked = {'torch', 'transformers', 'nemo'} & set(sys.modules); "
        "assert not leaked, leaked"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
