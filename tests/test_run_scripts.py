import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dry_run(script: str, *server_args: str) -> str:
    result = subprocess.run(
        ["bash", str(ROOT / script), "--dry-run", *server_args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_cpu_script_dry_run_selects_cpu_and_forwards_server_arguments():
    output = _dry_run("run_cpu.sh", "--port", "9001")

    assert "https://download.pytorch.org/whl/cpu" in output
    assert "torchcodec" in output
    assert "api_server.py --device cpu --port 9001" in output


def test_gpu_script_dry_run_selects_cuda_and_forwards_server_arguments():
    output = _dry_run("run_gpu.sh", "--diarizer", "sortformer")

    assert "https://download.pytorch.org/whl/cu128" in output
    assert "flash-attn --no-build-isolation" in output
    assert "api_server.py --device cuda --diarizer sortformer" in output


def test_project_metadata_supports_editable_install_without_dependencies(tmp_path):
    environment = tmp_path / "venv"
    subprocess.run(["python3", "-m", "venv", str(environment)], check=True)

    subprocess.run(
        [str(environment / "bin" / "pip"), "install", "--no-deps", "-e", str(ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
