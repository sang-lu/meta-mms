from pathlib import Path


def test_readme_documents_required_language_and_both_response_keys():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "language" in text
    assert "language_code" in text
    assert "language_code_3" in text


def test_readme_documents_cpu_and_gpu_start_scripts():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "./run_cpu.sh" in text
    assert "./run_gpu.sh" in text
