from pathlib import Path


def test_readme_documents_required_language_and_both_response_keys():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "language" in text
    assert "language_code" in text
    assert "language_code_3" in text

