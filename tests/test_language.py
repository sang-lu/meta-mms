import pytest

from language import normalize_language


def test_two_letter_code_maps_to_mms_adapter_and_legacy_response_key():
    selection = normalize_language("VI")
    assert selection.adapter_code == "vie"
    assert selection.response_key == "language_code"
    assert selection.response_code == "vi"


def test_three_letter_code_uses_three_letter_response_key():
    selection = normalize_language("eng")
    assert selection.adapter_code == "eng"
    assert selection.response_key == "language_code_3"
    assert selection.response_code == "eng"


@pytest.mark.parametrize("value", [None, "", "english", "zz", "zzz", "vie-script_latin"])
def test_invalid_or_unsupported_language_is_rejected(value):
    with pytest.raises(ValueError, match="Unsupported language code"):
        normalize_language(value)


def test_language_selection_normalizes_whitespace_and_case():
    selection = normalize_language("  Vie  ")
    assert selection.request_code == "vie"
    assert selection.adapter_code == "vie"

