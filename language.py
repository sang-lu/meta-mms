from dataclasses import dataclass

import pycountry

from mms_languages import SUPPORTED_MMS_ADAPTERS


@dataclass(frozen=True)
class LanguageSelection:
    request_code: str
    adapter_code: str
    response_key: str
    response_code: str


def normalize_language(value: str | None) -> LanguageSelection:
    code = (value or "").strip().lower()
    if len(code) == 2 and code.isalpha():
        language = pycountry.languages.get(alpha_2=code)
        adapter = getattr(language, "alpha_3", None)
        if adapter in SUPPORTED_MMS_ADAPTERS:
            return LanguageSelection(code, adapter, "language_code", code)
    elif len(code) == 3 and code.isalpha() and code in SUPPORTED_MMS_ADAPTERS:
        return LanguageSelection(code, code, "language_code_3", code)
    raise ValueError("Unsupported language code")
