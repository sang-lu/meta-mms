import sys

import pytest

from diarization import create_diarizer


def test_unknown_diarizer_is_rejected_without_importing_nemo():
    with pytest.raises(ValueError, match="Unknown diarizer"):
        create_diarizer("unknown", "cpu")
    assert "nemo" not in sys.modules

