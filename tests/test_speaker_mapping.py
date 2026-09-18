import math

from speaker_mapping import map_words_to_speakers


def test_words_are_assigned_using_their_start_timestamp():
    words = map_words_to_speakers(
        [
            {"text": "hello ", "timestamp": (0.05, 0.20)},
            {"text": "there", "timestamp": (0.25, 0.40)},
        ],
        [(0, 220, 4), (220, 500, 9)],
    )
    assert words == [
        {"word": "hello", "start_time": 50, "end_time": 200, "speaker": 4},
        {"word": "there", "start_time": 250, "end_time": 400, "speaker": 9},
    ]


def test_empty_diarization_uses_a_single_fallback_speaker():
    words = map_words_to_speakers(
        [{"text": "hello", "timestamp": (0.0, 0.1)}],
        [],
    )
    assert words[0]["speaker"] == 0


def test_invalid_chunks_are_filtered_and_times_are_clamped():
    words = map_words_to_speakers(
        [
            {"text": "  ", "timestamp": (0.0, 0.1)},
            {"text": "bad", "timestamp": (math.nan, 0.1)},
            {"text": "word", "timestamp": (-0.01, -0.02)},
            {"text": "missing"},
        ],
        [(0, 100, 3)],
    )
    assert words == [{"word": "word", "start_time": 0, "end_time": 0, "speaker": 3}]


def test_final_speaker_is_retained_after_final_span():
    words = map_words_to_speakers(
        [
            {"text": "one", "timestamp": (0.0, 0.1)},
            {"text": "two", "timestamp": (2.0, 2.1)},
        ],
        [(0, 100, 8)],
    )
    assert [word["speaker"] for word in words] == [8, 8]

