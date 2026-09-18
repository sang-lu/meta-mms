from language import normalize_language
from schema import build_result


def _wsm(*rows):
    return [
        {"word": word, "start_time": start, "end_time": end, "speaker": speaker}
        for word, start, end, speaker in rows
    ]


def test_schema_uses_the_selected_two_letter_response_field():
    result = build_result(
        "job",
        normalize_language("vi"),
        1,
        [{"word": "xin", "start_time": 0, "end_time": 100, "speaker": 0}],
    )
    assert result["language_code"] == "vi"
    assert "language_code_3" not in result


def test_schema_uses_the_selected_three_letter_response_field():
    result = build_result("job", normalize_language("vie"), 1, [])
    assert result["language_code_3"] == "vie"
    assert "language_code" not in result


def test_single_speaker_groups_words_into_one_utterance():
    result = build_result(
        "job-1",
        normalize_language("eng"),
        1,
        _wsm(("Hello", 0, 200, 0), ("world.", 200, 500, 0)),
    )
    assert result["text"] == "Hello world."
    assert result["utterances"] == [
        {
            "speaker": "A",
            "start": 0,
            "end": 500,
            "words": [
                {"text": "Hello", "start": 0, "end": 200, "speaker": "A"},
                {"text": "world.", "start": 200, "end": 500, "speaker": "A"},
            ],
            "text": "Hello world.",
        }
    ]


def test_speaker_changes_split_utterances_and_letters_follow_first_appearance():
    result = build_result(
        "job-1",
        normalize_language("eng"),
        1,
        _wsm(("Hi", 0, 100, 7), ("there", 100, 200, 3), ("again", 200, 300, 7)),
    )
    assert [utterance["speaker"] for utterance in result["utterances"]] == ["A", "B", "A"]


def test_empty_words_produce_empty_text_and_utterances():
    result = build_result("job-1", normalize_language("eng"), 0, [])
    assert result["text"] == ""
    assert result["utterances"] == []
