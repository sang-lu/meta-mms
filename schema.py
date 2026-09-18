from language import LanguageSelection


def build_result(
    job_id: str,
    language: LanguageSelection,
    audio_duration_seconds: int,
    words: list[dict],
) -> dict:
    speaker_letters: dict[int, str] = {}
    utterances = []
    current = None

    for word in words:
        speaker_id = word["speaker"]
        if speaker_id not in speaker_letters:
            speaker_letters[speaker_id] = chr(ord("A") + len(speaker_letters))
        speaker = speaker_letters[speaker_id]
        if current is None or speaker != current["speaker"]:
            if current is not None:
                utterances.append(current)
            current = {
                "speaker": speaker,
                "start": word["start_time"],
                "end": word["end_time"],
                "words": [],
            }
        current["end"] = word["end_time"]
        current["words"].append(
            {
                "text": word["word"],
                "start": word["start_time"],
                "end": word["end_time"],
                "speaker": speaker,
            }
        )

    if current is not None:
        utterances.append(current)
    for utterance in utterances:
        utterance["text"] = " ".join(word["text"] for word in utterance["words"])

    return {
        "id": job_id,
        "status": "completed",
        language.response_key: language.response_code,
        "audio_duration": audio_duration_seconds,
        "text": " ".join(word["word"] for word in words),
        "utterances": utterances,
    }
