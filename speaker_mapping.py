import math


def _valid_timestamp(timestamp) -> tuple[float, float] | None:
    if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
        return None
    try:
        start, end = float(timestamp[0]), float(timestamp[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end):
        return None
    return start, end


def map_words_to_speakers(
    chunks: list[dict], speaker_spans: list[tuple[int, int, int]]
) -> list[dict]:
    valid_words = []
    for chunk in chunks:
        text = chunk.get("text")
        timestamp = _valid_timestamp(chunk.get("timestamp"))
        if not isinstance(text, str) or not text.strip() or timestamp is None:
            continue
        start, end = timestamp
        start_ms = max(0, round(start * 1000))
        end_ms = max(start_ms, round(end * 1000))
        valid_words.append((text.strip(), start_ms, end_ms))

    if not valid_words:
        return []
    if not speaker_spans:
        return [
            {
                "word": text,
                "start_time": start_ms,
                "end_time": end_ms,
                "speaker": 0,
            }
            for text, start_ms, end_ms in valid_words
        ]

    spans = sorted(speaker_spans, key=lambda span: span[0])
    span_index = 0
    mapped = []
    for text, start_ms, end_ms in valid_words:
        while span_index < len(spans) - 1 and start_ms >= spans[span_index][1]:
            span_index += 1
        mapped.append(
            {
                "word": text,
                "start_time": start_ms,
                "end_time": end_ms,
                "speaker": spans[span_index][2],
            }
        )
    return mapped
