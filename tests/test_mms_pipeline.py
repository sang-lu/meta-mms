import numpy as np

import mms_pipeline
from mms_pipeline import MMSModels


class FakeTokenizer:
    def __init__(self):
        self.target_lang = None

    def set_target_lang(self, language):
        self.target_lang = language


class FakeModel:
    def __init__(self):
        self.loaded_adapter = None
        self.load_adapter_calls = []

    def load_adapter(self, language):
        self.loaded_adapter = language
        self.load_adapter_calls.append(language)


class FakeASR:
    def __init__(self):
        self.calls = []

    def __call__(self, waveform, **kwargs):
        self.calls.append((waveform, kwargs))
        return {"text": "xin chao", "chunks": [{"text": "xin", "timestamp": (0.0, 0.2)}]}


class FakeModels:
    def __init__(self):
        self.model = FakeModel()
        self.processor = type("Processor", (), {"tokenizer": FakeTokenizer()})()
        self.asr = FakeASR()
        self.current_adapter = None


def test_transcribe_switches_adapter_then_requests_word_timestamps():
    models = FakeModels()
    text, chunks = mms_pipeline.transcribe(
        models, np.zeros(16_000, dtype=np.float32), "vie", 8
    )
    assert models.processor.tokenizer.target_lang == "vie"
    assert models.model.loaded_adapter == "vie"
    assert text == "xin chao"
    assert chunks[0]["timestamp"] == (0.0, 0.2)
    assert models.asr.calls[0][1] == {
        "return_timestamps": "word",
        "chunk_length_s": 30,
        "stride_length_s": 5,
        "batch_size": 8,
    }


def test_transcribe_does_not_reload_the_current_adapter():
    models = FakeModels()
    models.current_adapter = "eng"
    mms_pipeline.transcribe(models, np.zeros(10, dtype=np.float32), "eng", 8)
    assert models.model.load_adapter_calls == []


def test_loaded_models_retain_device_for_audio_separation():
    models = MMSModels(object(), object(), object(), device="cuda")
    assert models.device == "cuda"


class FakeWaveform:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape

    def mean(self, dim):
        assert dim == 0
        return FakeWaveform(self.values.mean(axis=0))

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeAudioBackend:
    class functional:
        @staticmethod
        def resample(waveform, source_rate, target_rate):
            assert source_rate == 8_000
            assert target_rate == 16_000
            return FakeWaveform(waveform.values.repeat(2))

    @staticmethod
    def load(path):
        assert path == "audio.wav"
        return FakeWaveform([[0.0, 1.0], [1.0, 0.0]]), 8_000


def test_load_audio_converts_stereo_resamples_and_returns_contiguous_float32(monkeypatch):
    monkeypatch.setattr(mms_pipeline, "_load_torchaudio", lambda: FakeAudioBackend)
    waveform, rate = mms_pipeline.load_audio("audio.wav")
    assert rate == 16_000
    assert waveform.dtype == np.float32
    assert waveform.flags.c_contiguous
    assert waveform.tolist() == [0.5, 0.5, 0.5, 0.5]


def test_load_audio_falls_back_to_soundfile_when_torchaudio_decoder_is_unavailable(
    monkeypatch,
):
    class BrokenAudioBackend:
        @staticmethod
        def load(_path):
            raise ImportError("torchcodec unavailable")

    class FakeSoundFile:
        @staticmethod
        def read(_path, always_2d, dtype):
            assert always_2d is True
            assert dtype == "float32"
            return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32), 16_000

    class FakeTorch:
        @staticmethod
        def from_numpy(values):
            return FakeWaveform(values)

    monkeypatch.setattr(mms_pipeline, "_load_torchaudio", lambda: BrokenAudioBackend)
    monkeypatch.setattr(mms_pipeline, "_load_soundfile", lambda: FakeSoundFile)
    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)
    waveform, rate = mms_pipeline.load_audio("audio.wav")
    assert rate == 16_000
    assert waveform.tolist() == [0.5, 0.5]
