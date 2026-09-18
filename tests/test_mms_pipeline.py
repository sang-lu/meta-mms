import sys

import numpy as np
import pytest

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


def test_load_models_uses_bfloat16_and_flash_attention_on_supported_cuda(monkeypatch):
    class FakeCUDA:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported(**_kwargs):
            return True

        @staticmethod
        def empty_cache():
            return None

    class FakeTorch:
        bfloat16 = "bfloat16"
        cuda = FakeCUDA()

    class FakeProcessor:
        tokenizer = object()
        feature_extractor = object()

        @staticmethod
        def from_pretrained(model_id):
            assert model_id == "facebook/mms-1b-all"
            return FakeProcessor()

    class FakeModel:
        def to(self, device):
            assert device == "cuda"
            return self

    class FakeModelLoader:
        loaded_kwargs = None

        @staticmethod
        def from_pretrained(model_id, **kwargs):
            assert model_id == "facebook/mms-1b-all"
            FakeModelLoader.loaded_kwargs = kwargs
            return FakeModel()

    class FakeTransformers:
        AutoModelForCTC = FakeModelLoader
        AutoProcessor = FakeProcessor

        @staticmethod
        def pipeline(*_args, **_kwargs):
            return object()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "transformers", FakeTransformers)
    monkeypatch.setattr(
        mms_pipeline, "_flash_attention_available", lambda: True, raising=False
    )

    mms_pipeline.load_models("facebook/mms-1b-all", "cuda")

    assert FakeModelLoader.loaded_kwargs == {
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
    }


def test_model_load_kwargs_falls_back_to_fp32_when_bfloat16_is_unsupported(caplog):
    class FakeCUDA:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported(**_kwargs):
            return False

    class FakeTorch:
        cuda = FakeCUDA()

    assert mms_pipeline._model_load_kwargs("cuda", FakeTorch) == {}
    assert "BF16 is unsupported" in caplog.text


def test_model_load_kwargs_falls_back_to_fp32_with_older_pytorch(caplog):
    class FakeCUDA:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeCUDA()

    assert mms_pipeline._model_load_kwargs("cuda", FakeTorch) == {}
    assert "BF16 is unsupported" in caplog.text


def test_model_load_kwargs_rejects_bfloat16_emulation(caplog):
    class FakeCUDA:
        calls = []

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported(*, including_emulation=True):
            FakeCUDA.calls.append(including_emulation)
            return including_emulation

    class FakeTorch:
        cuda = FakeCUDA()

    assert mms_pipeline._model_load_kwargs("cuda", FakeTorch) == {}
    assert FakeCUDA.calls == [False]
    assert "BF16 is unsupported" in caplog.text


def test_model_load_kwargs_uses_bfloat16_without_flash_attention_when_unavailable(
    monkeypatch, caplog
):
    class FakeCUDA:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported(**_kwargs):
            return True

    class FakeTorch:
        bfloat16 = "bfloat16"
        cuda = FakeCUDA()

    monkeypatch.setattr(mms_pipeline, "_flash_attention_available", lambda: False)

    assert mms_pipeline._model_load_kwargs("cuda", FakeTorch) == {
        "torch_dtype": "bfloat16"
    }
    assert "Flash Attention 2 is unavailable" in caplog.text


def test_load_ctc_model_retries_without_flash_attention_when_backend_rejects_it(
    monkeypatch, caplog
):
    class FakeCUDA:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported(**_kwargs):
            return True

    class FakeTorch:
        bfloat16 = "bfloat16"
        cuda = FakeCUDA()

    class FakeModelLoader:
        calls = []

        @staticmethod
        def from_pretrained(model_id, **kwargs):
            FakeModelLoader.calls.append((model_id, kwargs))
            if kwargs.get("attn_implementation") == "flash_attention_2":
                raise ImportError("Flash Attention 2 is not installed")
            return object()

    monkeypatch.setattr(mms_pipeline, "_flash_attention_available", lambda: True)

    model = mms_pipeline._load_ctc_model(
        FakeModelLoader, "facebook/mms-1b-all", "cuda", FakeTorch
    )

    assert model is not None
    assert FakeModelLoader.calls == [
        (
            "facebook/mms-1b-all",
            {"torch_dtype": "bfloat16", "attn_implementation": "flash_attention_2"},
        ),
        (
            "facebook/mms-1b-all",
            {"torch_dtype": "bfloat16", "attn_implementation": "eager"},
        ),
    ]
    assert "continuing with eager attention" in caplog.text


def test_load_ctc_model_retries_in_fp32_when_bfloat16_loading_fails(
    monkeypatch, caplog
):
    class FakeCUDA:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported(**_kwargs):
            return True

    class FakeTorch:
        bfloat16 = "bfloat16"
        cuda = FakeCUDA()

    class FakeModelLoader:
        calls = []

        @staticmethod
        def from_pretrained(model_id, **kwargs):
            FakeModelLoader.calls.append((model_id, kwargs))
            if kwargs.get("torch_dtype") == "bfloat16":
                raise RuntimeError("BFloat16 is unsupported")
            return object()

    monkeypatch.setattr(mms_pipeline, "_flash_attention_available", lambda: False)

    model = mms_pipeline._load_ctc_model(
        FakeModelLoader, "facebook/mms-1b-all", "cuda", FakeTorch
    )

    assert model is not None
    assert FakeModelLoader.calls == [
        ("facebook/mms-1b-all", {"torch_dtype": "bfloat16"}),
        ("facebook/mms-1b-all", {}),
    ]
    assert "continuing in FP32" in caplog.text


def test_load_ctc_model_retries_without_attention_argument_when_transformers_rejects_it(
    monkeypatch, caplog
):
    class FakeCUDA:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported(**_kwargs):
            return True

    class FakeTorch:
        bfloat16 = "bfloat16"
        cuda = FakeCUDA()

    class FakeModelLoader:
        calls = []

        @staticmethod
        def from_pretrained(model_id, **kwargs):
            FakeModelLoader.calls.append((model_id, kwargs))
            if "attn_implementation" in kwargs:
                raise TypeError("unexpected keyword argument 'attn_implementation'")
            return object()

    monkeypatch.setattr(mms_pipeline, "_flash_attention_available", lambda: True)

    model = mms_pipeline._load_ctc_model(
        FakeModelLoader, "facebook/mms-1b-all", "cuda", FakeTorch
    )

    assert model is not None
    assert FakeModelLoader.calls == [
        (
            "facebook/mms-1b-all",
            {"torch_dtype": "bfloat16", "attn_implementation": "flash_attention_2"},
        ),
        ("facebook/mms-1b-all", {"torch_dtype": "bfloat16"}),
    ]
    assert "default attention" in caplog.text


@pytest.mark.parametrize(
    ("exception_type", "error_message"),
    [
        (ImportError, "FlashAttention2 is incompatible with this device"),
        (ValueError, "This model does not support Flash Attention 2 yet."),
    ],
)
def test_load_ctc_model_retries_for_transformers_flash_attention_errors(
    monkeypatch, exception_type, error_message
):
    class FakeCUDA:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported(**_kwargs):
            return True

    class FakeTorch:
        bfloat16 = "bfloat16"
        cuda = FakeCUDA()

    class FakeModelLoader:
        calls = 0

        @staticmethod
        def from_pretrained(_model_id, **kwargs):
            FakeModelLoader.calls += 1
            if kwargs.get("attn_implementation") == "flash_attention_2":
                raise exception_type(error_message)
            return object()

    monkeypatch.setattr(mms_pipeline, "_flash_attention_available", lambda: True)

    assert (
        mms_pipeline._load_ctc_model(
            FakeModelLoader, "facebook/mms-1b-all", "cuda", FakeTorch
        )
        is not None
    )
    assert FakeModelLoader.calls == 2


@pytest.mark.parametrize(
    ("flash_available", "error_message"),
    [
        (True, "corrupt flash-cache"),
        (False, "corrupt bfloat16-experiment"),
    ],
)
def test_load_ctc_model_reraises_unrelated_model_load_errors(
    monkeypatch, flash_available, error_message
):
    class FakeCUDA:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported(**_kwargs):
            return True

    class FakeTorch:
        bfloat16 = "bfloat16"
        cuda = FakeCUDA()

    class FakeModelLoader:
        calls = 0

        @staticmethod
        def from_pretrained(_model_id, **_kwargs):
            FakeModelLoader.calls += 1
            raise RuntimeError(error_message)

    monkeypatch.setattr(
        mms_pipeline, "_flash_attention_available", lambda: flash_available
    )

    with pytest.raises(RuntimeError, match=error_message):
        mms_pipeline._load_ctc_model(
            FakeModelLoader, "facebook/mms-1b-all", "cuda", FakeTorch
        )
    assert FakeModelLoader.calls == 1


class FakeWaveform:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape

    def mean(self, dim):
        assert dim == 0
        return FakeWaveform(self.values.mean(axis=0))

    def __getitem__(self, index):
        return FakeWaveform(self.values[index])

    def numel(self):
        return self.values.size

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


def test_load_audio_normalizes_peak_after_stereo_conversion_and_resampling(monkeypatch):
    monkeypatch.setattr(mms_pipeline, "_load_torchaudio", lambda: FakeAudioBackend)
    waveform, rate = mms_pipeline.load_audio("audio.wav")
    assert rate == 16_000
    assert waveform.dtype == np.float32
    assert waveform.flags.c_contiguous
    np.testing.assert_allclose(waveform, [0.95, 0.95, 0.95, 0.95])


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ([0.00001, -0.00001], [0.00001, -0.00001]),
        ([0.0001, -0.0001], [0.95, -0.95]),
        ([0.0, 0.0], [0.0, 0.0]),
    ],
)
def test_load_audio_does_not_amplify_near_silence(monkeypatch, samples, expected):
    class QuietAudioBackend:
        @staticmethod
        def load(_path):
            return FakeWaveform([samples]), 16_000

    monkeypatch.setattr(mms_pipeline, "_load_torchaudio", lambda: QuietAudioBackend)

    waveform, _ = mms_pipeline.load_audio("quiet.wav")

    np.testing.assert_allclose(waveform, expected)


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
    np.testing.assert_allclose(waveform, [0.95, 0.95])


def test_load_audio_rejects_empty_soundfile_fallback_before_resampling(monkeypatch):
    class BrokenAudioBackend:
        @staticmethod
        def load(_path):
            raise ImportError("torchcodec unavailable")

        class functional:
            @staticmethod
            def resample(*_args):
                raise AssertionError("empty audio must not be resampled")

    class FakeSoundFile:
        @staticmethod
        def read(_path, always_2d, dtype):
            return np.empty((0, 1), dtype=np.float32), 8_000

    class FakeTorch:
        @staticmethod
        def from_numpy(values):
            return FakeWaveform(values)

    monkeypatch.setattr(mms_pipeline, "_load_torchaudio", lambda: BrokenAudioBackend)
    monkeypatch.setattr(mms_pipeline, "_load_soundfile", lambda: FakeSoundFile)
    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)

    with pytest.raises(ValueError, match="Audio contains no samples"):
        mms_pipeline.load_audio("audio.wav")
