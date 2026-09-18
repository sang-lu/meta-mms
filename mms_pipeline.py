import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np

logger = logging.getLogger(__name__)

_NORMALIZED_PEAK = 0.95
_NORMALIZATION_MIN_PEAK = np.float32(1e-4)


@dataclass
class MMSModels:
    model: object
    processor: object
    asr: object
    current_adapter: str | None = None
    device: str = "cpu"


def _load_torchaudio():
    import torchaudio

    return torchaudio


def _load_soundfile():
    import soundfile

    return soundfile


def _normalize_peak(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < _NORMALIZATION_MIN_PEAK:
        return audio
    return np.asarray(audio * (_NORMALIZED_PEAK / peak), dtype=np.float32)


def _flash_attention_available() -> bool:
    return find_spec("flash_attn") is not None


def _native_bfloat16_supported(cuda: object) -> bool:
    supported = getattr(cuda, "is_bf16_supported", None)
    if not callable(supported):
        return False
    try:
        return supported(including_emulation=False)
    except TypeError:
        return False


def _model_load_kwargs(device: str, torch: object) -> dict:
    if not device.startswith("cuda"):
        return {}
    if not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; loading the model in FP32.")
        return {}
    if not _native_bfloat16_supported(torch.cuda):
        logger.warning("BF16 is unsupported; loading the model in FP32.")
        return {}

    kwargs = {"torch_dtype": torch.bfloat16}
    if _flash_attention_available():
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        logger.warning("Flash Attention 2 is unavailable; using default attention with BF16.")
    return kwargs


def _bfloat16_error(error: Exception) -> bool:
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "bfloat16 is unsupported",
            "bfloat16 is not supported",
            "bf16 is unsupported",
            "bf16 is not supported",
            "does not support bfloat16",
            "does not support bf16",
            "not implemented for bfloat16",
            "not implemented for bf16",
        )
    )


def _unsupported_attention_keyword_error(error: Exception) -> bool:
    message = str(error).casefold()
    return isinstance(error, TypeError) and (
        "unexpected keyword argument" in message and "attn_implementation" in message
    )


def _flash_attention_error(error: Exception) -> bool:
    message = str(error).casefold()
    if _unsupported_attention_keyword_error(error):
        return True
    if isinstance(error, ImportError):
        return any(
            marker in message
            for marker in ("flash attention", "flash_attn", "flashattention2")
        )
    return isinstance(error, ValueError) and any(
        marker in message
        for marker in (
            "does not support flash attention 2",
            "does not support flash_attention_2",
        )
    )


def _load_ctc_model(model_class: object, model_id: str, device: str, torch: object):
    kwargs = _model_load_kwargs(device, torch)
    while True:
        try:
            return model_class.from_pretrained(model_id, **kwargs)
        except Exception as error:
            if (
                kwargs.get("attn_implementation") == "flash_attention_2"
                and _flash_attention_error(error)
            ):
                if _unsupported_attention_keyword_error(error):
                    kwargs.pop("attn_implementation")
                    fallback = "default attention"
                else:
                    kwargs["attn_implementation"] = "eager"
                    fallback = "eager attention"
                logger.warning(
                    "Flash Attention 2 is unavailable (%s); continuing with %s using BF16.",
                    error,
                    fallback,
                )
                continue
            if "torch_dtype" in kwargs and _bfloat16_error(error):
                kwargs.pop("torch_dtype")
                kwargs.pop("attn_implementation", None)
                logger.warning("BF16 loading failed (%s); continuing in FP32.", error)
                continue
            raise


def load_models(model_id: str, device: str) -> MMSModels:
    import torch
    from transformers import AutoModelForCTC, AutoProcessor, pipeline

    processor = AutoProcessor.from_pretrained(model_id)
    model = _load_ctc_model(AutoModelForCTC, model_id, device, torch).to(device)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if device.startswith("cuda") else -1,
    )
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return MMSModels(model=model, processor=processor, asr=asr, device=device)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    torchaudio = _load_torchaudio()
    try:
        waveform, sample_rate = torchaudio.load(path)
    except (ImportError, OSError) as error:
        logger.warning("Torchaudio decoder unavailable (%s); using SoundFile fallback", error)
        import torch

        soundfile = _load_soundfile()
        audio, sample_rate = soundfile.read(path, always_2d=True, dtype="float32")
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).T)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform[0]
    if waveform.numel() == 0:
        raise ValueError("Audio contains no samples")
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    audio = np.asarray(waveform.cpu().numpy(), dtype=np.float32).reshape(-1)
    audio = _normalize_peak(audio)
    return np.ascontiguousarray(audio), 16_000


def transcribe(
    models: MMSModels,
    waveform: np.ndarray,
    adapter_code: str,
    batch_size: int,
) -> tuple[str, list[dict]]:
    if adapter_code != models.current_adapter:
        models.processor.tokenizer.set_target_lang(adapter_code)
        models.model.load_adapter(adapter_code)
        models.current_adapter = adapter_code
    output = models.asr(
        waveform,
        return_timestamps="word",
        chunk_length_s=30,
        stride_length_s=5,
        batch_size=batch_size,
    )
    return output.get("text", "").strip(), output.get("chunks", [])


def separate_vocals(audio_path: str, output_dir: str, device: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            "--two-stems=vocals",
            audio_path,
            "-o",
            output_dir,
            "--device",
            device,
        ],
        check=False,
    )
    if result.returncode != 0:
        logger.warning(
            "Source splitting failed, using original audio file. "
            "Use --no-stem argument to disable it."
        )
        return audio_path
    return os.path.join(
        output_dir,
        "htdemucs",
        os.path.splitext(os.path.basename(audio_path))[0],
        "vocals.wav",
    )
