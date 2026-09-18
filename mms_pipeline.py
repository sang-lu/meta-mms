import logging
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


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


def load_models(model_id: str, device: str) -> MMSModels:
    import torch
    from transformers import AutoModelForCTC, AutoProcessor, pipeline

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).to(device)
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
