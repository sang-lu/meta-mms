import logging
import os

import mms_pipeline
from diarization import create_diarizer
from jobstore import JobStore
from language import LanguageSelection
from schema import build_result
from speaker_mapping import map_words_to_speakers

logger = logging.getLogger(__name__)


def _diarize(diarizer, waveform):
    import torch

    return diarizer.diarize(torch.from_numpy(waveform).unsqueeze(0))


def process_one_job(models, diarizer, store: JobStore, job_id: str, *, temp_dir: str) -> None:
    try:
        store.set_status(job_id, {"status": "processing"})
        params = store.params(job_id)
        selection = LanguageSelection(**params["language"])
        audio_path = store.audio_path(job_id)
        os.makedirs(temp_dir, exist_ok=True)

        if params.get("no_stem", True):
            vocal_target = audio_path
        else:
            vocal_target = mms_pipeline.separate_vocals(
                audio_path, temp_dir, getattr(models, "device", "cpu")
            )

        waveform, _sample_rate = mms_pipeline.load_audio(vocal_target)
        _transcript, chunks = mms_pipeline.transcribe(
            models,
            waveform,
            selection.adapter_code,
            params.get("batch_size", 8),
        )
        speaker_spans = _diarize(diarizer, waveform)
        words = map_words_to_speakers(chunks, speaker_spans)
        result = build_result(job_id, selection, int(len(waveform) / 16_000), words)

        store.set_result(job_id, result)
        store.set_status(job_id, {"status": "completed"})
    except Exception as error:
        logger.exception("Job %s failed", job_id)
        store.set_status(job_id, {"status": "failed", "error": str(error)})


def worker_loop(
    job_queue, jobs_dir: str, model_id: str, device: str, diarizer_name: str
) -> None:
    models = mms_pipeline.load_models(model_id, device)
    diarizer = create_diarizer(diarizer_name, device)
    store = JobStore(jobs_dir)
    temp_dir = os.path.join(jobs_dir, f"_worker_temp_{os.getpid()}")

    while True:
        job_id = job_queue.get()
        if job_id is None:
            return
        process_one_job(models, diarizer, store, job_id, temp_dir=temp_dir)
