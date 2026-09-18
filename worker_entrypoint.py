def run(job_queue, jobs_dir: str, model_id: str, device: str, diarizer_name: str) -> None:
    import worker

    worker.worker_loop(job_queue, jobs_dir, model_id, device, diarizer_name)
