"""Single-worker model-training queue with SQLite-backed task history."""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from .database import save_training_job
from .trainer import train_model

ProgressCallback = Callable[[float, str], None]
Trainer = Callable[[str | Path, ProgressCallback | None], dict[str, object]]
CompletionCallback = Callable[[], None]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class TrainingJob:
    id: str
    username: str
    engine_name: str
    dataset_name: str
    dataset_path: str
    base_version: str = ""
    model_family: str = "legacy_svm"
    training_task: str = ""
    target_language: str = "all"
    status: str = "queued"
    progress: float = 0.0
    log: str = "等待训练"
    error: str | None = None
    model_version: str | None = None
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    on_complete: CompletionCallback | None = None
    trainer: Trainer | None = None

    def persistence_payload(self) -> dict[str, object]:
        payload = self.__dict__.copy()
        payload.pop("dataset_path", None)
        payload.pop("on_complete", None)
        payload.pop("trainer", None)
        return payload


class TrainingJobQueue:
    """Run expensive training work serially so web requests remain responsive."""

    def __init__(self, trainer: Trainer = train_model, max_pending: int = 3) -> None:
        self._trainer = trainer
        self._max_pending = max(1, int(max_pending))
        self._queue: queue.Queue[TrainingJob] = queue.Queue(maxsize=self._max_pending)
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._active_count = 0

    def submit(
        self,
        username: str,
        dataset_name: str,
        dataset_path: str | Path,
        engine_name: str = "TF-IDF / SVM",
        base_version: str = "",
        model_family: str = "legacy_svm",
        training_task: str = "",
        target_language: str = "all",
        trainer: Trainer | None = None,
        on_complete: CompletionCallback | None = None,
    ) -> TrainingJob:
        with self._lock:
            if self._active_count >= self._max_pending:
                raise RuntimeError("训练队列已满，请等待现有任务完成。")
            job = TrainingJob(
                id=uuid.uuid4().hex,
                username=username,
                engine_name=engine_name,
                dataset_name=dataset_name,
                dataset_path=str(Path(dataset_path).resolve()),
                base_version=base_version,
                model_family=model_family,
                training_task=training_task,
                target_language=target_language,
                created_at=_now(),
                on_complete=on_complete,
                trainer=trainer,
            )
            save_training_job(job.persistence_payload())
            self._queue.put_nowait(job)
            self._active_count += 1
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._work_loop,
                    name="xiezhi-model-training",
                    daemon=True,
                )
                self._worker.start()
            return job

    def has_active_jobs(self) -> bool:
        with self._lock:
            return self._active_count > 0

    def _work_loop(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                return
            try:
                self._run(job)
            finally:
                with self._lock:
                    self._active_count = max(0, self._active_count - 1)
                self._queue.task_done()

    def _run(self, job: TrainingJob) -> None:
        job.status = "running"
        job.started_at = _now()
        job.progress = 0.02
        job.log = "正在读取训练数据"
        save_training_job(job.persistence_payload())

        def update(progress: float, stage: str) -> None:
            job.progress = min(0.99, max(job.progress, float(progress)))
            job.log = str(stage)
            save_training_job(job.persistence_payload())

        try:
            metrics = (job.trainer or self._trainer)(job.dataset_path, update)
            job.model_version = str(metrics.get("model_version") or "") or None
            job.status = "completed"
            job.progress = 1.0
            job.log = "训练完成，质量门禁通过后已自动发布" if metrics.get("published") else "训练完成，未通过质量门禁"
            job.finished_at = _now()
            if metrics.get("published") and job.on_complete:
                try:
                    job.on_complete()
                except Exception as exc:  # Training succeeded even if the live cache cannot refresh.
                    job.log += f"；运行时模型重新加载失败：{str(exc)[:300]}"
            save_training_job(job.persistence_payload())
        except Exception as exc:  # pragma: no cover - defensive background path
            job.status = "failed"
            job.error = str(exc)[:1000]
            job.log = "训练失败"
            job.finished_at = _now()
            save_training_job(job.persistence_payload())


training_jobs = TrainingJobQueue()
