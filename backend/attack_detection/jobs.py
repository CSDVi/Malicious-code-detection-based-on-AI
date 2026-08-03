"""In-process queue shared by file and project scans."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .cancellation import ScanCancelled

ProgressCallback = Callable[[int, int, str], None]
JobWork = Callable[[threading.Event, ProgressCallback], dict[str, Any]]
JobUpdateCallback = Callable[["ScanJob"], None]
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class ScanJob:
    id: str
    mode: str
    username: str
    target_name: str
    target_type: str = "project"
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    processed_files: int = 0
    total_files: int = 0
    stage: str = "等待执行"
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    update_callback: JobUpdateCallback | None = field(default=None, repr=False, compare=False)
    last_progress_notify: float = field(default=0.0, repr=False, compare=False)


class ScanJobQueue:
    def __init__(self) -> None:
        self._jobs: dict[str, ScanJob] = {}
        self._lock = threading.Lock()

    def submit(
        self, mode: str, username: str, target_name: str, work: JobWork,
        on_update: JobUpdateCallback | None = None,
        *, target_type: str = "project",
    ) -> ScanJob:
        job = ScanJob(
            id=uuid.uuid4().hex, mode=mode, username=username,
            target_name=target_name, target_type=target_type,
            update_callback=on_update,
        )
        with self._lock:
            self._jobs[job.id] = job
        self._notify(job)
        thread = threading.Thread(target=self._run, args=(job.id, work), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> ScanJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, username: str, limit: int = 20) -> list[ScanJob]:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.username == username]
            return sorted(jobs, key=lambda item: item.created_at, reverse=True)[:limit]

    def active_ids(self) -> set[str]:
        with self._lock:
            return {
                job_id for job_id, job in self._jobs.items()
                if job.status not in TERMINAL_STATUSES
            }

    def cancel(self, job_id: str, username: str) -> ScanJob | None:
        notify = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.username != username:
                return None
            if job.status not in TERMINAL_STATUSES:
                job.cancel_event.set()
                job.status = "cancelling"
                job.stage = "正在停止"
                notify = True
        if notify:
            self._notify(job)
        return job

    def _run(self, job_id: str, work: JobWork) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.stage = "已停止"
                job.finished_at = time.time()
                cancelled_before_start = True
            else:
                cancelled_before_start = False
                job.status = "running"
                job.stage = "准备文件" if job.target_type == "file" else "准备项目文件"
                job.started_at = time.time()
        self._notify(job)
        if cancelled_before_start:
            return
        try:
            result = work(job.cancel_event, lambda done, total, stage: self._progress(job_id, done, total, stage))
            with self._lock:
                if job.cancel_event.is_set() or result.get("cancelled"):
                    job.result = None
                    job.status = "cancelled"
                    job.stage = "已停止"
                else:
                    job.result = result
                    job.status = "completed"
                    job.stage = "检测完成"
                job.finished_at = time.time()
            self._notify(job)
        except ScanCancelled:
            with self._lock:
                job.result = None
                job.error = None
                job.status = "cancelled"
                job.stage = "已停止"
                job.finished_at = time.time()
            self._notify(job)
        except Exception as exc:  # pragma: no cover - defensive background path
            with self._lock:
                job.error = str(exc)
                job.status = "failed"
                job.stage = "执行失败"
                job.finished_at = time.time()
            self._notify(job)

    def _progress(self, job_id: str, done: int, total: int, stage: str) -> None:
        notify = False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.status == "cancelling":
                return
            next_stage = str(stage)
            now = time.monotonic()
            notify = (
                next_stage != job.stage
                or now - job.last_progress_notify >= 1.0
            )
            job.processed_files = max(0, int(done))
            job.total_files = max(0, int(total))
            job.stage = next_stage
            if notify:
                job.last_progress_notify = now
        if notify:
            self._notify(job)

    @staticmethod
    def _notify(job: ScanJob) -> None:
        if not job.update_callback:
            return
        try:
            job.update_callback(job)
        except Exception as exc:
            # A persistence problem must not turn a completed scan into a failure.
            job.error = (
                f"result persistence failed: {exc}"
                if job.status == "completed"
                else job.error
            )
            return


scan_jobs = ScanJobQueue()
