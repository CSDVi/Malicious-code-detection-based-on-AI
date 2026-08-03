import sys
import threading
import time
import json

import pytest

import attack_detection.database as database
from attack_detection.cancellation import ScanCancelled, run_cancellable
from attack_detection.engines.xgb_engine import _line_attributions
from attack_detection.jobs import ScanJobQueue
from attack_detection.project_scanner import _select_deep_candidates


def _wait_for_status(queue: ScanJobQueue, job_id: str, statuses: set[str]) -> str:
    deadline = time.time() + 2
    while time.time() < deadline:
        job = queue.get(job_id)
        if job and job.status in statuses:
            return job.status
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {statuses}")


def test_job_continues_without_page_state_and_requires_explicit_cancel():
    queue = ScanJobQueue()

    def work(cancel_event, progress):
        for index in range(100):
            if cancel_event.is_set():
                return {"cancelled": True}
            progress(index, 100, "扫描文件")
            time.sleep(0.005)
        return {"ok": True}

    job = queue.submit("standard", "alice", "project.zip", work)
    assert _wait_for_status(queue, job.id, {"running"}) == "running"
    assert queue.list("alice")[0].id == job.id
    queue.cancel(job.id, "alice")
    assert _wait_for_status(queue, job.id, {"cancelled"}) == "cancelled"
    assert queue.get(job.id).result is None


def test_job_cannot_be_cancelled_by_another_user():
    queue = ScanJobQueue()
    job = queue.submit("quick", "alice", "project.zip", lambda _event, _progress: {"ok": True})
    assert queue.cancel(job.id, "bob") is None


def test_frequent_progress_updates_do_not_write_persistence_for_every_file():
    queue = ScanJobQueue()
    persisted = []

    def work(_cancel_event, progress):
        for index in range(100):
            progress(index + 1, 100, "快速分析")
        return {"ok": True}

    job = queue.submit(
        "quick", "alice", "project.zip", work,
        on_update=lambda current: persisted.append(
            (current.status, current.processed_files, current.stage)
        ),
    )

    assert _wait_for_status(queue, job.id, {"completed"}) == "completed"
    assert queue.get(job.id).processed_files == 100
    assert len(persisted) < 10
    assert persisted[-1][0] == "completed"


def test_file_scan_job_keeps_target_type_and_percentage_progress():
    queue = ScanJobQueue()

    def work(_cancel_event, progress):
        progress(30, 100, "AI 模型检测中")
        progress(100, 100, "检测完成")
        return {"risk_score": 42, "final_decision": "unknown"}

    job = queue.submit(
        "auto",
        "alice",
        "sample.py",
        work,
        target_type="file",
    )

    assert _wait_for_status(queue, job.id, {"completed"}) == "completed"
    completed = queue.get(job.id)
    assert completed.target_type == "file"
    assert completed.processed_files == 100
    assert completed.total_files == 100


def test_cancellable_subprocess_is_terminated_promptly():
    cancel_event = threading.Event()
    timer = threading.Timer(0.15, cancel_event.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(ScanCancelled):
            run_cancellable(
                [sys.executable, "-c", "import time; time.sleep(20)"],
                input_text="",
                cwd=".",
                timeout=20,
                cancel_event=cancel_event,
                poll_interval=0.05,
            )
    finally:
        timer.cancel()
    assert time.monotonic() - started < 3


def test_cancellable_subprocess_preserves_large_json_stdin():
    payload = json.dumps({
        "requests": [{"content": "x" * 80_000, "language": "php"} for _ in range(8)]
    })

    completed = run_cancellable(
        [
            sys.executable,
            "-c",
            "import json,sys,time; data=json.load(sys.stdin); time.sleep(.2); "
            "print(len(data['requests']), len(data['requests'][0]['content']))",
        ],
        input_text=payload,
        cwd=".",
        timeout=5,
        cancel_event=threading.Event(),
        poll_interval=0.05,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "8 80000"


def test_xgboost_line_attribution_stops_between_masked_predictions():
    cancel_event = threading.Event()
    prediction_count = 0

    def predict(_content):
        nonlocal prediction_count
        prediction_count += 1
        cancel_event.set()
        return 0.4

    with pytest.raises(ScanCancelled):
        _line_attributions(
            "first()\nsecond()\nthird()",
            0.9,
            predict,
            cancel_event=cancel_event,
        )

    assert prediction_count == 1


def test_persisted_scan_jobs_can_be_cancelled_and_reconciled_after_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "scan-jobs.db"))
    database.init_database()

    def save(job_id: str, status: str) -> None:
        database.save_scan_job({
            "id": job_id,
            "username": "alice",
            "mode": "auto",
            "target_name": "project.zip",
            "status": status,
            "created_at": time.time(),
            "started_at": time.time(),
            "finished_at": None,
            "processed_files": 0,
            "total_files": 48,
            "stage": "正在停止" if status == "cancelling" else "快速分析",
        })

    save("persisted-cancel", "cancelling")
    cancelled = database.cancel_persisted_scan_job("persisted-cancel", "alice")
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    assert cancelled["stage"] == "已停止"
    assert cancelled["finished_at"]

    save("orphaned-running", "running")
    save("still-active", "running")
    assert database.reconcile_interrupted_scan_jobs({"still-active"}) == 1
    assert database.get_scan_job("orphaned-running", "alice")["status"] == "cancelled"
    assert database.get_scan_job("orphaned-running", "alice")["stage"] == "后端重启，任务已停止"
    assert database.get_scan_job("still-active", "alice")["status"] == "running"


def test_ai_decision_policy_survives_history_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        database,
        "DB_PATH",
        str(tmp_path / "decision-policy.db"),
    )
    database.init_database()
    result = {
        "filename": "model-only.py",
        "language": "python",
        "file_hash": "a" * 64,
        "hashes": {
            "md5": "b" * 32,
            "sha1": "c" * 40,
            "sha256": "a" * 64,
        },
        "risk_score": 91,
        "risk_level": "critical",
        "categories": [],
        "matches": [],
        "findings": [],
        "ml": {"label": "malicious", "probability": 0.91},
        "malicious_intent": {
            "label": "malicious",
            "probability": 0.91,
        },
        "vulnerability_risk": {
            "label": "disabled",
            "probability": None,
        },
        "final_decision": "malicious",
        "decision_authority": "ai",
        "decision_basis": "ai_model",
        "ai_decision": "malicious",
        "ai_participated": True,
        "ai_model_count": 1,
        "ai_decisive_model_count": 1,
        "ai_model_names": ["xgboost_malicious"],
        "ai_decisive_model_names": ["xgboost_malicious"],
        "ai_model_states": [{
            "name": "xgboost_malicious",
            "decision": "malicious",
            "probability": 0.91,
            "threshold": 0.5,
            "decisive": True,
        }],
        "ai_conflict": False,
        "ai_uncertain": False,
        "rule_fallback_used": False,
        "rule_fallback_reason": None,
        "rule_disagrees_with_ai": False,
        "engine_votes": {},
        "engines": [],
        "selected_mode": "quick",
        "effective_mode": "quick",
        "attack_techniques": [],
        "training_samples": 1,
    }

    record_id = database.save_detection("alice", result)
    saved = database.get_record(record_id, "alice")

    assert saved is not None
    assert saved["final_decision"] == "malicious"
    assert saved["decision_authority"] == "ai"
    assert saved["ai_model_names"] == ["xgboost_malicious"]
    assert saved["rule_fallback_used"] is False


def test_compact_project_job_list_does_not_parse_result_json(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        database,
        "DB_PATH",
        str(tmp_path / "compact-jobs.db"),
    )
    database.init_database()
    database.save_scan_job({
        "id": "compact-job",
        "username": "alice",
        "mode": "quick",
        "target_name": "demo.zip",
        "status": "completed",
        "risk_score": 80,
        "final_decision": "malicious",
        "created_at": "2026-07-31T00:00:00",
        "result": {"file_results": [{"large": "x" * 10000}]},
        "processed_files": 1,
        "total_files": 1,
        "stage": "检测完成",
    })
    monkeypatch.setattr(
        database.json,
        "loads",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("compact list parsed result_json")
        ),
    )

    jobs = database.list_scan_jobs(
        "alice",
        include_result=False,
    )

    assert jobs[0]["id"] == "compact-job"
    assert jobs[0]["result"] is None


def test_scan_job_list_can_filter_file_and_project_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "target-types.db"))
    database.init_database()
    base = {
        "username": "alice",
        "mode": "auto",
        "status": "completed",
        "created_at": "2026-08-01T00:00:00",
    }
    database.save_scan_job({
        **base,
        "id": "file-job",
        "target_type": "file",
        "target_name": "sample.py",
    })
    database.save_scan_job({
        **base,
        "id": "project-job",
        "target_type": "project",
        "target_name": "sample.zip",
    })

    assert [job["id"] for job in database.list_scan_jobs(
        "alice", target_type="file",
    )] == ["file-job"]
    assert [job["id"] for job in database.list_scan_jobs(
        "alice", target_type="project",
    )] == ["project-job"]


def test_project_deep_candidates_are_bounded_and_language_validated():
    records = [
        {"filename": f"src/File{index}.java", "content": "class A {}", "language": "java"}
        for index in range(80)
    ] + [{"filename": "src/app.js", "content": "eval(input)", "language": "javascript"}]
    results = [{
        "risk_score": 100 - (index % 12),
        "engines": [{
            "name": "xgboost_malicious",
            "status": "completed",
            "decision": (
                "malicious" if (index % 12) >= 10 else "benign"
            ),
            "probability": (index % 12) / 20,
            "threshold": 0.5,
        }],
    } for index in range(len(records))]
    selected = _select_deep_candidates(
        records,
        results,
        20,
        supported_languages={"java"},
    )
    assert len(selected) == 20
    assert all(records[index]["language"] == "java" for index in selected)
    assert 71 in selected
