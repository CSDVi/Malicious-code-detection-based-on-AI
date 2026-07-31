"""SQLite persistence for Xiezhi CodeGuard."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .languages import display_language
from .rules import RULES

DB_PATH = str(Path(__file__).resolve().parents[1] / "data" / "attack_detection.db")
RULE_BY_ID = {rule.rule_id: rule for rule in RULES}


class User(UserMixin):
    def __init__(self, row: sqlite3.Row | dict[str, Any]):
        self.id = str(row["id"])
        self.username = row["username"]
        self.role = row.get("role", "analyst") if isinstance(row, dict) else row["role"]


@contextmanager
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=1.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_database() -> None:
    with get_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'analyst',
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS detection_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            filename TEXT NOT NULL,
            language TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            md5 TEXT,
            sha1 TEXT,
            sha256 TEXT,
            project_job_id TEXT,
            risk_score INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            categories TEXT NOT NULL,
            rule_matches TEXT NOT NULL,
            ml_label TEXT NOT NULL,
            ml_probability REAL NOT NULL,
            created_at TEXT NOT NULL
        )""")
        _relax_detection_probability_null(conn)
        conn.execute("""CREATE TABLE IF NOT EXISTS scan_jobs (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            mode TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_name TEXT NOT NULL,
            status TEXT NOT NULL,
            risk_score INTEGER,
            final_decision TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        )""")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_jobs_user_created "
            "ON scan_jobs(username, created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_jobs_user_status_created "
            "ON scan_jobs(username, status, created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_detection_user_created "
            "ON detection_records(username, created_at DESC)"
        )
        conn.execute("""CREATE TABLE IF NOT EXISTS scan_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            record_id INTEGER,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            language TEXT NOT NULL,
            size_bytes INTEGER
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS engine_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id INTEGER,
            job_id TEXT,
            engine_name TEXT NOT NULL,
            status TEXT NOT NULL,
            probability REAL,
            threshold REAL,
            model_version TEXT,
            duration_ms INTEGER,
            reason TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id INTEGER,
            job_id TEXT,
            source TEXT NOT NULL,
            rule_id TEXT,
            category TEXT,
            cwe TEXT,
            behavior TEXT,
            severity INTEGER,
            line INTEGER,
            snippet TEXT,
            description TEXT,
            repair_advice TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL UNIQUE,
            engine_name TEXT NOT NULL,
            metrics TEXT,
            deployment_status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS training_jobs (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            engine_name TEXT NOT NULL,
            status TEXT NOT NULL,
            progress REAL,
            log TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT
        )""")
        _ensure_column(conn, "training_jobs", "dataset_name", "TEXT")
        _ensure_column(conn, "training_jobs", "model_version", "TEXT")
        _ensure_column(conn, "training_jobs", "base_version", "TEXT")
        _ensure_column(conn, "training_jobs", "model_family", "TEXT")
        _ensure_column(conn, "training_jobs", "training_task", "TEXT")
        _ensure_column(conn, "training_jobs", "target_language", "TEXT")
        _ensure_column(conn, "training_jobs", "started_at", "TEXT")
        conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id INTEGER,
            username TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL
        )""")
        _ensure_column(conn, "detection_records", "final_decision", "TEXT")
        _ensure_column(conn, "detection_records", "malicious_probability", "REAL")
        _ensure_column(conn, "detection_records", "vulnerability_label", "TEXT")
        _ensure_column(conn, "detection_records", "vulnerability_probability", "REAL")
        _ensure_column(conn, "detection_records", "engine_votes", "TEXT")
        _ensure_column(conn, "detection_records", "engines", "TEXT")
        _ensure_column(conn, "detection_records", "selected_mode", "TEXT")
        _ensure_column(conn, "detection_records", "effective_mode", "TEXT")
        _ensure_column(conn, "detection_records", "escalation_reason", "TEXT")
        _ensure_column(conn, "detection_records", "attack_techniques", "TEXT")
        _ensure_column(conn, "detection_records", "model_version", "TEXT")
        _ensure_column(conn, "detection_records", "training_samples", "INTEGER")
        _ensure_column(conn, "detection_records", "md5", "TEXT")
        _ensure_column(conn, "detection_records", "sha1", "TEXT")
        _ensure_column(conn, "detection_records", "sha256", "TEXT")
        _ensure_column(conn, "detection_records", "project_job_id", "TEXT")
        _ensure_column(conn, "scan_jobs", "result_json", "TEXT")
        _ensure_column(conn, "scan_jobs", "processed_files", "INTEGER")
        _ensure_column(conn, "scan_jobs", "total_files", "INTEGER")
        _ensure_column(conn, "scan_jobs", "stage", "TEXT")
        default_password = os.environ.get("XIEZHI_ADMIN_PASSWORD", "admin123")
        if not conn.execute("SELECT id FROM users WHERE username=?", ("admin",)).fetchone():
            conn.execute(
                "INSERT INTO users(username,password_hash,role,created_at) VALUES(?,?,?,?)",
                ("admin", generate_password_hash(default_password), "admin", datetime.now().isoformat(timespec="seconds")),
            )


def authenticate(username: str, password: str) -> User | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username.strip(),)).fetchone()
    if row and check_password_hash(row["password_hash"], password):
        return User(row)
    return None


def create_user(username: str, password: str) -> bool:
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users(username,password_hash,role,created_at) VALUES(?,?,?,?)",
                (username.strip(), generate_password_hash(password), "analyst", datetime.now().isoformat(timespec="seconds")),
            )
        return True
    except sqlite3.Error:
        return False


def get_user_by_id(user_id: int) -> User | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return User(row) if row else None


def save_detection(username: str, result: dict[str, Any]) -> int:
    with get_connection() as conn:
        record_id = _insert_detection(conn, username, result)
        _save_engine_runs(conn, record_id, None, result.get("engines", []))
        _save_findings(conn, record_id, None, result.get("findings", result.get("matches", [])))
        return record_id


def list_records(username: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    with get_connection() as conn:
        if username:
            rows = conn.execute(
                "SELECT * FROM detection_records WHERE username=? AND COALESCE(project_job_id,'')='' ORDER BY created_at DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM detection_records WHERE COALESCE(project_job_id,'')='' ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
    return [_record(row) for row in rows]


def list_record_summaries(
    username: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Read history rows without loading saved engines and evidence JSON."""

    columns = (
        "id,username,filename,language,file_hash,risk_score,risk_level,"
        "categories,final_decision,effective_mode,created_at"
    )
    with get_connection() as conn:
        if username:
            rows = conn.execute(
                f"SELECT {columns} FROM detection_records "
                "WHERE username=? AND COALESCE(project_job_id,'')='' "
                "ORDER BY created_at DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {columns} FROM detection_records "
                "WHERE COALESCE(project_job_id,'')='' "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    output = []
    for row in rows:
        item = dict(row)
        try:
            item["categories"] = json.loads(
                item.get("categories") or "[]"
            )
        except json.JSONDecodeError:
            item["categories"] = []
        output.append(item)
    return output


def get_record(record_id: int, username: str | None = None) -> dict[str, Any] | None:
    with get_connection() as conn:
        if username:
            row = conn.execute("SELECT * FROM detection_records WHERE id=? AND username=?", (record_id, username)).fetchone()
        else:
            row = conn.execute("SELECT * FROM detection_records WHERE id=?", (record_id,)).fetchone()
    return _record(row) if row else None


def get_statistics(username: str | None = None) -> dict[str, Any]:
    query = (
        "SELECT filename,risk_score,risk_level,language,categories "
        "FROM detection_records WHERE COALESCE(project_job_id,'')=''"
    )
    params: tuple[Any, ...] = ()
    if username:
        query += " AND username=?"
        params = (username,)
    query += " ORDER BY created_at DESC LIMIT 1000"
    with get_connection() as conn:
        records = [dict(row) for row in conn.execute(query, params).fetchall()]
    level_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    for record in records:
        level_counts[record["risk_level"]] = level_counts.get(record["risk_level"], 0) + 1
        concrete_language = display_language(
            str(record.get("language") or "unknown"),
            str(record.get("filename") or ""),
        )
        language_counts[concrete_language] = language_counts.get(concrete_language, 0) + 1
        try:
            categories = json.loads(record.get("categories") or "[]")
        except json.JSONDecodeError:
            categories = []
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "total": len(records),
        "level_counts": level_counts,
        "category_counts": category_counts,
        "language_counts": language_counts,
        "average_score": round(sum(r["risk_score"] for r in records) / len(records), 1) if records else 0,
    }


def list_training_jobs(username: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    with get_connection() as conn:
        if username:
            rows = conn.execute(
                "SELECT * FROM training_jobs WHERE username=? ORDER BY created_at DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(row) for row in rows]


def save_training_job(payload: dict[str, Any]) -> None:
    """Persist a model-training task and its latest execution state."""
    with get_connection() as conn:
        for column, column_type in (
            ("dataset_name", "TEXT"), ("model_version", "TEXT"),
            ("base_version", "TEXT"), ("model_family", "TEXT"),
            ("training_task", "TEXT"), ("target_language", "TEXT"), ("started_at", "TEXT"),
        ):
            _ensure_column(conn, "training_jobs", column, column_type)
        conn.execute(
            """INSERT INTO training_jobs(
                id,username,engine_name,dataset_name,base_version,model_family,
                training_task,target_language,status,progress,log,error,model_version,
                created_at,started_at,finished_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, progress=excluded.progress,
                log=excluded.log, error=excluded.error,
                model_version=excluded.model_version,
                started_at=excluded.started_at, finished_at=excluded.finished_at""",
            (
                payload["id"], payload["username"], payload["engine_name"],
                payload.get("dataset_name"), payload.get("base_version"), payload.get("model_family"),
                payload.get("training_task"), payload.get("target_language"),
                payload["status"], payload.get("progress"), payload.get("log"),
                payload.get("error"), payload.get("model_version"),
                payload["created_at"], payload.get("started_at"), payload.get("finished_at"),
            ),
        )


def save_scan_job(payload: dict[str, Any]) -> None:
    """Persist project task state so it survives page refreshes and restarts."""
    with get_connection() as conn:
        for column, column_type in (("result_json", "TEXT"), ("processed_files", "INTEGER"),
                                    ("total_files", "INTEGER"), ("stage", "TEXT")):
            _ensure_column(conn, "scan_jobs", column, column_type)
        conn.execute(
            """INSERT INTO scan_jobs(
                id,username,mode,target_type,target_name,status,risk_score,final_decision,error,
                created_at,started_at,finished_at,result_json,processed_files,total_files,stage
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, risk_score=excluded.risk_score,
                final_decision=excluded.final_decision, error=excluded.error,
                started_at=excluded.started_at, finished_at=excluded.finished_at,
                result_json=excluded.result_json, processed_files=excluded.processed_files,
                total_files=excluded.total_files, stage=excluded.stage""",
            (
                payload["id"], payload["username"], payload["mode"], "project", payload["target_name"],
                payload["status"], payload.get("risk_score"), payload.get("final_decision"), payload.get("error"),
                _iso_time(payload.get("created_at")), _iso_time(payload.get("started_at")),
                _iso_time(payload.get("finished_at")),
                json.dumps(payload.get("result"), ensure_ascii=False) if payload.get("result") is not None else None,
                payload.get("processed_files", 0), payload.get("total_files", 0), payload.get("stage"),
            ),
        )


def list_scan_jobs(
    username: str,
    limit: int = 20,
    *,
    include_result: bool = True,
) -> list[dict[str, Any]]:
    columns = (
        "*"
        if include_result
        else (
            "id,username,mode,target_type,target_name,status,risk_score,"
            "final_decision,error,created_at,started_at,finished_at,"
            "processed_files,total_files,stage"
        )
    )
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT {columns} FROM scan_jobs WHERE username=? "
            "ORDER BY created_at DESC LIMIT ?",
            (username, limit),
        ).fetchall()
    return [_scan_job(row, include_result=include_result) for row in rows]


def get_scan_job(job_id: str, username: str | None = None) -> dict[str, Any] | None:
    with get_connection() as conn:
        query = "SELECT * FROM scan_jobs WHERE id=?"
        params: list[Any] = [job_id]
        if username:
            query += " AND username=?"
            params.append(username)
        row = conn.execute(query, params).fetchone()
    return _scan_job(row) if row else None


def cancel_persisted_scan_job(job_id: str, username: str) -> dict[str, Any] | None:
    """Finish a persisted-only task that no longer has an in-process worker."""

    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM scan_jobs WHERE id=? AND username=?",
            (job_id, username),
        ).fetchone()
        if row is None:
            return None
        if row["status"] not in {"completed", "failed", "cancelled"}:
            conn.execute(
                """UPDATE scan_jobs
                SET status='cancelled', stage='已停止',
                    finished_at=COALESCE(finished_at, ?)
                WHERE id=? AND username=?""",
                (datetime.now().isoformat(timespec="seconds"), job_id, username),
            )
            row = conn.execute(
                "SELECT * FROM scan_jobs WHERE id=? AND username=?",
                (job_id, username),
            ).fetchone()
    return _scan_job(row)


def reconcile_interrupted_scan_jobs(active_job_ids: set[str] | None = None) -> int:
    """Mark jobs orphaned by a backend restart as stopped."""

    active = active_job_ids or set()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id FROM scan_jobs WHERE status IN ('queued','running','cancelling')"
        ).fetchall()
        orphaned = [str(row["id"]) for row in rows if str(row["id"]) not in active]
        if not orphaned:
            return 0
        finished_at = datetime.now().isoformat(timespec="seconds")
        conn.executemany(
            """UPDATE scan_jobs
            SET status='cancelled', stage='后端重启，任务已停止',
                finished_at=COALESCE(finished_at, ?)
            WHERE id=?""",
            [(finished_at, job_id) for job_id in orphaned],
        )
    return len(orphaned)


def save_project_results(job_id: str, username: str, result: dict[str, Any]) -> None:
    """Save each project file once and link it to the durable project task."""
    file_results = result.get("file_results") or []
    if not file_results:
        return
    with get_connection() as conn:
        if conn.execute("SELECT 1 FROM scan_files WHERE job_id=? LIMIT 1", (job_id,)).fetchone():
            return
        for item in file_results:
            record_id = _insert_detection(conn, username, item, project_job_id=job_id)
            _save_engine_runs(conn, record_id, job_id, item.get("engines", []))
            _save_findings(conn, record_id, job_id, item.get("findings", item.get("matches", [])))
            conn.execute(
                "INSERT INTO scan_files(job_id,record_id,filename,file_hash,language,size_bytes) VALUES(?,?,?,?,?,?)",
                (job_id, record_id, item.get("filename", ""), item.get("file_hash", ""),
                 item.get("language", "unknown"), None),
            )


def _insert_detection(
    conn: sqlite3.Connection, username: str, result: dict[str, Any], project_job_id: str | None = None,
) -> int:
    for column, column_type in (
        ("final_decision", "TEXT"), ("malicious_probability", "REAL"),
        ("vulnerability_label", "TEXT"), ("vulnerability_probability", "REAL"),
        ("engine_votes", "TEXT"), ("engines", "TEXT"), ("selected_mode", "TEXT"),
        ("effective_mode", "TEXT"), ("escalation_reason", "TEXT"),
        ("attack_techniques", "TEXT"), ("model_version", "TEXT"),
        ("training_samples", "INTEGER"), ("md5", "TEXT"), ("sha1", "TEXT"), ("sha256", "TEXT"),
        ("project_job_id", "TEXT"), ("decision_policy", "TEXT"),
    ):
        _ensure_column(conn, "detection_records", column, column_type)
    hashes = dict(result.get("hashes") or {})
    hashes.setdefault("sha256", result.get("file_hash"))
    cursor = conn.execute(
        """INSERT INTO detection_records(
            username,filename,language,file_hash,md5,sha1,sha256,project_job_id,risk_score,risk_level,categories,rule_matches,
            ml_label,ml_probability,final_decision,malicious_probability,vulnerability_label,
            vulnerability_probability,engine_votes,engines,selected_mode,effective_mode,escalation_reason,
            attack_techniques,model_version,training_samples,decision_policy,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            username, result["filename"], result["language"], result["file_hash"], hashes.get("md5"),
            hashes.get("sha1"), hashes.get("sha256"), project_job_id, result["risk_score"], result["risk_level"],
            json.dumps(result.get("categories", []), ensure_ascii=False),
            json.dumps(result.get("matches", []), ensure_ascii=False), result.get("ml", {}).get("label", "unknown"),
            result.get("ml", {}).get("probability"), result.get("final_decision"),
            result.get("malicious_intent", {}).get("probability"), result.get("vulnerability_risk", {}).get("label"),
            result.get("vulnerability_risk", {}).get("probability"), json.dumps(result.get("engine_votes", {}), ensure_ascii=False),
            json.dumps(result.get("engines", []), ensure_ascii=False), result.get("selected_mode"), result.get("effective_mode"),
            result.get("escalation_reason"), json.dumps(result.get("attack_techniques", []), ensure_ascii=False),
            str(result.get("model_version") or ""), result.get("training_samples"),
            json.dumps({
                "decision_authority": result.get("decision_authority"),
                "decision_basis": result.get("decision_basis"),
                "ai_decision": result.get("ai_decision"),
                "ai_participated": bool(result.get("ai_participated")),
                "ai_model_count": int(result.get("ai_model_count") or 0),
                "ai_decisive_model_count": int(
                    result.get("ai_decisive_model_count") or 0
                ),
                "ai_model_names": list(result.get("ai_model_names") or []),
                "ai_decisive_model_names": list(
                    result.get("ai_decisive_model_names") or []
                ),
                "ai_model_states": list(
                    result.get("ai_model_states") or []
                ),
                "ai_conflict": bool(result.get("ai_conflict")),
                "ai_uncertain": bool(result.get("ai_uncertain")),
                "rule_fallback_used": bool(
                    result.get("rule_fallback_used")
                ),
                "rule_fallback_reason": result.get(
                    "rule_fallback_reason"
                ),
                "rule_disagrees_with_ai": bool(
                    result.get("rule_disagrees_with_ai")
                ),
            }, ensure_ascii=False),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    return int(cursor.lastrowid)


def _record(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["categories"] = json.loads(data["categories"])
    data["rule_matches"] = json.loads(data["rule_matches"])
    for match in data["rule_matches"]:
        rule = RULE_BY_ID.get(str(match.get("rule_id") or ""))
        if rule:
            match.setdefault("description", rule.description)
            match.setdefault("repair_advice", rule.repair_advice)
            match.setdefault("category", rule.category)
    data["engine_votes"] = json.loads(data.get("engine_votes") or "{}")
    data["engines"] = json.loads(data.get("engines") or "[]")
    data["attack_techniques"] = json.loads(data.get("attack_techniques") or "[]")
    try:
        decision_policy = json.loads(
            data.get("decision_policy") or "{}"
        )
    except json.JSONDecodeError:
        decision_policy = {}
    if isinstance(decision_policy, dict):
        data.update(decision_policy)
    data.setdefault("final_decision", data.get("ml_label", "unknown"))
    data.setdefault("malicious_probability", data.get("ml_probability"))
    data.setdefault("vulnerability_label", "unknown")
    data.setdefault("vulnerability_probability", None)
    engine_findings = [finding for engine in data["engines"] for finding in engine.get("findings", [])]
    if (
        not data.get("decision_authority")
        and data.get("final_decision") in {"malicious", "vulnerable"}
        and not data["rule_matches"]
        and not engine_findings
    ):
        # Legacy records created before the evidence gate must not continue to
        # present a model-only vote as a confirmed finding.
        data["final_decision"] = "unknown"
        data["legacy_decision_note"] = "历史记录仅有模型投票，未保存可定位代码证据"
    data["hashes"] = {
        "md5": data.get("md5"), "sha1": data.get("sha1"),
        "sha256": data.get("sha256") or data.get("file_hash"),
    }
    return data


def _scan_job(
    row: sqlite3.Row,
    *,
    include_result: bool = True,
) -> dict[str, Any]:
    data = dict(row)
    if include_result:
        raw_result = data.get("result_json")
        try:
            data["result"] = json.loads(raw_result) if raw_result else None
        except json.JSONDecodeError:
            data["result"] = None
    else:
        data["result"] = None
    data.setdefault("processed_files", 0)
    data.setdefault("total_files", 0)
    data.setdefault("stage", data.get("status", ""))
    return data


def _iso_time(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value)).isoformat(timespec="seconds")
    return str(value)


def _save_engine_runs(conn: sqlite3.Connection, record_id: int | None, job_id: str | None, engines: list[dict[str, Any]]) -> None:
    for engine in engines:
        conn.execute(
            """INSERT INTO engine_runs(
                record_id,job_id,engine_name,status,probability,threshold,model_version,
                duration_ms,reason,metadata,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record_id,
                job_id,
                engine.get("name"),
                engine.get("status"),
                engine.get("probability"),
                engine.get("threshold"),
                engine.get("model_version"),
                engine.get("duration_ms"),
                engine.get("reason") or engine.get("error"),
                json.dumps(engine.get("metadata", {}), ensure_ascii=False),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def _save_findings(conn: sqlite3.Connection, record_id: int | None, job_id: str | None, findings: list[dict[str, Any]]) -> None:
    for finding in findings:
        conn.execute(
            """INSERT INTO findings(
                record_id,job_id,source,rule_id,category,cwe,behavior,severity,line,
                snippet,description,repair_advice
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record_id,
                job_id,
                finding.get("source") or "rule_engine",
                finding.get("rule_id"),
                finding.get("category"),
                finding.get("cwe"),
                finding.get("behavior") or finding.get("risk_type"),
                finding.get("severity"),
                finding.get("line"),
                finding.get("snippet"),
                finding.get("description"),
                finding.get("repair_advice"),
            ),
        )


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def _relax_detection_probability_null(conn: sqlite3.Connection) -> None:
    columns = list(conn.execute("PRAGMA table_info(detection_records)"))
    ml_col = next((row for row in columns if row["name"] == "ml_probability"), None)
    if not ml_col or not int(ml_col["notnull"]):
        return
    existing = {row["name"] for row in columns}
    conn.execute("ALTER TABLE detection_records RENAME TO detection_records_old")
    conn.execute("""CREATE TABLE detection_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        filename TEXT NOT NULL,
        language TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        risk_score INTEGER NOT NULL,
        risk_level TEXT NOT NULL,
        categories TEXT NOT NULL,
        rule_matches TEXT NOT NULL,
        ml_label TEXT NOT NULL,
        ml_probability REAL,
        final_decision TEXT,
        malicious_probability REAL,
        vulnerability_label TEXT,
        vulnerability_probability REAL,
        engine_votes TEXT,
        engines TEXT,
        selected_mode TEXT,
        effective_mode TEXT,
        escalation_reason TEXT,
        attack_techniques TEXT,
        model_version TEXT,
        training_samples INTEGER,
        created_at TEXT NOT NULL
    )""")
    target = [
        "id", "username", "filename", "language", "file_hash", "risk_score", "risk_level",
        "categories", "rule_matches", "ml_label", "ml_probability", "final_decision",
        "malicious_probability", "vulnerability_label", "vulnerability_probability",
        "engine_votes", "engines", "selected_mode", "effective_mode", "escalation_reason",
        "attack_techniques", "model_version", "training_samples", "created_at",
    ]
    shared = [column for column in target if column in existing]
    joined = ",".join(shared)
    conn.execute(f"INSERT INTO detection_records({joined}) SELECT {joined} FROM detection_records_old")
    conn.execute("DROP TABLE detection_records_old")
