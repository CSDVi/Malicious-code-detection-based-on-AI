"""Main routes."""

from __future__ import annotations

import json
import time
from threading import RLock

from flask import Blueprint, flash, jsonify, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user

from attack_detection.database import (
    authenticate,
    create_user,
    list_record_summaries,
    list_scan_jobs,
)
from attack_detection.fusion import risk_level as score_risk_level

main_bp = Blueprint("main", __name__)

SESSION_TIMEOUT_SECONDS = 8
_frontend_state_lock = RLock()
_frontend_state: dict[str, object] = {
    "sessions": {},
    "ever_connected": False,
    "shutdown_requested": False,
}


@main_bp.route("/")
@login_required
def dashboard():
    show_opening = bool(session.pop("show_dashboard_opening", False))
    recent_records = _merge_recent_detection_records(
        list_record_summaries(current_user.username, 20),
        list_scan_jobs(
            current_user.username,
            50,
            include_result=False,
        ),
        limit=6,
    )
    return render_template(
        "index.html",
        recent_records=recent_records,
        show_opening=show_opening,
    )


def _merge_recent_detection_records(
    single_records: list[dict[str, object]],
    project_jobs: list[dict[str, object]],
    *,
    limit: int = 6,
) -> list[dict[str, object]]:
    """Build one newest-first dashboard timeline from completed detections."""

    merged: list[dict[str, object]] = []
    for record in single_records:
        merged.append({
            "record_type": "single",
            "record_type_label": "单文件",
            "id": record.get("id"),
            "target_name": record.get("filename") or "未命名文件",
            "risk_level": record.get("risk_level") or "safe",
            "risk_score": record.get("risk_score", 0),
            "mode": record.get("effective_mode") or record.get("mode") or "未记录",
            "created_at": str(record.get("created_at") or ""),
        })

    for job in project_jobs:
        result = job.get("result")
        if job.get("status") != "completed":
            continue
        project_result = result if isinstance(result, dict) else {}
        project_score = int(
            project_result.get("max_score")
            or job.get("risk_score")
            or 0
        )
        merged.append({
            "record_type": "project",
            "record_type_label": "项目",
            "id": job.get("id"),
            "target_name": job.get("target_name") or "未命名项目",
            "risk_level": (
                project_result.get("risk_level")
                or score_risk_level(project_score)
            ),
            "risk_score": project_score,
            "mode": job.get("mode") or "未记录",
            "created_at": str(job.get("finished_at") or job.get("created_at") or ""),
        })

    merged.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return merged[:max(0, limit)]


@main_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        user = authenticate(request.form.get("username", ""), request.form.get("password", ""))
        if user:
            login_user(user)
            session["show_dashboard_opening"] = True
            return redirect(url_for("main.dashboard"))
        flash("用户名或密码错误")
    return render_template("login.html")


@main_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if len(username) < 3 or len(password) < 6:
            flash("用户名至少 3 位，密码至少 6 位")
        elif create_user(username, password):
            flash("注册成功，请登录")
            return redirect(url_for("main.login"))
        else:
            flash("用户名已存在")
    return render_template("register.html")


@main_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("main.login"))


@main_bp.route("/api/launcher/session/open", methods=["POST"])
def launcher_session_open():
    token = _request_token()
    if not token:
        return jsonify({"ok": False, "error": "missing token"}), 400
    _touch_session(token)
    return jsonify({"ok": True, "token": token})


@main_bp.route("/api/launcher/heartbeat", methods=["POST"])
def launcher_heartbeat():
    token = _request_token()
    if token:
        _touch_session(token)
    return jsonify({"ok": True})


@main_bp.route("/api/launcher/session/close", methods=["POST"])
def launcher_session_close():
    token = _request_token()
    if token:
        _close_session(token)
    return jsonify({"ok": True})


@main_bp.route("/api/launcher/shutdown", methods=["POST"])
def launcher_shutdown():
    with _frontend_state_lock:
        _frontend_state["shutdown_requested"] = True
    return jsonify({"ok": True, "shutdown_requested": True})


@main_bp.route("/api/launcher/status")
def launcher_status():
    sessions = _active_sessions()
    with _frontend_state_lock:
        ever_connected = bool(_frontend_state["ever_connected"])
        shutdown_requested = bool(_frontend_state["shutdown_requested"])
    frontend_closed = ever_connected and not bool(sessions)
    return jsonify({
        "active": bool(sessions),
        "active_count": len(sessions),
        "ever_connected": ever_connected,
        "frontend_closed": frontend_closed,
        "shutdown_requested": shutdown_requested,
        "timeout_seconds": SESSION_TIMEOUT_SECONDS,
    })


def _request_token() -> str:
    data = request.get_json(silent=True) or {}
    if not data and request.data:
        try:
            data = json.loads(request.data.decode("utf-8", errors="ignore") or "{}")
        except Exception:
            data = {}
    return str(data.get("token") or request.form.get("token") or request.args.get("token") or "").strip()


def _touch_session(token: str) -> None:
    with _frontend_state_lock:
        sessions = _frontend_state["sessions"]
        if isinstance(sessions, dict):
            sessions[token] = time.time()
        _frontend_state["ever_connected"] = True


def _close_session(token: str) -> None:
    with _frontend_state_lock:
        sessions = _frontend_state["sessions"]
        if isinstance(sessions, dict):
            sessions.pop(token, None)


def _active_sessions() -> dict[str, float]:
    with _frontend_state_lock:
        sessions = _frontend_state["sessions"]
        if not isinstance(sessions, dict):
            return {}
        now = time.time()
        expired = [token for token, seen in sessions.items() if now - float(seen) > SESSION_TIMEOUT_SECONDS]
        for token in expired:
            sessions.pop(token, None)
        return dict(sessions)
