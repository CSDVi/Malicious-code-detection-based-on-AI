"""Launcher/browser lifecycle tests that do not open a network port."""

from __future__ import annotations

import pytest
from flask import Flask

from web.routes import main_routes


@pytest.fixture()
def launcher_client():
    app = Flask(__name__)
    app.register_blueprint(main_routes.main_bp)
    with main_routes._frontend_state_lock:
        main_routes._frontend_state["sessions"] = {}
        main_routes._frontend_state["ever_connected"] = False
        main_routes._frontend_state["shutdown_requested"] = False
    yield app.test_client()
    with main_routes._frontend_state_lock:
        main_routes._frontend_state["sessions"] = {}
        main_routes._frontend_state["ever_connected"] = False
        main_routes._frontend_state["shutdown_requested"] = False


def test_explicit_page_close_marks_frontend_closed_immediately(launcher_client):
    opened = launcher_client.post(
        "/api/launcher/session/open",
        json={"token": "page-a"},
    )
    assert opened.status_code == 200
    assert launcher_client.get("/api/launcher/status").get_json()["active"] is True

    closed = launcher_client.post(
        "/api/launcher/session/close",
        json={"token": "page-a"},
    )

    assert closed.status_code == 200
    status = launcher_client.get("/api/launcher/status").get_json()
    assert status["active"] is False
    assert status["active_count"] == 0
    assert status["frontend_closed"] is True


def test_closing_one_page_keeps_other_page_active(launcher_client):
    launcher_client.post("/api/launcher/session/open", json={"token": "page-a"})
    launcher_client.post("/api/launcher/session/open", json={"token": "page-b"})

    launcher_client.post("/api/launcher/session/close", json={"token": "page-a"})

    status = launcher_client.get("/api/launcher/status").get_json()
    assert status["active"] is True
    assert status["active_count"] == 1
    assert status["frontend_closed"] is False
