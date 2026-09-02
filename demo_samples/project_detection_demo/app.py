from flask import Flask, request

from services.command_service import run_report
from services.export_service import build_export_url

app = Flask(__name__)


@app.get("/report")
def report():
    name = request.args.get("name", "demo")
    result = run_report(name)
    return {"result": result, "export": build_export_url(name)}
