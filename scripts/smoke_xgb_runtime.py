"""Small runtime smoke test for the active routed XGBoost release."""

from __future__ import annotations

import json
import sys

ROOT = __file__
sys.path.insert(0, "backend")

from attack_detection.engines.xgb_engine import XGBoostEngine


def main() -> None:
    engine = XGBoostEngine()
    samples = [
        (
            "safe",
            "public class Safe { int add(int a,int b){return a+b;} }",
        ),
        (
            "sql_sink",
            (
                "public class Demo { void run(String x) { "
                "java.sql.Statement s = conn.createStatement(); "
                's.executeQuery("SELECT * FROM users WHERE id=" + x); } }'
            ),
        ),
    ]
    for name, code in samples:
        print(name, json.dumps(engine.scan(code, "java"), ensure_ascii=False))


if __name__ == "__main__":
    main()
