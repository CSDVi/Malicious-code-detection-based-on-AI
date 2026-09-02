"""Evaluate cross-file chain recall and GATv2 Top-1 attribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from attack_detection.cross_file_evaluation import (  # noqa: E402
    evaluate_cross_file_challenge,
    load_challenge_manifest,
)


DEFAULT_MANIFEST = (
    BACKEND_DIR / "attack_detection" / "challenge_manifests"
    / "real_world_cross_file_challenge.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure complete-chain recall and GATv2 most-suspicious-component "
            "Top-1 hit rate on the existing independent challenge set."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-gatv2",
        action="store_true",
        help="Measure the static complete-chain path only.",
    )
    args = parser.parse_args()

    report = evaluate_cross_file_challenge(
        load_challenge_manifest(args.manifest),
        workspace_root=args.workspace_root,
        run_gat=not args.skip_gatv2,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
