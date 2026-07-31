"""ByteCNN-TCN dataset export entry point."""

from __future__ import annotations

import argparse
import json

from attack_detection.training.mamba_dataset import export_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ByteCNN-TCN multi-task dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-code-bytes", type=int, default=8_192)
    args = parser.parse_args()
    report = export_dataset(
        args.dataset, args.output_dir, max_code_bytes=max(256, args.max_code_bytes),
    )
    report["consumer"] = "ByteCNN-TCN"
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
