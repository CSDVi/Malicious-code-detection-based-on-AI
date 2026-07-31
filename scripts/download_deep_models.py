"""Download curated code-model backbones without pulling duplicate framework weights.

These checkpoints are inputs to Xiezhi training and candidate validation. Except for
the explicitly marked bootstrap checkpoint, they are not vulnerability detectors
until task heads are fine-tuned and pass the project's per-language release gates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "backend" / "models" / "pretrained"
ALLOW_PATTERNS = [
    "*.json",
    "*.md",
    "*.model",
    "*.safetensors",
    "*.txt",
    "LICENSE*",
    "merges.txt",
    "pytorch_model.bin",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
]

MODELS = {
    "codeberta-small": {
        "repo_id": "huggingface/CodeBERTa-small-v1",
        "profiles": ["quick"],
        "languages": ["go", "java", "javascript", "php", "python", "ruby"],
        "role": "84M multilingual encoder backbone for Xiezhi malicious/vulnerability heads",
        "runtime_ready": False,
    },
    "multilingual-cwe-bootstrap": {
        "repo_id": "ayshajavd/graphcodebert-vuln-classifier",
        "profiles": ["quick", "bootstrap"],
        "languages": ["python", "javascript", "java", "c", "cpp", "php", "go"],
        "role": "third-party multi-label CWE candidate used only for pipeline validation",
        "runtime_ready": False,
        "warning": "Self-reported model with weak rare-CWE precision and lower expected PHP/Go performance; do not publish without local evaluation.",
    },
    "graphcodebert-base": {
        "repo_id": "microsoft/graphcodebert-base",
        "profiles": ["standard"],
        "languages": ["go", "java", "javascript", "php", "python", "ruby"],
        "role": "125M data-flow-aware backbone for standard-mode dual-task fine-tuning",
        "runtime_ready": False,
    },
    "codet5p-220m": {
        "repo_id": "Salesforce/codet5p-220m",
        "profiles": ["deep"],
        "languages": ["c", "cpp", "csharp", "go", "java", "javascript", "php", "python", "ruby"],
        "role": "220M encoder-decoder backbone for deep-mode code understanding",
        "runtime_ready": False,
    },
    "qwen2.5-coder-0.5b": {
        "repo_id": "Qwen/Qwen2.5-Coder-0.5B",
        "profiles": ["deep-wide"],
        "languages": ["multilingual-92"],
        "role": "optional 0.49B broad-language long-context candidate for Shell/TypeScript and uncommon languages",
        "runtime_ready": False,
        "warning": "CPU-heavy causal model; use only after classification fine-tuning and INT8/ONNX benchmarking.",
    },
}


def selected_models(profile: str) -> list[tuple[str, dict[str, object]]]:
    if profile == "all":
        return list(MODELS.items())
    return [
        (name, model)
        for name, model in MODELS.items()
        if profile in model["profiles"]
    ]


def download(profile: str, output_root: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "缺少 huggingface_hub；先执行：\n"
            r"D:\software\anaconda\envs\drone\python.exe -m pip install -r backend\requirements-transformer.txt"
        ) from exc

    chosen = selected_models(profile)
    if not chosen:
        raise SystemExit(f"配置 {profile!r} 没有对应模型")
    output_root.mkdir(parents=True, exist_ok=True)
    for name, metadata in chosen:
        target = output_root / name
        print(f"[download] {metadata['repo_id']} -> {target}", flush=True)
        snapshot_download(
            repo_id=str(metadata["repo_id"]),
            local_dir=target,
            allow_patterns=ALLOW_PATTERNS,
        )
        manifest = {
            "schema_version": 1,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "local_name": name,
            **metadata,
        }
        (target / "xiezhi-pretrained.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"[done] downloaded {len(chosen)} checkpoint(s) into {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download curated Xiezhi deep-model candidates")
    parser.add_argument(
        "--profile",
        choices=["bootstrap", "quick", "standard", "deep", "deep-wide", "all"],
        default="quick",
        help="quick downloads the light encoder and bootstrap candidate; all is roughly several GB",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list", action="store_true", help="show selections without downloading")
    args = parser.parse_args()
    chosen = selected_models(args.profile)
    if args.list:
        print(json.dumps({name: model for name, model in chosen}, ensure_ascii=False, indent=2))
        return
    download(args.profile, args.output.resolve())


if __name__ == "__main__":
    main()
