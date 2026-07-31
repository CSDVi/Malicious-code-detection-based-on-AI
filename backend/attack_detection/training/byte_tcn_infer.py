"""Single-request ByteCNN-TCN inference used by the Flask CPU adapter."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from attack_detection.task_policy import task_enabled


def infer(content: str, language: str, model_dir: str | Path) -> dict[str, Any]:
    import torch
    root = Path(model_dir)
    manifest = json.loads((root / "bytetcn_manifest.json").read_text(encoding="utf-8"))
    return _infer_loaded(content, language, root, manifest, torch, {})


def infer_many(requests: list[dict[str, Any]], model_dir: str | Path) -> list[dict[str, Any]]:
    """Run many files in one interpreter so model weights are loaded only once."""

    import torch

    root = Path(model_dir)
    manifest = json.loads((root / "bytetcn_manifest.json").read_text(encoding="utf-8"))
    model_cache: dict[tuple[str, str], Any] = {}
    output = []
    for request in requests:
        start = time.perf_counter()
        result = _infer_loaded(
            str(request.get("content") or ""), str(request.get("language") or "unknown"),
            root, manifest, torch, model_cache,
        )
        result["duration_ms"] = int((time.perf_counter() - start) * 1000)
        output.append(result)
    return output


def _infer_loaded(
    content: str,
    language: str,
    root: Path,
    manifest: dict[str, Any],
    torch: Any,
    model_cache: dict[tuple[str, str], Any],
) -> dict[str, Any]:
    from attack_detection.training.byte_tcn_model import from_config

    task_support = manifest.get("task_language_support", {})
    supported_tasks = {
        task for task, languages in task_support.items()
        if task_enabled(task) and language in languages
    }
    if not supported_tasks:
        return {
            "status": "unavailable",
            "reason": f"ByteCNN-TCN has no validated task for language: {language}",
            "model_version": manifest.get("model_version"),
        }
    torch.set_num_threads(1)
    default_weights = str((manifest.get("files") or ["bytetcn_multitask.pt"])[0])
    task_outputs = {}
    task_lines = {}
    settings_by_task = {}
    for task in supported_tasks:
        settings = _task_settings(manifest, task, language)
        settings_by_task[task] = settings
        config = settings.get("config") or manifest["config"]
        weights_name = str(settings.get("file") or default_weights)
        cache_key = (weights_name, json.dumps(config, sort_keys=True))
        if cache_key not in model_cache:
            model = from_config(config)
            state = torch.load(root / weights_name, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            model_cache[cache_key] = model
        task_outputs[task], byte_lines = _forward_task(
            model_cache[cache_key], content, task, config, torch,
        )
        task_lines[task] = byte_lines
    probabilities = {}
    for task in ("malicious_intent",):
        if task in supported_tasks:
            temperature = float(
                settings_by_task[task].get("temperature", manifest["temperatures"][task])
            )
            probabilities[task] = _sigmoid(float(task_outputs[task][task][0]) / max(temperature, 1e-4))
        else:
            probabilities[task] = None
    thresholds = {
        task: float(
            settings_by_task.get(task, {}).get("threshold", manifest["thresholds"][task])
        )
        for task in ("malicious_intent",)
    }
    decisions = {
        task: probability is not None and probability >= float(thresholds[task])
        for task, probability in probabilities.items()
    }
    primary_task = "malicious_intent"
    decision = "malicious" if decisions.get(primary_task) else "benign"
    behavior_labels = _labels(
        task_outputs["malicious_intent"]["behavior_labels"][0],
        settings_by_task["malicious_intent"].get(
            "behavior_vocabulary", manifest.get("behavior_vocabulary", [])
        ),
        float(settings_by_task["malicious_intent"].get(
            "auxiliary_thresholds", manifest.get("auxiliary_thresholds", {})
        ).get("behavior_labels", 0.5)), torch,
    ) if "malicious_intent" in supported_tasks else []
    cwe_labels = []
    line_scores = []
    auxiliary_thresholds = settings_by_task[primary_task].get(
        "auxiliary_thresholds", manifest.get("auxiliary_thresholds", {})
    )
    line_scores = _line_scores(
        task_outputs[primary_task]["line_localization"][0], task_lines[primary_task],
        float(auxiliary_thresholds.get("line_localization", 0.5)), torch,
    )
    return {
        "status": "completed",
        "decision": decision,
        "primary_task": primary_task,
        "probability": probabilities[primary_task],
        "threshold": float(thresholds[primary_task]),
        "task_probabilities": probabilities,
        "task_thresholds": thresholds,
        "behavior_labels": behavior_labels,
        "cwe_labels": cwe_labels,
        "line_scores": line_scores,
        "model_version": manifest["model_version"],
    }


def _task_settings(
    manifest: dict[str, Any], task: str, language: str,
) -> dict[str, Any]:
    """Resolve a routed artifact while remaining compatible with task-only manifests."""

    task_settings = (manifest.get("task_models") or {}).get(task, {})
    language_settings = task_settings.get("by_language")
    if isinstance(language_settings, dict):
        selected = language_settings.get(language)
        if isinstance(selected, dict):
            return selected
    return task_settings


def _encode(content: str, max_length: int, torch: Any) -> tuple[Any, Any, list[int]]:
    raw = content.encode("utf-8", errors="replace")[: max_length - 2]
    ids = [1] + [byte + 4 for byte in raw] + [2]
    lines, line = [0], 1
    for byte in raw:
        lines.append(line)
        if byte == 10:
            line += 1
    lines.append(0)
    input_ids = torch.tensor([ids], dtype=torch.long)
    return input_ids, torch.ones_like(input_ids, dtype=torch.float32), lines


def _forward_task(
    model: Any, content: str, task: str, config: dict[str, Any], torch: Any,
) -> tuple[dict[str, Any], list[int]]:
    """Run a routed task, honoring the whole-file window policy saved by training."""

    if not config.get("windowed_inference"):
        input_ids, attention_mask, byte_lines = _encode(
            content, int(config["max_length"]), torch,
        )
        with torch.inference_mode():
            return model(input_ids, attention_mask), byte_lines

    window_bytes = int(config.get("window_bytes", int(config["max_length"]) - 2))
    stride = max(1, int(config.get("window_stride", window_bytes)))
    maximum_file_bytes = max(
        window_bytes, int(config.get("maximum_file_bytes", window_bytes)),
    )
    raw = content.encode("utf-8", errors="replace")[:maximum_file_bytes]
    starts = list(range(0, max(1, len(raw) - window_bytes + 1), stride))
    final_start = max(0, len(raw) - window_bytes)
    if final_start not in starts:
        starts.append(final_start)

    outputs = []
    for start in starts:
        window = raw[start:start + window_bytes].decode("utf-8", errors="ignore")
        input_ids, attention_mask, _ = _encode(window, window_bytes + 2, torch)
        with torch.inference_mode():
            outputs.append(model(input_ids, attention_mask))

    # The training evaluator classifies a file by its most suspicious window.
    # Preserve the same aggregation in production so offline metrics are honest.
    aggregated = dict(outputs[0])
    aggregated[task] = torch.stack([value[task] for value in outputs]).amax(dim=0)
    return aggregated, []


def _labels(logits: Any, vocabulary: list[str], threshold: float, torch: Any) -> list[dict[str, Any]]:
    probabilities = torch.sigmoid(logits).tolist()
    return [
        {"label": label, "score": float(probabilities[index])}
        for index, label in enumerate(vocabulary)
        if index < len(probabilities) and probabilities[index] >= threshold
    ]


def _line_scores(logits: Any, lines: list[int], threshold: float, torch: Any) -> list[dict[str, Any]]:
    probabilities = torch.sigmoid(logits).tolist()
    by_line: dict[int, float] = {}
    for line, probability in zip(lines, probabilities):
        if line:
            by_line[line] = max(by_line.get(line, 0.0), float(probability))
    return [
        {"line": line, "score": score}
        for line, score in sorted(by_line.items()) if score >= threshold
    ][:20]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ByteCNN-TCN inference request")
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    payload = json.loads(sys.stdin.read())
    if isinstance(payload.get("requests"), list):
        value = {"results": infer_many(payload["requests"], args.model_dir)}
    else:
        value = infer(str(payload.get("content") or ""), str(payload.get("language") or "unknown"), args.model_dir)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
