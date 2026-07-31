"""Runtime inference for strict-gated CodeT5+ task classifiers."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from attack_detection.task_policy import task_enabled


TASK_DECISIONS = {
    "malicious_intent": "malicious",
    "vulnerability_risk": "vulnerable",
}
_PERSISTENT_BUNDLES: dict[str, dict[str, Any]] = {}


def infer_requests(
    model_dir: str | Path,
    requests: list[dict[str, str]],
    *,
    keep_loaded: bool = False,
) -> list[dict[str, Any]]:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, T5Config

    from attack_detection.training.artifact_contracts import validate_codet5p_manifest
    from attack_detection.training.codet5p_model import CodeT5PClassifier

    root = Path(model_dir).resolve()
    registry_path = root / "codet5p_registry.json"
    if not registry_path.is_file():
        return [_unavailable("CodeT5+ 220M registry is missing") for _ in requests]
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [_unavailable(f"CodeT5+ 220M registry cannot be read: {exc}") for _ in requests]

    routes = registry.get("active_routes") or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        default_cpu_threads = max(1, min(12, os.cpu_count() or 6))
        torch.set_num_threads(max(
            1,
            int(os.environ.get(
                "XIEZHI_CODET5_CPU_THREADS",
                str(default_cpu_threads),
            )),
        ))
    bundles: dict[str, dict[str, Any]] = (
        _PERSISTENT_BUNDLES if keep_loaded else {}
    )

    def load_bundle(version: str) -> dict[str, Any]:
        if version in bundles:
            return bundles[version]
        if keep_loaded and bundles:
            # Keep one 220M route resident. Evict it only when a project needs a
            # different language route, avoiding unbounded RAM/VRAM growth.
            bundles.clear()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        artifact = root / "codet5p_artifacts" / version
        manifest_path = artifact / "codet5p_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"manifest is missing for CodeT5+ 220M version {version}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        errors = validate_codet5p_manifest(manifest, artifact)
        if errors:
            raise RuntimeError("; ".join(errors))
        tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True)
        config = T5Config.from_pretrained(artifact, local_files_only=True)
        model = CodeT5PClassifier.from_config(config, dropout=float((manifest.get("config") or {}).get("dropout", 0.15)))
        state = load_file(str(artifact / "codet5p_classifier.safetensors"), device="cpu")
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        bundles[version] = {
            "artifact": artifact,
            "manifest": manifest,
            "tokenizer": tokenizer,
            "model": model,
        }
        return bundles[version]

    def infer_one(
        request_index: int,
        request: dict[str, str],
        precomputed_probabilities: dict[tuple[int, str], float],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        content = str(request.get("content") or "")
        language = str(request.get("language") or "").lower()
        task_outputs: dict[str, dict[str, Any]] = {}
        errors = []
        for task, decision in TASK_DECISIONS.items():
            if not task_enabled(task):
                continue
            task_routes = routes.get(task) if isinstance(routes.get(task), dict) else {}
            version = str(task_routes.get(language) or task_routes.get("all") or "")
            if not version:
                continue
            try:
                bundle = load_bundle(version)
                manifest = bundle["manifest"]
                supported = {str(value).lower() for value in manifest.get("supported_languages") or []}
                if language not in supported and "all" not in supported:
                    continue
                probability_key = (request_index, version)
                if probability_key in precomputed_probabilities:
                    probability = precomputed_probabilities[probability_key]
                else:
                    probability = _predict_probability(
                        bundle["model"],
                        bundle["tokenizer"],
                        content,
                        manifest,
                        device,
                    )
                task_outputs[task] = {
                    "decision": decision,
                    "probability": probability,
                    "threshold": float(manifest["threshold"]),
                    "model_version": version,
                }
            except Exception as exc:
                errors.append(f"{task}: {exc}")
        if not task_outputs:
            reason = "; ".join(errors) if errors else f"no active CodeT5+ 220M route for language {language or 'unknown'}"
            result = _unavailable(reason)
            result["duration_ms"] = int((time.perf_counter() - started) * 1000)
            return result

        positives = [
            (task, value)
            for task, value in task_outputs.items()
            if value["probability"] >= value["threshold"]
        ]
        if positives:
            primary_task, primary = max(
                positives,
                key=lambda item: item[1]["probability"] - item[1]["threshold"],
            )
            decision = primary["decision"]
        else:
            primary_task, primary = max(task_outputs.items(), key=lambda item: item[1]["probability"])
            decision = "benign"
        return {
            "status": "completed",
            "decision": decision,
            "probability": primary["probability"],
            "threshold": primary["threshold"],
            "model_version": ", ".join(sorted({value["model_version"] for value in task_outputs.values()})),
            "primary_task": primary_task,
            "task_probabilities": {
                task: value["probability"] for task, value in task_outputs.items()
            },
            "task_thresholds": {
                task: value["threshold"] for task, value in task_outputs.items()
            },
            "task_versions": {
                task: value["model_version"] for task, value in task_outputs.items()
            },
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }

    results: list[dict[str, Any] | None] = [None] * len(requests)
    for grouped in _group_requests_by_route(routes, requests):
        language = str(grouped[0][1].get("language") or "").lower()
        for task in TASK_DECISIONS:
            task_routes = routes.get(task) if isinstance(routes.get(task), dict) else {}
            version = str(task_routes.get(language) or task_routes.get("all") or "")
            if version and task_enabled(task):
                load_bundle(version)
        scheduled = sorted(
            grouped,
            key=lambda pair: len(str(pair[1].get("content") or "")),
            reverse=True,
        )
        precomputed_probabilities: dict[tuple[int, str], float] = {}
        if device.type == "cuda":
            versions = []
            for task in TASK_DECISIONS:
                if not task_enabled(task):
                    continue
                task_routes = routes.get(task) if isinstance(routes.get(task), dict) else {}
                version = str(task_routes.get(language) or task_routes.get("all") or "")
                if version and version not in versions:
                    versions.append(version)
            for version in versions:
                bundle = load_bundle(version)
                manifest = bundle["manifest"]
                supported = {
                    str(value).lower()
                    for value in manifest.get("supported_languages") or []
                }
                if language not in supported and "all" not in supported:
                    continue
                indexed_contents = [
                    (index, str(request.get("content") or ""))
                    for index, request in scheduled
                ]
                try:
                    probabilities = _predict_probabilities_cuda(
                        bundle["model"],
                        bundle["tokenizer"],
                        indexed_contents,
                        manifest,
                        device,
                    )
                except (RuntimeError, ValueError):
                    # Keep the established one-file path as a correctness fallback
                    # for unusual tokenizers or GPUs with insufficient free VRAM.
                    continue
                precomputed_probabilities.update({
                    (index, version): probability
                    for index, probability in probabilities.items()
                })
            group_outputs = [
                infer_one(index, request, precomputed_probabilities)
                for index, request in scheduled
            ]
        else:
            with ThreadPoolExecutor(
                max_workers=min(2, len(grouped)),
                thread_name_prefix="codet5-infer",
            ) as executor:
                group_outputs = list(executor.map(
                    lambda pair: infer_one(pair[0], pair[1], precomputed_probabilities),
                    scheduled,
                ))
        for (index, _request), output in zip(scheduled, group_outputs):
            results[index] = output

        # Active language routes normally use separate 220M artifacts. Releasing
        # the completed group prevents mixed-language projects from retaining all
        # of those weights at once and forcing the host into memory paging.
        if not keep_loaded:
            bundles.clear()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if any(result is None for result in results):
        raise RuntimeError("CodeT5+ 220M grouped inference did not produce every requested result")
    return [result for result in results if result is not None]


def _group_requests_by_route(
    routes: dict[str, Any],
    requests: list[dict[str, str]],
) -> list[list[tuple[int, dict[str, str]]]]:
    """Keep each language/model route contiguous while preserving output indices."""

    grouped: dict[tuple[str, tuple[str, ...]], list[tuple[int, dict[str, str]]]] = {}
    for index, request in enumerate(requests):
        language = str(request.get("language") or "").lower()
        versions = []
        for task in TASK_DECISIONS:
            if not task_enabled(task):
                continue
            task_routes = routes.get(task) if isinstance(routes.get(task), dict) else {}
            versions.append(str(task_routes.get(language) or task_routes.get("all") or ""))
        grouped.setdefault((language, tuple(versions)), []).append((index, request))
    return list(grouped.values())


def _predict_probability(model: Any, tokenizer: Any, content: str, manifest: dict[str, Any], device: Any) -> float:
    import torch

    config = manifest.get("config") or {}
    maximum_characters = int(config.get("maximum_code_characters") or 120_000)
    max_length = int(config.get("max_length") or 512)
    stride = min(int(config.get("stride") or 128), max_length // 2)
    maximum_windows = max(1, int(config.get("maximum_eval_windows") or 24))
    encoded = tokenizer(
        content[:maximum_characters],
        add_special_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation=True,
        padding=True,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][:maximum_windows].to(device)
    attention_mask = encoded["attention_mask"][:maximum_windows].to(device)
    with torch.inference_mode():
        logits, _pooled = model(input_ids=input_ids, attention_mask=attention_mask)
    maximum_logit = float(logits.max().detach().cpu())
    return _calibrate_logit(maximum_logit, manifest)


def _predict_probabilities_cuda(
    model: Any,
    tokenizer: Any,
    indexed_contents: list[tuple[int, str]],
    manifest: dict[str, Any],
    device: Any,
) -> dict[int, float]:
    """Run windows from multiple files in CUDA batches without changing coverage."""

    import torch

    config = manifest.get("config") or {}
    maximum_characters = int(config.get("maximum_code_characters") or 120_000)
    max_length = int(config.get("max_length") or 512)
    stride = min(int(config.get("stride") or 128), max_length // 2)
    maximum_windows = max(1, int(config.get("maximum_eval_windows") or 24))
    batch_size = max(
        1,
        int(os.environ.get("XIEZHI_CODET5_GPU_WINDOW_BATCH_SIZE", "64")),
    )
    pad_token_id = int(tokenizer.pad_token_id or 0)
    windows: list[tuple[int, Any, Any]] = []
    for request_index, content in indexed_contents:
        encoded = tokenizer(
            content[:maximum_characters],
            add_special_tokens=True,
            max_length=max_length,
            stride=stride,
            truncation=True,
            padding=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][:maximum_windows]
        attention_mask = encoded["attention_mask"][:maximum_windows]
        windows.extend(
            (request_index, input_ids[row], attention_mask[row])
            for row in range(input_ids.shape[0])
        )

    maximum_logits = {
        request_index: -math.inf
        for request_index, _content in indexed_contents
    }
    with torch.inference_mode():
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset:offset + batch_size]
            width = max(int(input_ids.shape[0]) for _index, input_ids, _mask in batch)
            batch_input_ids = torch.full(
                (len(batch), width),
                pad_token_id,
                dtype=batch[0][1].dtype,
            )
            batch_attention_mask = torch.zeros(
                (len(batch), width),
                dtype=batch[0][2].dtype,
            )
            for row, (_index, input_ids, attention_mask) in enumerate(batch):
                row_width = int(input_ids.shape[0])
                batch_input_ids[row, :row_width] = input_ids
                batch_attention_mask[row, :row_width] = attention_mask
            logits, _pooled = model(
                input_ids=batch_input_ids.to(device),
                attention_mask=batch_attention_mask.to(device),
            )
            for (request_index, _input_ids, _attention_mask), logit in zip(
                batch,
                logits.detach().cpu().tolist(),
            ):
                maximum_logits[request_index] = max(
                    maximum_logits[request_index],
                    float(logit),
                )
    return {
        request_index: _calibrate_logit(maximum_logit, manifest)
        for request_index, maximum_logit in maximum_logits.items()
    }


def _calibrate_logit(maximum_logit: float, manifest: dict[str, Any]) -> float:
    temperature = max(1e-6, float(manifest.get("temperature") or 1.0))
    calibrated = maximum_logit / temperature
    if calibrated >= 0:
        return 1.0 / (1.0 + math.exp(-calibrated))
    exp_value = math.exp(calibrated)
    return exp_value / (1.0 + exp_value)


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "decision": "unknown",
        "probability": None,
        "threshold": None,
        "model_version": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()
    if args.server:
        _serve(args.model_dir)
        return
    payload = json.loads(sys.stdin.read() or "{}")
    requests = payload.get("requests")
    if not isinstance(requests, list):
        requests = [{
            "content": str(payload.get("content") or ""),
            "language": str(payload.get("language") or ""),
        }]
        print(json.dumps(infer_requests(args.model_dir, requests)[0], ensure_ascii=False))
        return
    print(json.dumps({"results": infer_requests(args.model_dir, requests)}, ensure_ascii=False))


def _serve(model_dir: str) -> None:
    """Serve file-backed requests while retaining the active model in memory."""

    for line in sys.stdin:
        try:
            command = json.loads(line)
            request_id = str(command["id"])
            input_path = Path(str(command["input_path"]))
            output_path = Path(str(command["output_path"]))
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            requests = payload.get("requests")
            if not isinstance(requests, list):
                raise ValueError("persistent CodeT5 request has no request list")
            output = {
                "results": infer_requests(
                    model_dir,
                    requests,
                    keep_loaded=True,
                )
            }
            output_path.write_text(
                json.dumps(output, ensure_ascii=False),
                encoding="utf-8",
            )
            print(
                "XIEZHI_RESULT "
                + json.dumps({"id": request_id, "status": "completed"}),
                flush=True,
            )
        except Exception as exc:
            print(
                "XIEZHI_RESULT "
                + json.dumps({
                    "id": str(command.get("id") or "") if "command" in locals() else "",
                    "status": "failed",
                    "error": str(exc),
                }),
                flush=True,
            )


if __name__ == "__main__":
    main()
