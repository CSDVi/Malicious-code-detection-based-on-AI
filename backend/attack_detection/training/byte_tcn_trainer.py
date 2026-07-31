"""Train, calibrate, and evaluate ByteCNN-TCN on CPU or CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("ByteCNN-TCN training requires PyTorch") from exc
    from attack_detection.training.byte_tcn_model import ByteCNNTCN

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    root = Path(args.dataset_dir)
    records = {split: _load_jsonl(root / f"{split}.jsonl") for split in ("train", "validation", "test")}
    if args.limit:
        records = {split: values[: args.limit] for split, values in records.items()}
    behaviors = _vocabulary(records["train"], "behavior_labels", args.max_behavior_labels)
    cwes = _vocabulary(records["train"], "cwe_labels", args.max_cwe_labels)
    behavior_index = {value: index for index, value in enumerate(behaviors)}
    cwe_index = {value: index for index, value in enumerate(cwes)}

    class CodeDataset(Dataset):
        def __init__(self, values: list[dict[str, Any]]) -> None:
            self.values = values
        def __len__(self) -> int:
            return len(self.values)
        def __getitem__(self, index: int) -> dict[str, Any]:
            return _encode(self.values[index], args.max_length, behavior_index, cwe_index, torch)

    loaders = {
        split: DataLoader(
            CodeDataset(values), batch_size=args.batch_size, shuffle=split == "train",
            collate_fn=lambda batch: _collate(batch, torch), num_workers=0,
        )
        for split, values in records.items()
    }
    model = ByteCNNTCN(
        channels=args.channels, embedding_dim=args.embedding_dim, layers=args.layers,
        kernel_size=args.kernel_size, dropout=args.dropout,
        behavior_labels=len(behaviors), cwe_labels=len(cwes),
    ).to(device)
    positive_weights = {
        "malicious_intent": _positive_weight(records["train"], "malicious_intent", torch, device),
        "vulnerability_risk": _positive_weight(records["train"], "vulnerability_risk", torch, device),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None
    best_score = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["input_ids"], batch["attention_mask"])
            loss = _loss(
                outputs, batch, functional, positive_weights,
                {
                    "malicious_intent": args.malicious_loss_weight,
                    "vulnerability_risk": args.vulnerability_loss_weight,
                },
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch["input_ids"].shape[0]
        raw_validation = _collect(model, loaders["validation"], device, torch)
        validation = _evaluate_raw(raw_validation)
        score = (validation["malicious_intent"]["f1"] + validation["vulnerability_risk"]["f1"]) / 2
        row = {"epoch": epoch, "train_loss": total_loss / max(1, len(records["train"])), "validation": validation}
        history.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
        if score > best_score + 1e-6:
            best_score = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise SystemExit("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    raw_validation = _collect(model, loaders["validation"], device, torch)
    calibration = {
        task: _calibrate(raw_validation[task]["logits"], raw_validation[task]["labels"], torch)
        for task in ("malicious_intent", "vulnerability_risk")
    }
    validation = _evaluate_raw(raw_validation, calibration)
    raw_test = _collect(model, loaders["test"], device, torch)
    test = _evaluate_raw(raw_test, calibration)
    test_by_language = _evaluate_by_language(
        model, records["test"], args, behavior_index, cwe_index, calibration, device, torch,
    )
    eligible_languages = {
        task: _eligible_languages(records, task)
        for task in ("malicious_intent", "vulnerability_risk")
    }
    robust_languages = {
        task: _eligible_languages(
            records, task, {"train": 20, "validation": 10, "test": 20},
        )
        for task in ("malicious_intent", "vulnerability_risk")
    }
    task_language_support = {
        task: sorted(
            language for language in languages
            if _passes_deployment_gate(test_by_language[language][task])
        )
        for task, languages in eligible_languages.items()
    }
    supported_languages = sorted({language for languages in task_language_support.values() for language in languages})

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "bytetcn_multitask.pt"
    torch.save(best_state, weights_path)
    dataset_hash = _sha256(root / "manifest.json")
    version = f"bytetcn-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{dataset_hash[:12]}"
    thresholds = {task: float(values["threshold"]) for task, values in calibration.items()}
    temperatures = {task: float(values["temperature"]) for task, values in calibration.items()}
    manifest = {
        "schema_version": 1,
        "model_version": version,
        "architecture": "ByteCNN with residual dilated temporal convolution blocks",
        "dataset_sha256": dataset_hash,
        "supported_languages": supported_languages,
        "task_language_support": task_language_support,
        "language_support_tiers": {
            task: {
                "validated": sorted(set(task_language_support[task]) & set(robust_languages[task])),
                "provisional": sorted(set(task_language_support[task]) - set(robust_languages[task])),
            }
            for task in task_language_support
        },
        "deployment_gate": {
            "minimum_test_f1": 0.5,
            "maximum_false_positive_rate": 0.2,
            "maximum_false_negative_rate": 0.5,
        },
        "output_heads": ["malicious_intent", "vulnerability_risk", "behavior_labels", "cwe_labels", "line_localization"],
        "thresholds": thresholds,
        "temperatures": temperatures,
        "auxiliary_thresholds": {"behavior_labels": 0.5, "cwe_labels": 0.5, "line_localization": 0.5},
        "calibrated": True,
        "calibration_method": "validation temperature scaling for the two binary task heads",
        "validation_metrics": validation,
        "test_metrics": test,
        "test_metrics_by_language": test_by_language,
        "eligible_task_languages": eligible_languages,
        "robust_task_languages": robust_languages,
        "behavior_vocabulary": behaviors,
        "cwe_vocabulary": cwes,
        "config": model.config | {"max_length": args.max_length},
        "training": {
            "device": str(device), "torch_version": torch.__version__, "epochs_completed": len(history),
            "seed": args.seed, "batch_size": args.batch_size, "threads": args.threads,
            "task_loss_weights": {
                "malicious_intent": args.malicious_loss_weight,
                "vulnerability_risk": args.vulnerability_loss_weight,
            },
            "limited_smoke_run": bool(args.limit),
        },
        "files": [weights_path.name],
        "runtime_ready": not bool(args.limit) and bool(supported_languages),
        "runtime_note": "CPU inference uses the configured external PyTorch interpreter.",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output / "bytetcn_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "bytetcn_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _encode(record: dict[str, Any], max_length: int, behavior_index: dict[str, int], cwe_index: dict[str, int], torch: Any) -> dict[str, Any]:
    raw = str(record.get("code") or "").encode("utf-8", errors="replace")[: max_length - 2]
    input_ids = [1] + [byte + 4 for byte in raw] + [2]
    line_numbers, line = [0], 1
    for byte in raw:
        line_numbers.append(line)
        if byte == 10:
            line += 1
    line_numbers.append(0)
    positive_lines = {
        number for item in record.get("line_labels") or []
        for number in range(int(item["start_line"]), int(item["end_line"]) + 1)
    }
    behavior = [0.0] * len(behavior_index)
    for value in record.get("behavior_labels") or []:
        if value in behavior_index:
            behavior[behavior_index[value]] = 1.0
    cwe = [0.0] * len(cwe_index)
    for value in record.get("cwe_labels") or []:
        if value in cwe_index:
            cwe[cwe_index[value]] = 1.0
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "line_targets": torch.tensor([float(value in positive_lines) for value in line_numbers], dtype=torch.float32),
        "malicious_target": torch.tensor(float(record.get("malicious_intent") or 0), dtype=torch.float32),
        "malicious_mask": torch.tensor(record.get("malicious_intent") is not None, dtype=torch.bool),
        "vulnerability_target": torch.tensor(float(record.get("vulnerability_risk") or 0), dtype=torch.float32),
        "vulnerability_mask": torch.tensor(record.get("vulnerability_risk") is not None, dtype=torch.bool),
        "behavior_target": torch.tensor(behavior, dtype=torch.float32),
        "behavior_mask": torch.tensor(bool(record.get("behavior_mask")), dtype=torch.bool),
        "cwe_target": torch.tensor(cwe, dtype=torch.float32),
        "cwe_mask": torch.tensor(bool(record.get("cwe_mask")), dtype=torch.bool),
        "line_mask": torch.tensor(bool(record.get("line_mask")), dtype=torch.bool),
    }


def _collate(batch: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    maximum = max(item["input_ids"].shape[0] for item in batch)
    output = {
        "input_ids": torch.stack([_pad(item["input_ids"], maximum, 0, torch) for item in batch]),
        "line_targets": torch.stack([_pad(item["line_targets"], maximum, 0.0, torch) for item in batch]),
    }
    output["attention_mask"] = (output["input_ids"] != 0).float()
    for key in batch[0]:
        if key not in {"input_ids", "line_targets"}:
            output[key] = torch.stack([item[key] for item in batch])
    return output


def _pad(value: Any, length: int, fill: float | int, torch: Any) -> Any:
    return value if value.shape[0] == length else torch.cat([
        value, torch.full((length - value.shape[0],), fill, dtype=value.dtype),
    ])


def _loss(
    outputs: dict[str, Any], batch: dict[str, Any], functional: Any,
    positive_weights: dict[str, Any], task_loss_weights: dict[str, float],
) -> Any:
    losses = []
    for output_key, target_key, mask_key in (
        ("malicious_intent", "malicious_target", "malicious_mask"),
        ("vulnerability_risk", "vulnerability_target", "vulnerability_mask"),
    ):
        mask = batch[mask_key]
        if mask.any():
            task = "malicious_intent" if output_key == "malicious_intent" else "vulnerability_risk"
            losses.append(task_loss_weights[task] * functional.binary_cross_entropy_with_logits(
                outputs[output_key][mask], batch[target_key][mask], pos_weight=positive_weights[task],
            ))
    for output_key, target_key, mask_key in (
        ("behavior_labels", "behavior_target", "behavior_mask"),
        ("cwe_labels", "cwe_target", "cwe_mask"),
    ):
        mask = batch[mask_key]
        if mask.any() and outputs[output_key].shape[-1]:
            losses.append(0.35 * functional.binary_cross_entropy_with_logits(outputs[output_key][mask], batch[target_key][mask]))
    line_mask = batch["line_mask"]
    if line_mask.any():
        token_mask = batch["attention_mask"][line_mask].bool()
        losses.append(0.35 * functional.binary_cross_entropy_with_logits(
            outputs["line_localization"][line_mask][token_mask],
            batch["line_targets"][line_mask][token_mask],
            pos_weight=outputs["line_localization"].new_tensor(8.0),
        ))
    return sum(losses)


def _positive_weight(records: list[dict[str, Any]], task: str, torch: Any, device: Any) -> Any:
    labels = [int(record[task]) for record in records if record.get(task) is not None]
    positives = sum(labels)
    negatives = len(labels) - positives
    return torch.tensor(min(12.0, max(1.0, negatives / max(1, positives))), dtype=torch.float32, device=device)


def _collect(model: Any, loader: Any, device: Any, torch: Any) -> dict[str, dict[str, list[Any]]]:
    values = {task: {"logits": [], "labels": []} for task in ("malicious_intent", "vulnerability_risk")}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            for task, target, mask_key in (
                ("malicious_intent", "malicious_target", "malicious_mask"),
                ("vulnerability_risk", "vulnerability_target", "vulnerability_mask"),
            ):
                mask = batch[mask_key]
                values[task]["logits"].extend(outputs[task][mask].cpu().tolist())
                values[task]["labels"].extend(batch[target][mask].int().cpu().tolist())
    return values


def _evaluate_by_language(
    model: Any, records: list[dict[str, Any]], args: argparse.Namespace,
    behavior_index: dict[str, int], cwe_index: dict[str, int],
    calibration: dict[str, Any], device: Any, torch: Any,
) -> dict[str, dict[str, Any]]:
    from torch.utils.data import DataLoader, Dataset

    class LanguageDataset(Dataset):
        def __init__(self, values: list[dict[str, Any]]) -> None:
            self.values = values
        def __len__(self) -> int:
            return len(self.values)
        def __getitem__(self, index: int) -> dict[str, Any]:
            return _encode(self.values[index], args.max_length, behavior_index, cwe_index, torch)

    output = {}
    languages = sorted({str(record.get("language") or "unknown").lower() for record in records})
    for language in languages:
        language_records = [
            record for record in records
            if str(record.get("language") or "unknown").lower() == language
        ]
        loader = DataLoader(
            LanguageDataset(language_records), batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda batch: _collate(batch, torch), num_workers=0,
        )
        output[language] = _evaluate_raw(_collect(model, loader, device, torch), calibration)
    return output


def _eligible_languages(
    records: dict[str, list[dict[str, Any]]], task: str,
    minimums: dict[str, int] | None = None,
) -> list[str]:
    required = minimums or {"train": 20, "validation": 5, "test": 10}
    languages = {
        str(record.get("language") or "unknown").lower()
        for values in records.values() for record in values
    }
    eligible = []
    for language in sorted(languages):
        valid = True
        for split, minimum in required.items():
            labels = [
                int(record[task]) for record in records[split]
                if str(record.get("language") or "unknown").lower() == language
                and record.get(task) is not None
            ]
            if sum(labels) < minimum or len(labels) - sum(labels) < minimum:
                valid = False
                break
        if valid:
            eligible.append(language)
    return eligible


def _calibrate(logits: list[float], labels: list[int], torch: Any) -> dict[str, float]:
    if len(set(labels)) < 2:
        return {"temperature": 1.0, "threshold": 0.5}
    values = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32)
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=60)
    criterion = torch.nn.BCEWithLogitsLoss()
    def closure() -> Any:
        optimizer.zero_grad()
        loss = criterion(values / log_temperature.exp().clamp(0.05, 20.0), targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    temperature = float(log_temperature.exp().clamp(0.05, 20.0).item())
    threshold, _ = _best_metrics(logits, labels, temperature)
    return {"temperature": temperature, "threshold": threshold}


def _evaluate_raw(raw: dict[str, Any], calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    output = {}
    for task, values in raw.items():
        settings = (calibration or {}).get(task, {"temperature": 1.0})
        temperature = float(settings.get("temperature", 1.0))
        if "threshold" in settings:
            threshold = float(settings["threshold"])
            metrics = _metrics(values["logits"], values["labels"], threshold, temperature)
        else:
            threshold, metrics = _best_metrics(values["logits"], values["labels"], temperature)
        output[task] = metrics | {"samples": len(values["labels"]), "temperature": temperature, "threshold": threshold}
    return output


def _best_metrics(logits: list[float], labels: list[int], temperature: float) -> tuple[float, dict[str, float]]:
    best = (0.5, _metrics(logits, labels, 0.5, temperature))
    for step in range(10, 91):
        threshold = step / 100
        candidate = _metrics(logits, labels, threshold, temperature)
        if candidate["f1"] > best[1]["f1"]:
            best = threshold, candidate
    return best


def _metrics(logits: list[float], labels: list[int], threshold: float, temperature: float) -> dict[str, float]:
    probabilities = [_sigmoid(value / max(temperature, 1e-4)) for value in logits]
    predicted = [int(value >= threshold) for value in probabilities]
    tp = sum(a == b == 1 for a, b in zip(labels, predicted)); tn = sum(a == b == 0 for a, b in zip(labels, predicted))
    fp = sum(a == 0 and b == 1 for a, b in zip(labels, predicted)); fn = sum(a == 1 and b == 0 for a, b in zip(labels, predicted))
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    return {
        "accuracy": (tp + tn) / max(1, len(labels)), "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "false_positive_rate": fp / max(1, fp + tn), "false_negative_rate": fn / max(1, fn + tp),
        "brier_score": sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / max(1, len(labels)),
    }


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def _passes_deployment_gate(metrics: dict[str, float]) -> bool:
    return (
        metrics["f1"] >= 0.5
        and metrics["false_positive_rate"] <= 0.2
        and metrics["false_negative_rate"] <= 0.5
    )


def _vocabulary(records: list[dict[str, Any]], key: str, limit: int) -> list[str]:
    counts = Counter(str(value) for record in records for value in record.get(key) or [])
    return [value for value, _ in counts.most_common(limit)]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CPU-friendly ByteCNN-TCN multi-task model")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.0008)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--malicious-loss-weight", type=float, default=1.0)
    parser.add_argument("--vulnerability-loss-weight", type=float, default=1.5)
    parser.add_argument("--max-behavior-labels", type=int, default=64)
    parser.add_argument("--max-cwe-labels", type=int, default=128)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test record limit per split")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
