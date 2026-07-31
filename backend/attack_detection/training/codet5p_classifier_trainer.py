"""Fine-tune CodeT5+ 220M for repository-isolated code-risk classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from attack_detection.dataset import CodeSample, is_task_training_eligible, load_dataset


TASKS = {
    "vulnerability_risk": {"positive": "vulnerable", "negative": "benign"},
    "malicious_intent": {"positive": "malicious", "negative": "benign"},
}
QUALITY_GATE = {
    "minimum_precision": 0.90,
    "maximum_false_positive_rate": 0.10,
    "maximum_false_negative_rate": 0.10,
}
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "models" / "pretrained_cache"


class SampleDataset:
    def __init__(self, samples: list[CodeSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CodeSample:
        return self.samples[index]


def train_codet5p(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint: str = "Salesforce/codet5p-220m",
    base_version: str = "codet5p-220m-base",
    base_artifact_dir: str | Path | None = None,
    task: str = "vulnerability_risk",
    target_language: str = "all",
    epochs: int = 5,
    patience: int = 2,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    stride: int = 128,
    maximum_code_characters: int = 32768,
    maximum_train_windows: int = 3,
    maximum_eval_windows: int = 8,
    pairwise_weight: float = 0.2,
    pairwise_margin: float = 1.0,
    seed: int = 20260723,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, T5Config

    from attack_detection.training.codet5p_model import CodeT5PClassifier

    if task not in TASKS:
        raise ValueError(f"unsupported CodeT5+ 220M task: {task}")
    source = Path(dataset_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    positive = TASKS[task]["positive"]
    negative = TASKS[task]["negative"]
    all_samples = load_dataset(source)
    eligible = [
        sample for sample in all_samples
        if is_task_training_eligible(sample, task) and sample.label in {positive, negative}
    ]
    supported_languages = _select_languages(eligible, target_language, positive, negative)
    samples = [sample for sample in eligible if sample.language in supported_languages]
    partitions = {
        split: [sample for sample in samples if sample.split == split]
        for split in ("train", "validation", "test")
    }
    _validate_partitions(partitions, positive, negative)
    _validate_family_isolation(samples)

    resumed_from = ""
    if base_artifact_dir:
        base_root = Path(base_artifact_dir).resolve()
        weights = base_root / "codet5p_classifier.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"base CodeT5+ 220M weights are missing: {weights}")
        tokenizer = AutoTokenizer.from_pretrained(base_root, local_files_only=True, use_fast=True)
        config = T5Config.from_pretrained(base_root, local_files_only=True)
        model = CodeT5PClassifier.from_config(config)
        state = load_file(str(weights), device="cpu")
        base_manifest = _read_json(base_root / "codet5p_manifest.json")
        if str(base_manifest.get("task") or "") != task:
            state = {key: value for key, value in state.items() if not key.startswith("classifier.")}
        model.load_state_dict(state, strict=False)
        resumed_from = str(base_root)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            cache_dir=str(cache),
            use_fast=True,
        )
        model = CodeT5PClassifier.from_pretrained(
            checkpoint,
            cache_dir=str(cache),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_labels = [1 if sample.label == positive else 0 for sample in partitions["train"]]
    positives = sum(train_labels)
    negatives = len(train_labels) - positives
    pos_weight = torch.tensor(
        [max(1.0, negatives / max(1, positives))],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    def collate_train(rows: list[CodeSample]) -> dict[str, Any]:
        return _tokenize_rows(
            tokenizer,
            rows,
            positive,
            max_length=max_length,
            stride=stride,
            maximum_code_characters=maximum_code_characters,
            maximum_windows=maximum_train_windows,
        )

    def collate_eval(rows: list[CodeSample]) -> dict[str, Any]:
        return _tokenize_rows(
            tokenizer,
            rows,
            positive,
            max_length=max_length,
            stride=stride,
            maximum_code_characters=maximum_code_characters,
            maximum_windows=maximum_eval_windows,
        )

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    stale_epochs = 0
    progress_weights = destination / "codet5p_classifier_in_progress.safetensors"
    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        losses: list[float] = []
        pair_losses: list[float] = []
        batches = _pair_aware_batches(partitions["train"], max(2, batch_size), seed + epoch)
        loader = DataLoader(
            SampleDataset(partitions["train"]),
            batch_sampler=batches,
            collate_fn=collate_train,
            num_workers=0,
        )
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(**inputs)
                classification_loss = criterion(logits, labels)
                ranking_loss = _paired_ranking_loss(
                    logits,
                    batch["row_indices"].to(device),
                    batch["rows"],
                    positive,
                    margin=pairwise_margin,
                )
                loss = classification_loss + pairwise_weight * ranking_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            pair_losses.append(float(ranking_loss.detach().cpu()))

        validation_logits = _predict_logits(
            model,
            partitions["validation"],
            collate_eval,
            device,
            batch_size=max(1, batch_size),
            use_amp=use_amp,
        )
        validation_probabilities = _sigmoid_list(validation_logits)
        validation_metrics = _metrics(
            [1 if sample.label == positive else 0 for sample in partitions["validation"]],
            validation_probabilities,
            threshold=0.5,
        )
        epoch_score = (
            validation_metrics["f1"]
            - max(0.0, validation_metrics["false_positive_rate"] - 0.05)
            - max(0.0, validation_metrics["false_negative_rate"] - 0.05)
        )
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            "pairwise_loss": sum(pair_losses) / max(1, len(pair_losses)),
            "validation_at_0_5": validation_metrics,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (destination / "codet5p_history_in_progress.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if epoch_score > best_score + 1e-6:
            best_score = epoch_score
            stale_epochs = 0
            save_file(_cpu_state_dict(model), str(progress_weights))
        else:
            stale_epochs += 1
        if stale_epochs >= max(1, patience):
            break

    if not progress_weights.is_file():
        raise RuntimeError("CodeT5+ 220M training did not produce a checkpoint")
    model.load_state_dict(load_file(str(progress_weights), device="cpu"))
    model.to(device)

    validation_logits = _predict_logits(
        model,
        partitions["validation"],
        collate_eval,
        device,
        batch_size=max(1, batch_size),
        use_amp=use_amp,
    )
    validation_labels = [1 if sample.label == positive else 0 for sample in partitions["validation"]]
    temperature = _fit_temperature(validation_logits, validation_labels)
    validation_probabilities = _sigmoid_list([value / temperature for value in validation_logits])
    threshold = _select_threshold(validation_labels, validation_probabilities)
    validation_metrics = _metrics(validation_labels, validation_probabilities, threshold)

    test_logits = _predict_logits(
        model,
        partitions["test"],
        collate_eval,
        device,
        batch_size=max(1, batch_size),
        use_amp=use_amp,
    )
    test_labels = [1 if sample.label == positive else 0 for sample in partitions["test"]]
    test_probabilities = _sigmoid_list([value / temperature for value in test_logits])
    test_metrics = _metrics(test_labels, test_probabilities, threshold)
    test_by_language = _segment_metrics(
        partitions["test"], test_labels, test_probabilities, threshold, "language",
    )
    test_by_source = _segment_metrics(
        partitions["test"], test_labels, test_probabilities, threshold, "source",
    )
    passed = (
        _passes_gate(validation_metrics)
        and _passes_gate(test_metrics)
        and all(
            _passes_gate(metrics)
            for metrics in test_by_language.values()
            if metrics["positive_samples"] and metrics["negative_samples"]
        )
    )

    weights_name = "codet5p_classifier.safetensors"
    save_file(_cpu_state_dict(model), str(destination / weights_name))
    model.encoder.config.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    history_path = destination / "codet5p_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_sha256 = _sha256(source)
    version = (
        "codet5p-"
        + task.replace("_risk", "").replace("_intent", "")
        + "-"
        + "-".join(supported_languages)
        + "-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + dataset_sha256[:10]
    )
    files = sorted(
        path.name
        for path in destination.iterdir()
        if path.is_file()
        and path.name not in {
            "codet5p_manifest.json",
            "codet5p_classifier_in_progress.safetensors",
            "codet5p_history_in_progress.json",
        }
    )
    manifest = {
        "schema_version": 1,
        "model_family": "codet5p",
        "model_version": version,
        "base_version": base_version,
        "checkpoint": checkpoint,
        "task": task,
        "positive_label": positive,
        "negative_label": negative,
        "supported_languages": supported_languages,
        "dataset": str(source),
        "dataset_sha256": dataset_sha256,
        "samples": {split: len(rows) for split, rows in partitions.items()},
        "label_counts": {
            split: dict(Counter(sample.label for sample in rows))
            for split, rows in partitions.items()
        },
        "threshold": threshold,
        "temperature": temperature,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_metrics_by_language": test_by_language,
        "test_metrics_by_source": test_by_source,
        "deployment_gate": QUALITY_GATE,
        "passed_deployment_gate": passed,
        "runtime_ready": passed,
        "calibrated": True,
        "calibration_split": "validation",
        "family_isolation_verified": True,
        "config": {
            "max_length": max_length,
            "stride": stride,
            "maximum_code_characters": maximum_code_characters,
            "maximum_train_windows": maximum_train_windows,
            "maximum_eval_windows": maximum_eval_windows,
            "window_aggregation": "max_logit",
            "dropout": 0.15,
        },
        "training": {
            "device": str(device),
            "torch_version": torch.__version__,
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "pairwise_weight": pairwise_weight,
            "pairwise_margin": pairwise_margin,
            "seed": seed,
            "mixed_precision": use_amp,
            "resumed_from": resumed_from or None,
        },
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (destination / "codet5p_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _select_languages(
    samples: list[CodeSample],
    target_language: str,
    positive: str,
    negative: str,
) -> list[str]:
    requested = str(target_language or "all").lower()
    candidates = sorted({sample.language for sample in samples})
    if requested != "all":
        candidates = [requested]
    supported = []
    for language in candidates:
        valid = True
        for split in ("train", "validation", "test"):
            labels = {
                sample.label for sample in samples
                if sample.language == language and sample.split == split
            }
            if not {positive, negative}.issubset(labels):
                valid = False
                break
        if valid:
            supported.append(language)
    if not supported:
        raise ValueError(
            f"no selected language has both {negative}/{positive} labels in train, validation, and test"
        )
    return supported


def _validate_partitions(
    partitions: dict[str, list[CodeSample]],
    positive: str,
    negative: str,
) -> None:
    for split, rows in partitions.items():
        if not rows:
            raise ValueError(f"CodeT5+ 220M dataset split is empty: {split}")
        labels = {sample.label for sample in rows}
        if not {positive, negative}.issubset(labels):
            raise ValueError(f"CodeT5+ 220M dataset split is missing one class: {split}")


def _validate_family_isolation(samples: list[CodeSample]) -> None:
    assignments: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        family = sample.family.strip()
        if family:
            assignments[family].add(sample.split)
    overlaps = sorted(family for family, splits in assignments.items() if len(splits) > 1)
    if overlaps:
        raise ValueError(
            "repository/family leakage across splits: " + ", ".join(overlaps[:5])
        )


def _pair_aware_batches(
    samples: list[CodeSample],
    batch_size: int,
    seed: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    paired: dict[str, list[int]] = defaultdict(list)
    singles = []
    for index, sample in enumerate(samples):
        if sample.pair_id:
            paired[sample.pair_id].append(index)
        else:
            singles.append(index)
    units: list[list[int]] = []
    used: set[int] = set()
    for indices in paired.values():
        positives = [index for index in indices if samples[index].label != "benign"]
        negatives = [index for index in indices if samples[index].label == "benign"]
        if positives and negatives:
            units.append([positives[0], negatives[0]])
            used.update((positives[0], negatives[0]))
    single_indices = set(singles)
    singles.extend(index for index in range(len(samples)) if index not in used and index not in single_indices)
    units.extend([[index] for index in singles])
    rng.shuffle(units)
    batches: list[list[int]] = []
    current: list[int] = []
    for unit in units:
        if current and len(current) + len(unit) > batch_size:
            batches.append(current)
            current = []
        current.extend(unit)
    if current:
        batches.append(current)
    return batches


def _tokenize_rows(
    tokenizer: Any,
    rows: list[CodeSample],
    positive: str,
    *,
    max_length: int,
    stride: int,
    maximum_code_characters: int,
    maximum_windows: int,
) -> dict[str, Any]:
    import torch

    codes = [sample.code[:maximum_code_characters] for sample in rows]
    encoded = tokenizer(
        codes,
        add_special_tokens=True,
        max_length=max_length,
        stride=min(stride, max_length // 2),
        truncation=True,
        padding=True,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )
    mapping = encoded.pop("overflow_to_sample_mapping")
    counts: dict[int, int] = defaultdict(int)
    keep = []
    for index, row_index in enumerate(mapping.tolist()):
        if counts[row_index] >= maximum_windows:
            continue
        counts[row_index] += 1
        keep.append(index)
    keep_tensor = torch.tensor(keep, dtype=torch.long)
    input_ids = encoded["input_ids"].index_select(0, keep_tensor)
    attention_mask = encoded["attention_mask"].index_select(0, keep_tensor)
    row_indices = mapping.index_select(0, keep_tensor).to(torch.long)
    labels = torch.tensor(
        [1.0 if rows[index].label == positive else 0.0 for index in row_indices.tolist()],
        dtype=torch.float32,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "row_indices": row_indices,
        "labels": labels,
        "rows": rows,
    }


def _paired_ranking_loss(
    logits: Any,
    row_indices: Any,
    rows: list[CodeSample],
    positive: str,
    *,
    margin: float,
) -> Any:
    import torch
    from torch.nn import functional as F

    row_logits = []
    for index in range(len(rows)):
        values = logits[row_indices == index]
        row_logits.append(values.mean() if values.numel() else logits.new_zeros(()))
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for index, sample in enumerate(rows):
        if not sample.pair_id:
            continue
        key = "positive" if sample.label == positive else "negative"
        grouped[sample.pair_id][key] = row_logits[index]
    losses = [
        F.softplus(float(margin) - (values["positive"] - values["negative"]))
        for values in grouped.values()
        if "positive" in values and "negative" in values
    ]
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def _predict_logits(
    model: Any,
    samples: list[CodeSample],
    collate_fn: Any,
    device: Any,
    *,
    batch_size: int,
    use_amp: bool,
) -> list[float]:
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    output: list[float] = []
    loader = DataLoader(
        SampleDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, _ = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
            row_indices = batch["row_indices"].to(device)
            for index in range(len(batch["rows"])):
                values = logits[row_indices == index]
                output.append(float(values.max().detach().cpu()))
    return output


def _fit_temperature(logits: list[float], labels: list[int]) -> float:
    import torch
    from torch.nn import functional as F

    if not logits or len(set(labels)) < 2:
        return 1.0
    values = torch.tensor(logits, dtype=torch.float64)
    targets = torch.tensor(labels, dtype=torch.float64)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> Any:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(values / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def _select_threshold(labels: list[int], probabilities: list[float]) -> float:
    candidates = sorted({0.01, 0.99, *[round(index / 100, 2) for index in range(2, 99)]})
    scored = []
    for threshold in candidates:
        metrics = _metrics(labels, probabilities, threshold)
        violations = (
            max(0.0, QUALITY_GATE["minimum_precision"] - metrics["precision"])
            + max(0.0, metrics["false_positive_rate"] - QUALITY_GATE["maximum_false_positive_rate"])
            + max(0.0, metrics["false_negative_rate"] - QUALITY_GATE["maximum_false_negative_rate"])
        )
        scored.append((violations, -metrics["f1"], threshold))
    return float(min(scored)[2])


def _metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict[str, Any]:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / max(1, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "brier_score": sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / max(1, len(labels)),
        "samples": len(labels),
        "positive_samples": sum(labels),
        "negative_samples": len(labels) - sum(labels),
        "threshold": threshold,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _segment_metrics(
    samples: list[CodeSample],
    labels: list[int],
    probabilities: list[float],
    threshold: float,
    field: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[str(getattr(sample, field) or "unknown")].append(index)
    return {
        name: _metrics(
            [labels[index] for index in indices],
            [probabilities[index] for index in indices],
            threshold,
        )
        for name, indices in sorted(groups.items())
    }


def _passes_gate(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics.get("precision", 0.0)) >= QUALITY_GATE["minimum_precision"]
        and float(metrics.get("false_positive_rate", 1.0)) <= QUALITY_GATE["maximum_false_positive_rate"]
        and float(metrics.get("false_negative_rate", 1.0)) <= QUALITY_GATE["maximum_false_negative_rate"]
    )


def _sigmoid_list(values: Iterable[float]) -> list[float]:
    output = []
    for value in values:
        if value >= 0:
            output.append(1.0 / (1.0 + math.exp(-value)))
        else:
            exponent = math.exp(value)
            output.append(exponent / (1.0 + exponent))
    return output


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read CodeT5+ 220M manifest: {path}") from exc
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CodeT5+ 220M for code-risk classification")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default="Salesforce/codet5p-220m")
    parser.add_argument("--base-version", default="codet5p-220m-base")
    parser.add_argument("--base-artifact-dir")
    parser.add_argument("--task", choices=sorted(TASKS), default="vulnerability_risk")
    parser.add_argument("--target-language", default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--maximum-code-characters", type=int, default=32768)
    parser.add_argument("--maximum-train-windows", type=int, default=3)
    parser.add_argument("--maximum-eval-windows", type=int, default=8)
    parser.add_argument("--pairwise-weight", type=float, default=0.2)
    parser.add_argument("--pairwise-margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()
    manifest = train_codet5p(
        args.dataset,
        args.output_dir,
        checkpoint=args.checkpoint,
        base_version=args.base_version,
        base_artifact_dir=args.base_artifact_dir,
        task=args.task,
        target_language=args.target_language,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        stride=args.stride,
        maximum_code_characters=args.maximum_code_characters,
        maximum_train_windows=args.maximum_train_windows,
        maximum_eval_windows=args.maximum_eval_windows,
        pairwise_weight=args.pairwise_weight,
        pairwise_margin=args.pairwise_margin,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
