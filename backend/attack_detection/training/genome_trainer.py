"""Train a quality-gated twin GAT on paired source/artifact graphs.

Input is JSONL with ``source_graph``, ``artifact_graph``, ``label`` and
``split``. Label may be ``consistent``/``tampered`` or 0/1. The resulting
manifest is runtime-ready only when the independent test split passes every
configured release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .genome_features import EDGE_TYPES, NODE_TYPES, feature_dimension, graph_to_pyg


DEPLOYMENT_GATE = {
    "minimum_precision": 0.9,
    "maximum_false_positive_rate": 0.1,
    "maximum_false_negative_rate": 0.1,
}


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from torch_geometric.data import Batch
    except ImportError as exc:
        raise SystemExit("software-genome training requires torch and torch-geometric") from exc
    from .genome_twin_model import TwinGATVerifier

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = _load_records(Path(args.pairs))
    splits = {name: [] for name in ("train", "validation", "test")}
    for record in records:
        split = str(record.get("split") or "")
        if split not in splits:
            continue
        label = _label_value(record.get("label"))
        source_graph = record.get("source_graph")
        artifact_graph = record.get("artifact_graph")
        if label is None or not isinstance(source_graph, dict) or not isinstance(artifact_graph, dict):
            continue
        splits[split].append((
            graph_to_pyg(source_graph, "source", torch),
            graph_to_pyg(artifact_graph, "artifact", torch),
            label,
        ))
    for name, values in splits.items():
        if {item[2] for item in values} != {0, 1}:
            raise SystemExit(f"both consistent and tampered pairs are required in {name}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TwinGATVerifier(
        feature_dimension(),
        len(EDGE_TYPES),
        hidden=args.hidden,
        heads=args.heads,
        dropout=args.dropout,
        embedding_dim=args.embedding_dim,
    ).to(device)
    labels = [item[2] for item in splits["train"]]
    positives = sum(labels)
    negatives = len(labels) - positives
    positive_weight = torch.tensor([negatives / max(1, positives)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state = None
    best_f1 = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = list(splits["train"])
        random.shuffle(shuffled)
        total_loss = 0.0
        for chunk in _chunks(shuffled, args.batch_size):
            source_batch = Batch.from_data_list([item[0] for item in chunk]).to(device)
            artifact_batch = Batch.from_data_list([item[1] for item in chunk]).to(device)
            target = torch.tensor([item[2] for item in chunk], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(source_batch, artifact_batch), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(chunk)
        validation_probabilities, validation_labels = _predict(
            model, splits["validation"], args.batch_size, device, torch, Batch
        )
        threshold, validation_metrics = _best_threshold(validation_probabilities, validation_labels)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(shuffled),
            "validation": validation_metrics,
        })
        if validation_metrics["f1"] > best_f1 + 1e-6:
            best_f1 = validation_metrics["f1"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise SystemExit("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_probabilities, validation_labels = _predict(
        model, splits["validation"], args.batch_size, device, torch, Batch
    )
    threshold, validation_metrics = _best_threshold(validation_probabilities, validation_labels)
    test_probabilities, test_labels = _predict(
        model, splits["test"], args.batch_size, device, torch, Batch
    )
    test_metrics = _metrics(test_probabilities, test_labels, threshold)
    runtime_ready = _passes_gate(test_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_hash = hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest()
    version = f"genome-twin-gat-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{dataset_hash[:12]}"
    weights_path = output_dir / "genome_twin_classifier.pt"
    torch.save(best_state, weights_path)
    manifest = {
        "schema_version": 1,
        "model_version": version,
        "task": "source_artifact_tamper_detection",
        "architecture": "shared-weight twin GATv2",
        "positive_label": "tampered",
        "dataset_sha256": dataset_hash,
        "node_types": NODE_TYPES,
        "edge_types": EDGE_TYPES,
        "feature_dimension": feature_dimension(),
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "deployment_gate": DEPLOYMENT_GATE,
        "runtime_ready": runtime_ready,
        "runtime_note": "Only a quality-gated paired model may override the transparent baseline.",
        "training": {
            "device": str(device),
            "epochs_completed": len(history),
            "seed": args.seed,
            "hidden": args.hidden,
            "heads": args.heads,
            "dropout": args.dropout,
            "embedding_dim": args.embedding_dim,
        },
        "split_counts": {name: len(values) for name, values in splits.items()},
        "files": [weights_path.name],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output_dir / "genome_twin_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "genome_twin_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must contain an object")
            records.append(value)
    return records


def _label_value(value: object) -> int | None:
    if value in {1, True, "1", "tampered", "mismatch"}:
        return 1
    if value in {0, False, "0", "consistent", "match"}:
        return 0
    return None


def _chunks(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index:index + max(1, size)] for index in range(0, len(values), max(1, size))]


def _predict(model: Any, records: list[Any], batch_size: int, device: Any, torch: Any, batch_type: Any) -> tuple[list[float], list[int]]:
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for chunk in _chunks(records, batch_size):
            source_batch = batch_type.from_data_list([item[0] for item in chunk]).to(device)
            artifact_batch = batch_type.from_data_list([item[1] for item in chunk]).to(device)
            probabilities.extend(torch.sigmoid(model(source_batch, artifact_batch)).cpu().tolist())
            labels.extend(item[2] for item in chunk)
    return [float(value) for value in probabilities], labels


def _best_threshold(probabilities: list[float], labels: list[int]) -> tuple[float, dict[str, float]]:
    candidates = sorted({0.5, *probabilities})
    scored = [(threshold, _metrics(probabilities, labels, threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[1]["f1"], item[1]["precision"], -item[1]["false_positive_rate"]))


def _metrics(probabilities: list[float], labels: list[int], threshold: float) -> dict[str, float]:
    predictions = [int(value >= threshold) for value in probabilities]
    true_positive = sum(prediction == label == 1 for prediction, label in zip(predictions, labels))
    true_negative = sum(prediction == label == 0 for prediction, label in zip(predictions, labels))
    false_positive = sum(prediction == 1 and label == 0 for prediction, label in zip(predictions, labels))
    false_negative = sum(prediction == 0 and label == 1 for prediction, label in zip(predictions, labels))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / max(1e-12, precision + recall), 6),
        "false_positive_rate": round(false_positive / max(1, false_positive + true_negative), 6),
        "false_negative_rate": round(false_negative / max(1, false_negative + true_positive), 6),
        "accuracy": round((true_positive + true_negative) / max(1, len(labels)), 6),
    }


def _passes_gate(metrics: dict[str, float]) -> bool:
    return (
        metrics["precision"] >= DEPLOYMENT_GATE["minimum_precision"]
        and metrics["false_positive_rate"] <= DEPLOYMENT_GATE["maximum_false_positive_rate"]
        and metrics["false_negative_rate"] <= DEPLOYMENT_GATE["maximum_false_negative_rate"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "models"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(train(arguments), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
