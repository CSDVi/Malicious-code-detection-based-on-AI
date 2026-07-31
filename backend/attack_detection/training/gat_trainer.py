"""Train and evaluate a calibrated PyTorch Geometric GATv2 graph classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NODE_TYPES = ["package", "file", "function", "dangerous_api"]
EDGE_TYPES = ["contains", "declares", "import", "dependency", "call", "version_diff"]
LEGACY_LANGUAGES = ["python", "javascript", "typescript", "java", "php", "bash", "config", "unknown"]
LANGUAGES = [
    "python", "javascript", "typescript", "java", "kotlin", "php", "bash", "config", "go",
    "powershell", "batch", "c", "cpp", "csharp", "ruby", "rust", "scala", "lua", "perl",
    "html", "sql", "unknown",
]
NAME_BUCKETS = 32
LEXICAL_BUCKETS = 64
GRAPH_FEATURES = 16
API_TOKENS_V2 = [
    "eval", "exec", "system", "popen", "subprocess", "child_process", "shell_exec",
    "passthru", "proc_open", "pcntl_exec", "processbuilder", "runtime.getruntime",
    "os.exec", "requests.get", "urllib.request", "webclient", "fetch", "curl", "wget",
    "socket", "base64_decode", "gzinflate", "gzuncompress", "str_rot13", "preg_replace",
    "create_function", "file_put_contents", "fwrite", "move_uploaded_file", "chmod",
    "assert", "rawurldecode", "urldecode", "hex2bin", "convert_uudecode",
    "call_user_func", "call_user_func_array", "file_get_contents", "fopen", "readfile",
    "unlink", "rename", "copy", "scandir", "glob", "opendir", "readdir", "fsockopen",
    "stream_socket_client", "php://input", "$_post", "$_get", "$_request", "$_cookie",
    "$_files", "$_server", "$_env",
    "behavior_input_execution_chain", "behavior_decode_execution_chain",
    "behavior_input_file_write_chain", "behavior_remote_execution_chain",
    "behavior_encoded_payload", "behavior_variable_function_call",
]
API_TOKENS_V3 = API_TOKENS_V2 + [
    "invoke-expression", "iex", "start-process", "new-object", "downloadstring",
    "downloadfile", "invoke-webrequest", "invoke-restmethod", "frombase64string",
    "encodedcommand", "add-mppreference", "set-mppreference", "register-scheduledtask",
    "schtasks", "certutil", "bitsadmin", "mshta", "rundll32", "regsvr32", "wmic",
    "cmd", "powershell", "bash", "sh", "netcat", "nc", "chattr", "crontab", "nohup",
    "winexec", "shellexecute", "createprocess", "virtualalloc", "virtualprotect",
    "writeprocessmemory", "createremotethread", "loadlibrary", "getprocaddress",
    "internetopenurl", "urldownloadtofile", "winhttpopen", "regsetvalue", "openprocess",
    "adjusttokenprivileges", "process.start", "assembly.load", "powershell.create",
    "registry.setvalue", "dllimport", "connect", "recv", "send",
    "behavior_process_injection_chain", "behavior_encoded_command",
    "behavior_defense_evasion_chain", "behavior_persistence_chain",
]
API_TOKENS_V4 = API_TOKENS_V3 + [
    "behavior_reverse_shell", "behavior_credential_access", "behavior_phishing",
    "behavior_security_evasion", "behavior_embedded_executable",
    "behavior_uac_bypass", "behavior_command_and_control",
]
API_TOKENS_V5 = API_TOKENS_V4 + [
    "context_ci_or_build", "context_documented_system_utility",
]
API_TOKENS_V6 = API_TOKENS_V5 + [
    "behavior_network_flood", "behavior_fork_bomb",
]
API_TOKENS_V8 = API_TOKENS_V6 + [
    "behavior_download_execute_pipe",
]
# Compatibility alias for callers that still refer to the schema-v2 vocabulary.
API_TOKENS = API_TOKENS_V2
LANGUAGE_MINIMUMS = {"train": 20, "validation": 10, "test": 10}
DEPLOYMENT_GATE = {
    "minimum_precision": 0.9,
    "maximum_false_positive_rate": 0.1,
    "maximum_false_negative_rate": 0.1,
}


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
    except ImportError as exc:
        raise SystemExit("GATv2 training requires torch and torch-geometric") from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = _load_jsonl(Path(args.graphs))
    positive_label = "vulnerable" if args.task == "vulnerability_risk" else "malicious"
    if args.limit:
        records = _balanced_limit(records, args.limit, positive_label)
    accepted_labels = {positive_label, "fixed" if args.task == "vulnerability_risk" else "benign"}
    if args.task == "vulnerability_risk":
        accepted_labels.add("benign")
    datasets = {split: [] for split in ("train", "validation", "test")}
    records_by_split = {split: [] for split in ("train", "validation", "test")}
    for record in records:
        split = str(record.get("split") or "")
        if split in datasets and record.get("label") in accepted_labels:
            datasets[split].append(_to_data(
                record,
                torch,
                positive_label,
                feature_schema_version=args.feature_schema_version,
            ))
            records_by_split[split].append(record)
    if any(not values for values in datasets.values()):
        raise SystemExit("Each split must contain graph records")
    for split, values in datasets.items():
        labels = {int(item.y.item()) for item in values}
        if labels != {0, 1}:
            raise SystemExit(f"Both labels are required in {split}: {labels}")

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            input_dim = feature_dimension(LANGUAGES, args.feature_schema_version)
            edge_dim = len(EDGE_TYPES)
            self.conv1 = GATv2Conv(input_dim, args.hidden, heads=args.heads, edge_dim=edge_dim, dropout=args.dropout)
            self.conv2 = GATv2Conv(args.hidden * args.heads, args.hidden, heads=1, concat=False, edge_dim=edge_dim, dropout=args.dropout)
            pooled_dim = args.hidden * 2 if args.pooling == "mean_max" else args.hidden
            pooled_dim += graph_feature_dimension(args.feature_schema_version)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(pooled_dim, args.hidden), torch.nn.ReLU(),
                torch.nn.Dropout(args.dropout), torch.nn.Linear(args.hidden, 2),
            )

        def forward(self, batch: Any) -> Any:
            value = functional.elu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))
            value = functional.dropout(value, p=args.dropout, training=self.training)
            value = functional.elu(self.conv2(value, batch.edge_index, batch.edge_attr))
            pooled = global_mean_pool(value, batch.batch)
            if args.pooling == "mean_max":
                pooled = torch.cat((pooled, global_max_pool(value, batch.batch)), dim=1)
            if graph_feature_dimension(args.feature_schema_version):
                pooled = torch.cat((pooled, batch.graph_features), dim=1)
            return self.classifier(pooled)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = Model().to(device)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(datasets["validation"], batch_size=args.batch_size)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size)
    positives = sum(int(item.y.item()) for item in datasets["train"])
    negatives = len(datasets["train"]) - positives
    weights = torch.tensor([len(datasets["train"]) / (2 * negatives), len(datasets["train"]) / (2 * positives)], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    best_state = None
    best_f1 = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_total += float(loss.item()) * batch.num_graphs
        val_logits, val_labels = _predict(model, validation_loader, device, torch)
        threshold, val_metrics = _best_threshold(val_logits, val_labels)
        history.append({"epoch": epoch, "train_loss": loss_total / len(datasets["train"]), "validation": val_metrics})
        if val_metrics["f1"] > best_f1 + 1e-6:
            best_f1 = val_metrics["f1"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(json.dumps(history[-1], ensure_ascii=True), flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise SystemExit("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    val_logits, val_labels = _predict(model, validation_loader, device, torch)
    temperature = _fit_temperature(val_logits, val_labels, torch)
    threshold, validation_metrics = _best_threshold(val_logits, val_labels, temperature)
    test_logits, test_labels = _predict(model, test_loader, device, torch)
    all_test_metrics = _metrics(test_logits, test_labels, threshold, temperature)
    language_coverage = _language_coverage(records_by_split, positive_label)
    eligible_languages = sorted(
        language for language, values in language_coverage.items() if values["eligible"]
    )
    language_thresholds, validation_metrics_by_language, test_metrics_by_language = (
        _calibrate_language_thresholds(
            records_by_split, val_logits, val_labels, test_logits, test_labels,
            eligible_languages, threshold, temperature,
        )
    )
    supported_languages = [
        language for language in eligible_languages
        if _passes_deployment_gate(test_metrics_by_language[language])
    ]
    test_metrics = _conservative_language_summary(
        test_metrics_by_language, supported_languages,
    ) if supported_languages else all_test_metrics

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = "gatv2_vulnerability" if args.task == "vulnerability_risk" else "gatv2"
    weights_path = output_dir / f"{artifact_stem}_classifier.pt"
    torch.save(best_state, weights_path)
    dataset_hash = _sha256(Path(args.graphs))
    version_prefix = "gatv2-vulnerability" if args.task == "vulnerability_risk" else "gatv2"
    version = f"{version_prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{dataset_hash[:12]}"
    manifest = {
        "schema_version": 1,
        "model_version": version,
        "task": args.task,
        "positive_label": positive_label,
        "negative_labels": sorted(accepted_labels - {positive_label}),
        "negative_label_scope": (
            "fixed means fixed for the indexed CVE, not globally vulnerability-free"
            if args.task == "vulnerability_risk" else "benign"
        ),
        "architecture": "torch_geometric.nn.GATv2Conv",
        "dataset_sha256": dataset_hash,
        "node_types": NODE_TYPES,
        "edge_types": EDGE_TYPES,
        "languages": LANGUAGES,
        "observed_languages": sorted(language_coverage),
        "supported_languages": supported_languages,
        "task_language_support": {args.task: supported_languages},
        "language_coverage": language_coverage,
        "language_thresholds": {
            language: language_thresholds[language] for language in supported_languages
        },
        "validation_metrics_by_language": validation_metrics_by_language,
        "test_metrics_by_language": test_metrics_by_language,
        "deployment_gate": DEPLOYMENT_GATE,
        "feature_schema": (
            "node_type+language+"
            + (
                "sha256_name_bucket+"
                if args.feature_schema_version < 4 else ""
            )
            + "normalized_in_out_degree"
            + ("+dangerous_api_identity" if args.feature_schema_version >= 2 else "")
            + (
                "+hashed_source_token_frequency"
                if args.feature_schema_version >= 7 else ""
            )
            + (
                "+graph_level_structural_behavior_summary"
                if args.feature_schema_version >= 9 else ""
            )
        ),
        "label_leakage_guard": "node.label and node.line_labels are excluded from features",
        "calibrated": True,
        "threshold_policy": (
            "upper quartile of the widest contiguous validation interval "
            "that passes the deployment gate"
        ),
        "temperature": temperature,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "all_test_metrics": all_test_metrics,
        "metric_scope": "worst validated language using validation-calibrated thresholds",
        "training": {
            "device": str(device), "epochs_completed": len(history), "seed": args.seed,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "hidden": args.hidden, "heads": args.heads, "dropout": args.dropout,
            "pooling": args.pooling,
            "feature_schema_version": args.feature_schema_version,
        },
        "split_counts": {split: len(values) for split, values in datasets.items()},
        "files": [weights_path.name],
        "runtime_ready": not bool(args.limit) and bool(supported_languages),
        "runtime_note": (
            "Project inference uses the configured external PyTorch interpreter and "
            "the dominant validated language's independently calibrated threshold."
        ),
        "limited_smoke_run": bool(args.limit),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output_dir / f"{artifact_stem}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{artifact_stem}_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _to_data(
    graph: dict[str, Any], torch: Any, positive_label: str = "malicious",
    languages: list[str] | None = None,
    feature_schema_version: int = 1,
) -> Any:
    from torch_geometric.data import Data
    nodes = list(graph.get("nodes") or [])
    index = {str(node.get("id")): position for position, node in enumerate(nodes)}
    sources, targets, edge_features = [], [], []
    indegree = [0] * len(nodes)
    outdegree = [0] * len(nodes)
    for edge in graph.get("edges") or []:
        source = index.get(str(edge.get("source")))
        target = index.get(str(edge.get("target")))
        edge_type = str(edge.get("type") or "")
        if source is None or target is None or edge_type not in EDGE_TYPES:
            continue
        feature = [float(edge_type == value) for value in EDGE_TYPES]
        for left, right in ((source, target), (target, source)):
            sources.append(left); targets.append(right); edge_features.append(feature)
            outdegree[left] += 1; indegree[right] += 1
    if not sources:
        sources = list(range(len(nodes))); targets = list(range(len(nodes)))
        edge_features = [[0.0] * len(EDGE_TYPES) for _ in nodes]
    maximum = max(1, len(nodes) - 1)
    features = []
    language_schema = languages or LANGUAGES
    for position, node in enumerate(nodes):
        node_type = str(node.get("type") or "")
        language = str(node.get("language") or "unknown")
        name = str(node.get("name") or node.get("id") or "")
        bucket = int(hashlib.sha256(name.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % NAME_BUCKETS
        name_features = (
            [float(bucket == value) for value in range(NAME_BUCKETS)]
            if feature_schema_version < 4 else []
        )
        lexical_features = []
        if feature_schema_version >= 7:
            raw_lexical = node.get("lexical_buckets")
            if isinstance(raw_lexical, list):
                lexical_features = [
                    float(value) for value in raw_lexical[:LEXICAL_BUCKETS]
                ]
            lexical_features.extend(
                [0.0] * (LEXICAL_BUCKETS - len(lexical_features))
            )
        features.append(
            [float(node_type == value) for value in NODE_TYPES]
            + [float(language == value) for value in language_schema]
            + name_features
            + [indegree[position] / maximum, outdegree[position] / maximum]
            + (
                [
                    float(node_type == "dangerous_api" and name.lower() == token)
                    for token in _api_tokens(feature_schema_version)
                ]
                if feature_schema_version >= 2 else []
            )
            + lexical_features
        )
    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor([sources, targets], dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float32),
        y=torch.tensor(1 if graph["label"] == positive_label else 0, dtype=torch.long),
        graph_features=torch.tensor(
            [_graph_features(graph)]
            if feature_schema_version >= 9 else [[]],
            dtype=torch.float32,
        ),
    )


def feature_dimension(languages: list[str], feature_schema_version: int = 1) -> int:
    return (
        len(NODE_TYPES) + len(languages)
        + (NAME_BUCKETS if feature_schema_version < 4 else 0) + 2
        + (len(_api_tokens(feature_schema_version)) if feature_schema_version >= 2 else 0)
        + (LEXICAL_BUCKETS if feature_schema_version >= 7 else 0)
    )


def graph_feature_dimension(feature_schema_version: int = 1) -> int:
    return GRAPH_FEATURES if feature_schema_version >= 9 else 0


def _graph_features(graph: dict[str, Any]) -> list[float]:
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    node_total = max(1, len(nodes))
    edge_total = max(1, len(edges))
    node_counts = {
        value: sum(str(node.get("type") or "") == value for node in nodes)
        for value in NODE_TYPES
    }
    edge_counts = {
        value: sum(str(edge.get("type") or "") == value for edge in edges)
        for value in EDGE_TYPES
    }
    api_names = [
        str(node.get("name") or "").lower()
        for node in nodes
        if node.get("type") == "dangerous_api"
    ]
    behavior_names = [name for name in api_names if name.startswith("behavior_")]
    high_risk = {
        "behavior_process_injection_chain", "behavior_encoded_command",
        "behavior_reverse_shell", "behavior_credential_access",
        "behavior_phishing", "behavior_security_evasion",
        "behavior_embedded_executable", "behavior_uac_bypass",
        "behavior_command_and_control", "behavior_network_flood",
        "behavior_fork_bomb", "behavior_download_execute_pipe",
    }
    remote = {
        "behavior_remote_execution_chain", "behavior_command_and_control",
        "behavior_download_execute_pipe",
    }
    persistence_evasion = {
        "behavior_persistence_chain", "behavior_defense_evasion_chain",
        "behavior_security_evasion", "behavior_uac_bypass",
    }
    file_nodes = [node for node in nodes if node.get("type") == "file"]
    lexical_files = sum(
        isinstance(node.get("lexical_buckets"), list)
        and any(float(value) != 0.0 for value in node["lexical_buckets"])
        for node in file_nodes
    )
    languages = {
        str(node.get("language") or "").lower()
        for node in file_nodes
        if node.get("language")
    }
    api_total = max(1, len(api_names))
    return [
        min(1.0, math.log1p(len(nodes)) / math.log1p(1000)),
        min(1.0, math.log1p(len(edges)) / math.log1p(2000)),
        node_counts["file"] / node_total,
        node_counts["function"] / node_total,
        node_counts["package"] / node_total,
        node_counts["dangerous_api"] / node_total,
        edge_counts["call"] / edge_total,
        (edge_counts["import"] + edge_counts["dependency"]) / edge_total,
        len(behavior_names) / api_total,
        (len(api_names) - len(behavior_names)) / api_total,
        float(bool(high_risk.intersection(api_names))),
        float(bool(remote.intersection(api_names))),
        float(bool(persistence_evasion.intersection(api_names))),
        min(1.0, len(edges) / node_total / 8.0),
        min(1.0, len(languages) / 8.0),
        lexical_files / max(1, len(file_nodes)),
    ]


def _api_tokens(feature_schema_version: int) -> list[str]:
    if feature_schema_version >= 8:
        return API_TOKENS_V8
    if feature_schema_version >= 6:
        return API_TOKENS_V6
    if feature_schema_version >= 5:
        return API_TOKENS_V5
    if feature_schema_version >= 4:
        return API_TOKENS_V4
    return API_TOKENS_V3 if feature_schema_version >= 3 else API_TOKENS_V2


def _predict(model: Any, loader: Any, device: Any, torch: Any) -> tuple[list[list[float]], list[int]]:
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits.extend(model(batch).detach().cpu().tolist())
            labels.extend(batch.y.detach().cpu().tolist())
    return logits, [int(value) for value in labels]


def _probabilities(logits: list[list[float]], temperature: float = 1.0) -> list[float]:
    output = []
    for negative, positive in logits:
        value = (positive - negative) / max(temperature, 1e-4)
        output.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value)))))
    return output


def _best_threshold(logits: list[list[float]], labels: list[int], temperature: float = 1.0) -> tuple[float, dict[str, float]]:
    best = (0.5, _metrics(logits, labels, 0.5, temperature))
    for step in range(10, 91):
        threshold = step / 100
        candidate = _metrics(logits, labels, threshold, temperature)
        if candidate["f1"] > best[1]["f1"]:
            best = (threshold, candidate)
    return best


def _metrics(logits: list[list[float]], labels: list[int], threshold: float, temperature: float = 1.0) -> dict[str, float]:
    probabilities = _probabilities(logits, temperature)
    predicted = [int(value >= threshold) for value in probabilities]
    tp = sum(a == b == 1 for a, b in zip(labels, predicted))
    tn = sum(a == b == 0 for a, b in zip(labels, predicted))
    fp = sum(a == 0 and b == 1 for a, b in zip(labels, predicted))
    fn = sum(a == 1 and b == 0 for a, b in zip(labels, predicted))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "accuracy": (tp + tn) / max(1, len(labels)), "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "false_positive_rate": fp / max(1, fp + tn), "false_negative_rate": fn / max(1, fn + tp),
        "roc_auc": _auc(probabilities, labels), "threshold": threshold,
    }


def _record_languages(record: dict[str, Any]) -> set[str]:
    return {
        str(node.get("language")).lower()
        for node in (record.get("nodes") or [])
        if node.get("type") == "file" and node.get("language")
    }


def _language_coverage(
    records_by_split: dict[str, list[dict[str, Any]]], positive_label: str,
) -> dict[str, Any]:
    languages = sorted({
        language
        for records in records_by_split.values()
        for record in records
        for language in _record_languages(record)
    })
    output: dict[str, Any] = {}
    for language in languages:
        split_counts = {}
        eligible = True
        for split, records in records_by_split.items():
            relevant = [record for record in records if language in _record_languages(record)]
            positive = sum(record.get("label") == positive_label for record in relevant)
            negative = len(relevant) - positive
            split_counts[split] = {"positive": positive, "negative": negative}
            minimum = LANGUAGE_MINIMUMS[split]
            eligible = eligible and positive >= minimum and negative >= minimum
        output[language] = {
            "splits": split_counts,
            "minimum_per_class": LANGUAGE_MINIMUMS,
            "eligible": eligible,
        }
    return output


def _calibrate_language_thresholds(
    records_by_split: dict[str, list[dict[str, Any]]],
    validation_logits: list[list[float]], validation_labels: list[int],
    test_logits: list[list[float]], test_labels: list[int],
    eligible_languages: list[str], global_threshold: float, temperature: float,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    languages = sorted({
        language
        for records in records_by_split.values()
        for record in records
        for language in _record_languages(record)
    })
    thresholds: dict[str, float] = {}
    validation_metrics: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}
    for language in languages:
        validation_indices = [
            index for index, record in enumerate(records_by_split["validation"])
            if language in _record_languages(record)
        ]
        test_indices = [
            index for index, record in enumerate(records_by_split["test"])
            if language in _record_languages(record)
        ]
        selected_threshold = global_threshold
        if language in eligible_languages:
            selected_threshold, selected_metrics = _best_gated_threshold(
                [validation_logits[index] for index in validation_indices],
                [validation_labels[index] for index in validation_indices], temperature,
            )
        else:
            selected_metrics = _metrics(
                [validation_logits[index] for index in validation_indices],
                [validation_labels[index] for index in validation_indices],
                selected_threshold, temperature,
            )
        selected_metrics["samples"] = len(validation_indices)
        thresholds[language] = selected_threshold
        validation_metrics[language] = selected_metrics
        test_metrics[language] = _metrics(
            [test_logits[index] for index in test_indices],
            [test_labels[index] for index in test_indices],
            selected_threshold, temperature,
        )
        test_metrics[language]["samples"] = len(test_indices)
    return thresholds, validation_metrics, test_metrics


def _best_gated_threshold(
    logits: list[list[float]], labels: list[int], temperature: float,
) -> tuple[float, dict[str, float]]:
    candidates: list[tuple[float, dict[str, float]]] = []
    for step in range(10, 91):
        threshold = step / 100
        metrics = _metrics(logits, labels, threshold, temperature)
        if (
            metrics["precision"] >= DEPLOYMENT_GATE["minimum_precision"]
            and metrics["false_positive_rate"] <= DEPLOYMENT_GATE["maximum_false_positive_rate"]
            and metrics["false_negative_rate"] <= DEPLOYMENT_GATE["maximum_false_negative_rate"]
        ):
            candidates.append((threshold, metrics))
    if not candidates:
        return _best_threshold(logits, labels, temperature)
    # A boundary threshold can look optimal on a small validation cohort yet
    # collapse under a tiny calibration shift. Select the upper quartile of
    # the widest contiguous gate-passing interval instead. This still uses
    # validation labels only, stays inside the FNR gate, and gives the
    # production scanner extra false-positive headroom.
    intervals: list[list[tuple[float, dict[str, float]]]] = []
    current: list[tuple[float, dict[str, float]]] = []
    for candidate in candidates:
        if current and round(candidate[0] - current[-1][0], 8) > 0.01000001:
            intervals.append(current)
            current = []
        current.append(candidate)
    if current:
        intervals.append(current)
    stable = max(
        intervals,
        key=lambda values: (
            len(values),
            sum(item[1]["f1"] for item in values) / len(values),
        ),
    )
    stable_target = stable[0][0] + (stable[-1][0] - stable[0][0]) * 0.75
    return min(
        stable,
        key=lambda item: (
            abs(item[0] - stable_target),
            -item[1]["f1"],
            item[1]["false_positive_rate"] + item[1]["false_negative_rate"],
        ),
    )


def _conservative_language_summary(
    metrics_by_language: dict[str, dict[str, float]], languages: list[str],
) -> dict[str, Any]:
    rows = [metrics_by_language[language] for language in languages]
    minimum_keys = ("accuracy", "precision", "recall", "f1", "roc_auc")
    maximum_keys = ("false_positive_rate", "false_negative_rate")
    output: dict[str, Any] = {
        key: min(float(row[key]) for row in rows) for key in minimum_keys
    }
    output.update({key: max(float(row[key]) for row in rows) for key in maximum_keys})
    output["samples"] = sum(int(row.get("samples", 0)) for row in rows)
    output["aggregation"] = "worst_validated_language"
    output["supported_languages"] = list(languages)
    return output


def _passes_deployment_gate(metrics: dict[str, float]) -> bool:
    return (
        float(metrics.get("precision", 0.0)) >= DEPLOYMENT_GATE["minimum_precision"]
        and float(metrics.get("false_positive_rate", 1.0))
        <= DEPLOYMENT_GATE["maximum_false_positive_rate"]
        and float(metrics.get("false_negative_rate", 1.0))
        <= DEPLOYMENT_GATE["maximum_false_negative_rate"]
    )


def _auc(probabilities: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(probabilities, labels))
    positives = sum(labels); negatives = len(labels) - positives
    if not positives or not negatives:
        return 0.0
    rank_sum = sum(rank for rank, (_, label) in enumerate(pairs, 1) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _fit_temperature(logits: list[list[float]], labels: list[int], torch: Any) -> float:
    values = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=80)
    criterion = torch.nn.CrossEntropyLoss()
    def closure() -> Any:
        optimizer.zero_grad()
        loss = criterion(values / log_temperature.exp().clamp(0.05, 20.0), targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 20.0).item())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _balanced_limit(
    records: list[dict[str, Any]], per_split_limit: int, positive_label: str = "malicious",
) -> list[dict[str, Any]]:
    output = []
    for split in ("train", "validation", "test"):
        values = [record for record in records if record.get("split") == split]
        half = max(1, per_split_limit // 2)
        negative_labels = {"fixed", "benign"} if positive_label == "vulnerable" else {"benign"}
        negatives = [record for record in values if record.get("label") in negative_labels][:half]
        positives = [record for record in values if record.get("label") == positive_label][:half]
        output.extend(negatives + positives)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a calibrated GATv2 graph classifier")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=("malicious_intent", "vulnerability_risk"), default="malicious_intent")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--pooling", choices=("mean", "mean_max"), default="mean_max")
    parser.add_argument(
        "--feature-schema-version",
        type=int,
        choices=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        default=9,
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test graph limit per split")
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
