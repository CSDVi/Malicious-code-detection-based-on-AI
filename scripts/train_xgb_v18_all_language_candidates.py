"""Train v18 malicious-intent XGBoost candidates for each language route."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROUTES = Path(
    os.environ.get(
        "XGB_ROUTE_DIR",
        str(ROOT / "backend/data/splits/xgb_v18_all_language_routes"),
    )
)
MODEL_ROOT = Path(
    os.environ.get(
        "XGB_MODEL_ROOT",
        str(ROOT / "backend/models/candidates/xgboost/v18_all_language_routes"),
    )
)
ARTIFACT_ROOT = Path(
    os.environ.get(
        "XGB_ARTIFACT_ROOT",
        str(ROOT / "artifacts/xgb_multilingual_optimization_20260727"),
    )
)
TRAIN_TAG = os.environ.get("XGB_TRAIN_TAG", "v18")
LOG_ROOT = ARTIFACT_ROOT / "logs" / f"{TRAIN_TAG}_all_language_routes"
SUMMARY_ROOT = ARTIFACT_ROOT / "evidence"
TRAINER = ROOT / "scripts/experiment_xgb_hybrid_language.py"

LANGUAGES = (
    "bash",
    "c",
    "config",
    "cpp",
    "go",
    "html",
    "java",
    "javascript",
    "php",
    "powershell",
    "python",
    "ruby",
    "rust",
)

QUALITY_GATE = {
    "min_precision": 0.90,
    "max_false_positive_rate": 0.10,
    "max_false_negative_rate": 0.10,
}


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    feature_mode: str
    text_transform: str
    positive_weight_mode: str
    positive_weight_multiplier: float
    hard_negative_weight: float
    hard_negative_fraction: float
    calibration_mode: str = "sigmoid"
    negative_source_weights: tuple[tuple[str, float], ...] = ()
    negative_family_weights: tuple[tuple[str, float], ...] = ()
    positive_source_weights: tuple[tuple[str, float], ...] = ()
    positive_family_weights: tuple[tuple[str, float], ...] = ()
    excluded_structured_features: tuple[str, ...] = ()
    threshold_target_fpr: float | None = None
    threshold_target_precision: float | None = None
    threshold_target_fnr: float | None = None
    threshold_plateau_position: float = 0.5


CONFIGS = {
    "raw_default": CandidateConfig(
        name="raw_default",
        feature_mode="hybrid",
        text_transform="raw",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
    ),
    "raw_balanced_hardneg": CandidateConfig(
        name="raw_balanced_hardneg",
        feature_mode="hybrid",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.0,
        hard_negative_weight=3.0,
        hard_negative_fraction=0.5,
    ),
    "raw_conservative_hardneg": CandidateConfig(
        name="raw_conservative_hardneg",
        feature_mode="hybrid",
        text_transform="raw",
        positive_weight_mode="default",
        positive_weight_multiplier=0.35,
        hard_negative_weight=4.0,
        hard_negative_fraction=0.65,
    ),
    "raw_upper_plateau": CandidateConfig(
        name="raw_upper_plateau",
        feature_mode="hybrid",
        text_transform="raw",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_plateau_position=0.95,
    ),
    "raw_strict_upper": CandidateConfig(
        name="raw_strict_upper",
        feature_mode="hybrid",
        text_transform="raw",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=3.0,
        hard_negative_fraction=0.5,
        threshold_target_fpr=0.03,
        threshold_plateau_position=0.95,
    ),
    "raw_recall_boost": CandidateConfig(
        name="raw_recall_boost",
        feature_mode="hybrid",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_plateau_position=0.25,
    ),
    "raw_precision_push": CandidateConfig(
        name="raw_precision_push",
        feature_mode="hybrid",
        text_transform="raw",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=4.0,
        hard_negative_fraction=0.65,
        threshold_target_precision=0.93,
        threshold_plateau_position=0.95,
    ),
    "raw_fnr_push": CandidateConfig(
        name="raw_fnr_push",
        feature_mode="hybrid",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_target_fnr=0.05,
        threshold_plateau_position=0.0,
    ),
    "behavior_v1_default": CandidateConfig(
        name="behavior_v1_default",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v1",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
    ),
    "behavior_v1_upper": CandidateConfig(
        name="behavior_v1_upper",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v1",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_plateau_position=0.95,
    ),
    "behavior_v1_relaxed": CandidateConfig(
        name="behavior_v1_relaxed",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v1",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_plateau_position=0.25,
    ),
    "behavior_v2_default": CandidateConfig(
        name="behavior_v2_default",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v2",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
    ),
    "behavior_v2_upper": CandidateConfig(
        name="behavior_v2_upper",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v2",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_plateau_position=0.95,
    ),
    "behavior_v2_relaxed": CandidateConfig(
        name="behavior_v2_relaxed",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v2",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_plateau_position=0.25,
    ),
    "behavior_v2_fnr_push": CandidateConfig(
        name="behavior_v2_fnr_push",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v2",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_target_fnr=0.05,
        threshold_plateau_position=0.0,
    ),
    "structured_default": CandidateConfig(
        name="structured_default",
        feature_mode="structured",
        text_transform="raw",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
    ),
    "structured_relaxed": CandidateConfig(
        name="structured_relaxed",
        feature_mode="structured",
        text_transform="raw",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_plateau_position=0.25,
    ),
    "structured_fnr_push": CandidateConfig(
        name="structured_fnr_push",
        feature_mode="structured",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.2,
        hard_negative_weight=3.0,
        hard_negative_fraction=0.5,
        threshold_target_fnr=0.05,
        threshold_plateau_position=0.0,
    ),
    "raw_source_gate": CandidateConfig(
        name="raw_source_gate",
        feature_mode="hybrid",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.2,
        hard_negative_weight=4.0,
        hard_negative_fraction=0.65,
        negative_source_weights=(
            ("github_go_benign_candidates", 3.0),
            ("pypi_popular_official", 3.0),
            ("pypi_official_registry", 2.0),
            ("crossvul", 2.0),
            ("zenodo_13870382", 2.0),
            ("github_rust_benign_candidates", 4.0),
            ("the_stack_smol_rust_expansion", 4.0),
        ),
        positive_source_weights=(
            ("github_go_malicious_candidates", 1.5),
            ("pypi_malregistry_ase2023", 1.25),
            ("github_ruby_malicious_candidates", 2.0),
            ("github_rust_malicious_candidates", 2.0),
        ),
        threshold_target_fpr=0.08,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.08,
        threshold_plateau_position=0.0,
    ),
    "behavior_v2_source_gate": CandidateConfig(
        name="behavior_v2_source_gate",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v2",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.4,
        hard_negative_weight=4.0,
        hard_negative_fraction=0.65,
        negative_source_weights=(
            ("github_go_benign_candidates", 3.0),
            ("pypi_popular_official", 3.0),
            ("pypi_official_registry", 2.0),
            ("crossvul", 2.0),
            ("zenodo_13870382", 2.0),
            ("github_rust_benign_candidates", 4.0),
            ("the_stack_smol_rust_expansion", 4.0),
        ),
        positive_source_weights=(
            ("github_go_malicious_candidates", 1.5),
            ("pypi_malregistry_ase2023", 1.25),
            ("github_ruby_malicious_candidates", 2.0),
            ("github_rust_malicious_candidates", 2.0),
        ),
        threshold_target_fpr=0.08,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.08,
        threshold_plateau_position=0.0,
    ),
    "structured_source_gate": CandidateConfig(
        name="structured_source_gate",
        feature_mode="structured",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=5.0,
        hard_negative_fraction=0.75,
        negative_source_weights=(
            ("github_go_benign_candidates", 3.0),
            ("pypi_popular_official", 3.0),
            ("pypi_official_registry", 2.0),
            ("crossvul", 2.0),
            ("zenodo_13870382", 2.0),
            ("github_rust_benign_candidates", 5.0),
            ("the_stack_smol_rust_expansion", 5.0),
        ),
        positive_source_weights=(
            ("github_go_malicious_candidates", 1.5),
            ("pypi_malregistry_ase2023", 1.25),
            ("github_ruby_malicious_candidates", 2.0),
            ("github_rust_malicious_candidates", 2.0),
        ),
        threshold_target_fpr=0.08,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.08,
        threshold_plateau_position=0.0,
    ),
    "raw_uncalibrated_gate": CandidateConfig(
        name="raw_uncalibrated_gate",
        feature_mode="hybrid",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.2,
        hard_negative_weight=3.0,
        hard_negative_fraction=0.5,
        calibration_mode="none",
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.0,
    ),
    "behavior_v2_uncalibrated_gate": CandidateConfig(
        name="behavior_v2_uncalibrated_gate",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v2",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.2,
        hard_negative_weight=3.0,
        hard_negative_fraction=0.5,
        calibration_mode="none",
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.0,
    ),
    "structured_uncalibrated_gate": CandidateConfig(
        name="structured_uncalibrated_gate",
        feature_mode="structured",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.2,
        hard_negative_weight=3.0,
        hard_negative_fraction=0.5,
        calibration_mode="none",
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.0,
    ),
    "behavior_v3_fnr_push": CandidateConfig(
        name="behavior_v3_fnr_push",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_target_fnr=0.05,
        threshold_plateau_position=0.0,
    ),
    "behavior_v3_uncalibrated_gate": CandidateConfig(
        name="behavior_v3_uncalibrated_gate",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.2,
        hard_negative_weight=3.0,
        hard_negative_fraction=0.5,
        calibration_mode="none",
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.0,
    ),
    "behavior_v3_uncalibrated_recall": CandidateConfig(
        name="behavior_v3_uncalibrated_recall",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.6,
        hard_negative_weight=4.0,
        hard_negative_fraction=0.60,
        calibration_mode="none",
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.05,
        threshold_plateau_position=0.0,
    ),
    "behavior_v3_uncalibrated_recall_high": CandidateConfig(
        name="behavior_v3_uncalibrated_recall_high",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="balanced",
        positive_weight_multiplier=2.0,
        hard_negative_weight=5.0,
        hard_negative_fraction=0.70,
        calibration_mode="none",
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.05,
        threshold_plateau_position=0.0,
    ),
    "behavior_v3_default": CandidateConfig(
        name="behavior_v3_default",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="default",
        positive_weight_multiplier=1.0,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
    ),
    "behavior_v3_precision_gate": CandidateConfig(
        name="behavior_v3_precision_gate",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.0,
        hard_negative_weight=4.0,
        hard_negative_fraction=0.65,
        negative_source_weights=(
            ("github_go_benign_candidates", 3.0),
            ("crossvul", 2.0),
            ("zenodo_13870382", 2.0),
            ("github_rust_benign_candidates", 4.0),
            ("the_stack_smol_rust_expansion", 4.0),
        ),
        threshold_target_fpr=0.05,
        threshold_target_precision=0.92,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "behavior_v2_fnr_upper": CandidateConfig(
        name="behavior_v2_fnr_upper",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v2",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_target_fnr=0.05,
        threshold_plateau_position=0.95,
    ),
    "behavior_v2_gate_upper": CandidateConfig(
        name="behavior_v2_gate_upper",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v2",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "behavior_v3_fnr_upper": CandidateConfig(
        name="behavior_v3_fnr_upper",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_target_fnr=0.05,
        threshold_plateau_position=0.95,
    ),
    "behavior_v3_gate_upper": CandidateConfig(
        name="behavior_v3_gate_upper",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=2.0,
        hard_negative_fraction=1 / 3,
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "raw_crossvul_hardneg": CandidateConfig(
        name="raw_crossvul_hardneg",
        feature_mode="hybrid",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=10.0,
        hard_negative_fraction=0.80,
        negative_source_weights=(("crossvul", 6.0),),
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "raw_crossvul_strict": CandidateConfig(
        name="raw_crossvul_strict",
        feature_mode="hybrid",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=14.0,
        hard_negative_fraction=0.90,
        negative_source_weights=(("crossvul", 12.0),),
        threshold_target_fpr=0.05,
        threshold_target_precision=0.93,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "behavior_v3_crossvul_hardneg": CandidateConfig(
        name="behavior_v3_crossvul_hardneg",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=10.0,
        hard_negative_fraction=0.80,
        negative_source_weights=(("crossvul", 6.0),),
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "behavior_v3_crossvul_strict": CandidateConfig(
        name="behavior_v3_crossvul_strict",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=14.0,
        hard_negative_fraction=0.90,
        negative_source_weights=(("crossvul", 12.0),),
        threshold_target_fpr=0.05,
        threshold_target_precision=0.93,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "structured_crossvul_hardneg": CandidateConfig(
        name="structured_crossvul_hardneg",
        feature_mode="structured",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=10.0,
        hard_negative_fraction=0.80,
        negative_source_weights=(("crossvul", 6.0),),
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "behavior_v3_exfil_family_weight": CandidateConfig(
        name="behavior_v3_exfil_family_weight",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=10.0,
        hard_negative_fraction=0.80,
        negative_source_weights=(("crossvul", 6.0),),
        positive_family_weights=(
            ("github_malicious_candidate:go:emp3r0r", 8.0),
        ),
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "structured_exfil_family_weight": CandidateConfig(
        name="structured_exfil_family_weight",
        feature_mode="structured",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=10.0,
        hard_negative_fraction=0.80,
        negative_source_weights=(("crossvul", 6.0),),
        positive_family_weights=(
            ("github_malicious_candidate:go:emp3r0r", 8.0),
        ),
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "behavior_v3_no_coarse_groups": CandidateConfig(
        name="behavior_v3_no_coarse_groups",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=6.0,
        hard_negative_fraction=0.65,
        negative_source_weights=(("crossvul", 4.0),),
        excluded_structured_features=(
            "file_local_behavior_group_count",
            "file_local_multi_behavior_group_proxy",
        ),
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "structured_no_coarse_groups": CandidateConfig(
        name="structured_no_coarse_groups",
        feature_mode="structured",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=6.0,
        hard_negative_fraction=0.65,
        negative_source_weights=(("crossvul", 4.0),),
        excluded_structured_features=(
            "file_local_behavior_group_count",
            "file_local_multi_behavior_group_proxy",
        ),
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "behavior_v3_count_only_groups": CandidateConfig(
        name="behavior_v3_count_only_groups",
        feature_mode="hybrid",
        text_transform="behavior_tokens_v3",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=6.0,
        hard_negative_fraction=0.65,
        negative_source_weights=(("crossvul", 4.0),),
        excluded_structured_features=(
            "file_local_multi_behavior_group_proxy",
        ),
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
    "structured_count_only_groups": CandidateConfig(
        name="structured_count_only_groups",
        feature_mode="structured",
        text_transform="raw",
        positive_weight_mode="balanced",
        positive_weight_multiplier=1.5,
        hard_negative_weight=6.0,
        hard_negative_fraction=0.65,
        negative_source_weights=(("crossvul", 4.0),),
        excluded_structured_features=(
            "file_local_multi_behavior_group_proxy",
        ),
        threshold_target_fpr=0.10,
        threshold_target_precision=0.90,
        threshold_target_fnr=0.10,
        threshold_plateau_position=0.95,
    ),
}


def _metric_deficit(metrics: dict[str, Any]) -> float:
    return (
        max(0.0, QUALITY_GATE["min_precision"] - float(metrics.get("precision", 0.0)))
        + max(0.0, float(metrics.get("false_positive_rate", 1.0)) - QUALITY_GATE["max_false_positive_rate"])
        + max(0.0, float(metrics.get("false_negative_rate", 1.0)) - QUALITY_GATE["max_false_negative_rate"])
    )


def _run_one(language: str, config: CandidateConfig, run_id: str) -> dict[str, Any]:
    route = ROUTES / f"{language}.jsonl"
    output = MODEL_ROOT / run_id / f"{language}_{config.name}"
    log = LOG_ROOT / run_id / f"{language}_{config.name}.stdout.log"
    err = LOG_ROOT / run_id / f"{language}_{config.name}.stderr.log"
    metrics_path = output.with_suffix(".json")
    model_path = output.with_suffix(".joblib")

    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(TRAINER),
        "--dataset",
        str(route),
        "--task",
        "malicious_intent",
        "--language",
        language,
        "--feature-mode",
        config.feature_mode,
        "--text-transform",
        config.text_transform,
        "--positive-weight-mode",
        config.positive_weight_mode,
        "--positive-weight-multiplier",
        str(config.positive_weight_multiplier),
        "--hard-negative-weight",
        str(config.hard_negative_weight),
        "--hard-negative-fraction",
        str(config.hard_negative_fraction),
        "--calibration-mode",
        config.calibration_mode,
        "--threshold-plateau-position",
        str(config.threshold_plateau_position),
        "--output",
        str(output),
    ]
    if config.threshold_target_fpr is not None:
        cmd.extend(["--threshold-target-fpr", str(config.threshold_target_fpr)])
    if config.threshold_target_precision is not None:
        cmd.extend(["--threshold-target-precision", str(config.threshold_target_precision)])
    if config.threshold_target_fnr is not None:
        cmd.extend(["--threshold-target-fnr", str(config.threshold_target_fnr)])
    for source, weight in config.negative_source_weights:
        cmd.extend(["--negative-source-weight", f"{source}={weight}"])
    for family, weight in config.negative_family_weights:
        cmd.extend(["--negative-family-weight", f"{family}={weight}"])
    for source, weight in config.positive_source_weights:
        cmd.extend(["--positive-source-weight", f"{source}={weight}"])
    for family, weight in config.positive_family_weights:
        cmd.extend(["--positive-family-weight", f"{family}={weight}"])
    for feature_name in config.excluded_structured_features:
        cmd.extend(["--exclude-structured-feature", feature_name])
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")

    with log.open("w", encoding="utf-8") as stdout, err.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )

    result: dict[str, Any] = {
        "language": language,
        "config": config.name,
        "returncode": completed.returncode,
        "model": str(model_path),
        "metrics": str(metrics_path),
        "stdout_log": str(log),
        "stderr_log": str(err),
    }
    if completed.returncode == 0 and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        test = metrics.get("test", {})
        validation = metrics.get("selected", {}).get("validation", {})
        split_counts = metrics.get("split_label_counts", {})
        validation_passed = bool(validation.get("quality_gate_passed"))
        test_passed = bool(test.get("quality_gate_passed"))
        result.update({
            "accuracy": test.get("accuracy"),
            "precision": test.get("precision"),
            "recall": test.get("recall"),
            "false_positive_rate": test.get("false_positive_rate"),
            "false_negative_rate": test.get("false_negative_rate"),
            "f1": test.get("f1"),
            "validation_accuracy": validation.get("accuracy"),
            "validation_precision": validation.get("precision"),
            "validation_recall": validation.get("recall"),
            "validation_false_positive_rate": validation.get(
                "false_positive_rate"
            ),
            "validation_false_negative_rate": validation.get(
                "false_negative_rate"
            ),
            "validation_f1": validation.get("f1"),
            "validation_quality_gate_passed": validation_passed,
            "test_quality_gate_passed": test_passed,
            # A candidate is eligible only if threshold selection passes on
            # validation and the frozen threshold also passes on test.
            "quality_gate_passed": validation_passed and test_passed,
            "metric_deficit": (
                _metric_deficit(validation) + _metric_deficit(test)
            ),
            "split_label_counts": split_counts,
            "low_positive_test_support": (
                int(split_counts.get("test", {}).get("malicious", 0)) < 30
            ),
        })
    return result


def _best_by_language(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for result in results:
        if result.get("returncode") != 0:
            continue
        language = str(result["language"])
        current = best.get(language)
        rank = (
            bool(result.get("quality_gate_passed")),
            -float(result.get("metric_deficit", 999.0)),
            float(result.get("f1") or 0.0),
            float(result.get("precision") or 0.0),
            float(result.get("recall") or 0.0),
        )
        if current is None:
            best[language] = {**result, "_rank": rank}
            continue
        if rank > tuple(current["_rank"]):
            best[language] = {**result, "_rank": rank}
    for result in best.values():
        result.pop("_rank", None)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="raw_default")
    parser.add_argument("--languages", default=",".join(LANGUAGES))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    selected_configs = [item.strip() for item in args.configs.split(",") if item.strip()]
    selected_languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    unknown_configs = [name for name in selected_configs if name not in CONFIGS]
    if unknown_configs:
        raise SystemExit(f"unknown config(s): {', '.join(unknown_configs)}")

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = [
        (language, CONFIGS[config_name], run_id)
        for language in selected_languages
        for config_name in selected_configs
    ]
    print(f"run_id={run_id} jobs={len(jobs)} workers={args.workers}")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_run_one, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "PASS" if result.get("quality_gate_passed") else "FAIL"
            if result.get("returncode") != 0:
                status = f"ERROR({result['returncode']})"
            print(
                f"{status} {result['language']} {result['config']} "
                f"VP={result.get('validation_precision')} "
                f"VFPR={result.get('validation_false_positive_rate')} "
                f"VFNR={result.get('validation_false_negative_rate')} "
                f"P={result.get('precision')} FPR={result.get('false_positive_rate')} "
                f"FNR={result.get('false_negative_rate')}"
            )

    summary = {
        "run_id": run_id,
        "configs": selected_configs,
        "languages": selected_languages,
        "quality_gate": QUALITY_GATE,
        "results": sorted(results, key=lambda item: (item["language"], item["config"])),
        "best_by_language": _best_by_language(results),
    }
    summary_path = SUMMARY_ROOT / f"{TRAIN_TAG}_all_language_training_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
