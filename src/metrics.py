# src/metrics.py
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.utils import get_logger

logger = get_logger("Metrics")

RESULTS_COLUMNS = [
    "experiment",
    "detector",
    "train_cond",
    "test_cond",
    "domain",
    "generator",
    "attack",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n",
    "seed",
]


def compute_tpr_at_fpr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    val_human_scores: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """Compute True Positive Rate at a fixed False Positive Rate threshold chosen strictly on human validation data."""
    if len(val_human_scores) == 0:
        logger.warning("Empty validation human scores; cannot compute threshold-calibrated TPR.")
        return float(np.nan)

    # Threshold where at most target_fpr of human validation samples exceed it
    threshold = float(np.quantile(val_human_scores, 1.0 - target_fpr))

    ai_mask = y_true == 1
    if not np.any(ai_mask):
        return float(np.nan)

    tpr = float(np.mean(y_scores[ai_mask] >= threshold))
    return tpr


def bootstrap_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% confidence intervals via vectorized bootstrap resampling."""
    n = len(y_true)
    if n < 5:
        return float(np.nan), float(np.nan)

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))
    scores: List[float] = []

    for idxs in boot_indices:
        y_b = y_true[idxs]
        s_b = y_scores[idxs]
        # Skip resamples with only one class
        if len(np.unique(y_b)) < 2:
            continue
        try:
            val = metric_fn(y_b, s_b)
            if not np.isnan(val):
                scores.append(val)
        except Exception:
            continue

    if len(scores) < 10:
        return float(np.nan), float(np.nan)

    low = float(np.percentile(scores, 100 * (alpha / 2)))
    high = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return low, high


def compute_binary_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    val_human_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute AUROC, F1, Accuracy, TPR@1% FPR, TPR@5% FPR with bootstrap 95% CIs."""
    results: Dict[str, Dict[str, float]] = {}
    classes = np.unique(y_true)
    n_samples = len(y_true)

    # Predictions at standard decision boundary
    y_pred = (y_scores >= threshold).astype(int)

    # 1. Accuracy
    acc = float(accuracy_score(y_true, y_pred))
    acc_low, acc_high = bootstrap_ci(
        y_true,
        y_scores,
        lambda yt, ys: accuracy_score(yt, (ys >= threshold).astype(int)),
        seed=seed,
    )
    results["accuracy"] = {"value": acc, "ci_low": acc_low, "ci_high": acc_high}

    # 2. F1
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f1_low, f1_high = bootstrap_ci(
        y_true,
        y_scores,
        lambda yt, ys: f1_score(yt, (ys >= threshold).astype(int), zero_division=0),
        seed=seed,
    )
    results["f1"] = {"value": f1, "ci_low": f1_low, "ci_high": f1_high}

    # 3. AUROC (requires both classes)
    if len(classes) >= 2:
        auroc = float(roc_auc_score(y_true, y_scores))
        auroc_low, auroc_high = bootstrap_ci(
            y_true, y_scores, lambda yt, ys: roc_auc_score(yt, ys), seed=seed
        )
        results["auroc"] = {"value": auroc, "ci_low": auroc_low, "ci_high": auroc_high}
    else:
        logger.warning("Single-class slice encountered; skipping AUROC calculation.")
        results["auroc"] = {"value": float(np.nan), "ci_low": float(np.nan), "ci_high": float(np.nan)}

    # 4. Calibrated TPR @ 1% and 5% FPR
    h_scores = (
        val_human_scores
        if val_human_scores is not None and len(val_human_scores) > 0
        else y_scores[y_true == 0]
    )

    for fpr in [0.01, 0.05]:
        metric_name = f"tpr_at_{int(fpr*100)}fpr"
        tpr_val = compute_tpr_at_fpr(y_true, y_scores, h_scores, target_fpr=fpr)
        tpr_low, tpr_high = bootstrap_ci(
            y_true,
            y_scores,
            lambda yt, ys: compute_tpr_at_fpr(yt, ys, h_scores, target_fpr=fpr),
            seed=seed,
        )
        results[metric_name] = {"value": tpr_val, "ci_low": tpr_low, "ci_high": tpr_high}

    return results


def evaluate(
    scores_df: pd.DataFrame,
    group_by: Optional[List[str]] = None,
    val_human_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate predictions, grouping slices and returning a tidy DataFrame."""
    if scores_df.empty:
        return pd.DataFrame()

    required_cols = {"label", "score"}
    missing = required_cols - set(scores_df.columns)
    if missing:
        raise ValueError(f"scores_df missing required columns: {missing}")

    records: List[Dict[str, Any]] = []

    if group_by:
        # Group evaluation
        for group_keys, sub_df in scores_df.groupby(group_by):
            if isinstance(group_keys, tuple):
                group_dict = dict(zip(group_by, group_keys))
            else:
                group_dict = {group_by[0]: group_keys}

            y_t = sub_df["label"].to_numpy().astype(int)
            y_s = sub_df["score"].to_numpy().astype(float)

            # Skip single-class groups
            if len(np.unique(y_t)) < 2:
                logger.warning(
                    f"Skipping single-class slice for group {group_dict} (n={len(sub_df)})."
                )
                continue

            metrics = compute_binary_metrics(
                y_t, y_s, val_human_scores=val_human_scores, threshold=threshold, seed=seed
            )
            for m_name, vals in metrics.items():
                row = {**group_dict, "metric": m_name, **vals, "n": len(sub_df)}
                records.append(row)
    else:
        # Global overall evaluation
        y_t = scores_df["label"].to_numpy().astype(int)
        y_s = scores_df["score"].to_numpy().astype(float)
        metrics = compute_binary_metrics(
            y_t, y_s, val_human_scores=val_human_scores, threshold=threshold, seed=seed
        )
        for m_name, vals in metrics.items():
            records.append({"metric": m_name, **vals, "n": len(scores_df)})

    return pd.DataFrame(records)


def compute_shifted_delta(
    eval_df: pd.DataFrame,
    baseline_cond: str = "in_domain",
    cond_col: str = "test_cond",
) -> pd.DataFrame:
    """Compute Delta (Shifted - Clean Baseline) across all matching metrics and groups."""
    if cond_col not in eval_df.columns or "metric" not in eval_df.columns or "value" not in eval_df.columns:
        return pd.DataFrame()

    base_df = eval_df[eval_df[cond_col] == baseline_cond]
    shift_df = eval_df[eval_df[cond_col] != baseline_cond]

    merge_keys = [c for c in eval_df.columns if c not in [cond_col, "value", "ci_low", "ci_high", "n"]]
    merged = pd.merge(shift_df, base_df, on=merge_keys, suffixes=("_shift", "_clean"))
    merged["delta_value"] = merged["value_shift"] - merged["value_clean"]

    delta_df = merged[
        merge_keys + [f"{cond_col}_shift", "metric", "delta_value", "value_clean", "value_shift"]
    ].copy()
    return delta_df


def append_results(
    results_path: Union[str, Path],
    records: Union[pd.DataFrame, List[Dict[str, Any]]],
) -> None:
    """Append structured evaluation rows to results.csv ensuring strict column alignment."""
    p = Path(results_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(records, list):
        df_new = pd.DataFrame(records)
    else:
        df_new = records.copy()

    # Fill missing expected columns with defaults
    for col in RESULTS_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = "N/A"

    df_new = df_new[RESULTS_COLUMNS]

    write_header = not p.exists() or p.stat().st_size == 0
    df_new.to_csv(p, mode="a", index=False, header=write_header, encoding="utf-8")
    logger.info(f"Appended {len(df_new)} metric records to {p}")
