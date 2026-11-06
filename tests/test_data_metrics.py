# tests/test_data_metrics.py
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import create_splits, sample_and_clean_data
from src.metrics import RESULTS_COLUMNS, append_results, evaluate
from src.utils import load_config, set_seed


class TestDataAndMetrics(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)

    def test_end_to_end_data_and_metrics_smoke(self):
        """End-to-end self-test on 200 smoke samples."""
        set_seed(42)
        out_path = Path(self.test_dir)

        # 1. Sample 200 items
        df = sample_and_clean_data(target_samples=200, seed=42)
        self.assertEqual(len(df), 200)
        self.assertIn("label", df.columns)
        self.assertIn("domain", df.columns)

        # 2. Splits & Leakage Assertions
        manifest = create_splits(df, output_dir=out_path, seed=42)
        self.assertTrue((out_path / "train.parquet").exists())
        self.assertTrue((out_path / "test.parquet").exists())
        self.assertTrue((out_path / "domain_class_dist.png").exists())
        self.assertTrue((out_path / "manifest.json").exists())

        # 3. Simulate detector predictions
        test_df = pd.read_parquet(out_path / "test.parquet")
        # Simulating scores with some signal
        test_df["score"] = np.where(
            test_df["label"] == 1,
            np.random.beta(5, 2, size=len(test_df)),
            np.random.beta(2, 5, size=len(test_df)),
        )

        val_df = pd.read_parquet(out_path / "val.parquet")
        val_human_scores = np.random.beta(2, 5, size=len(val_df[val_df["label"] == 0]))

        # 4. Evaluation
        eval_df = evaluate(
            scores_df=test_df,
            group_by=["domain"],
            val_human_scores=val_human_scores,
            seed=42,
        )
        self.assertFalse(eval_df.empty)
        metrics_found = set(eval_df["metric"].unique())
        self.assertTrue({"auroc", "f1", "accuracy", "tpr_at_1fpr", "tpr_at_5fpr"}.issubset(metrics_found))

        # 5. Append Results
        results_csv = out_path / "results.csv"
        eval_df["experiment"] = "smoke_test"
        eval_df["detector"] = "test_detector"
        eval_df["train_cond"] = "in_domain"
        eval_df["test_cond"] = "in_domain"
        eval_df["generator"] = "all"
        eval_df["attack"] = "none"
        eval_df["seed"] = 42

        append_results(results_csv, eval_df)
        self.assertTrue(results_csv.exists())

        saved_df = pd.read_csv(results_csv)
        self.assertEqual(list(saved_df.columns), RESULTS_COLUMNS)
        self.assertGreater(len(saved_df), 0)


def run_smoke_self_test():
    print("=" * 70)
    print("RUNNING MODULE 2 (DATA & METRICS) SMOKE SELF-TEST (200 SAMPLES)")
    print("=" * 70)
    cfg = load_config(tier_override="smoke")
    data_dir = Path(cfg["paths"]["data_dir"])

    df = sample_and_clean_data(target_samples=200, seed=cfg["runtime"]["seed"])
    print(f"[Self-Test] Sampled cleaned dataset: {len(df)} rows across domains: {df['domain'].unique().tolist()}")

    manifest = create_splits(df, output_dir=data_dir, seed=cfg["runtime"]["seed"])
    print(f"[Self-Test] Splits created: train={manifest['split_counts']['train']}, test={manifest['split_counts']['test']}")

    # Mock predictions on test split
    test_df = pd.read_parquet(data_dir / "test.parquet")
    test_df["score"] = np.where(
        test_df["label"] == 1,
        np.random.beta(4, 2, size=len(test_df)),
        np.random.beta(2, 4, size=len(test_df)),
    )
    val_human_scores = np.random.beta(2, 4, size=30)

    eval_df = evaluate(test_df, group_by=["domain"], val_human_scores=val_human_scores)
    print(f"[Self-Test] Evaluated metrics:\n{eval_df[['domain', 'metric', 'value', 'ci_low', 'ci_high']].head(6)}")

    results_csv = Path(cfg["paths"]["results_dir"]) / "results.csv"
    eval_df["experiment"] = "smoke_eval"
    eval_df["detector"] = "mock_model"
    eval_df["train_cond"] = "in_domain"
    eval_df["test_cond"] = "in_domain"
    eval_df["generator"] = "various"
    eval_df["attack"] = "none"
    eval_df["seed"] = 42
    append_results(results_csv, eval_df)

    print(f"[Self-Test] Appended metrics to: {results_csv}")
    print("=" * 70)
    print("MODULE 2 SELF-TEST COMPLETED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_self_test()
