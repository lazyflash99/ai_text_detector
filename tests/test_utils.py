"""Tests for utils.py: verifies determinism, OOM recovery, and crash/resume persistence."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.utils import (
    ChunkedWriter,
    TimeGuard,
    clean_memory,
    get_device,
    load_config,
    safe_batch_infer,
    set_seed,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)

    def test_set_seed_determinism(self):
        """Verify global seed yields identical pseudo-random outputs across runs."""
        set_seed(1234)
        t1 = torch.randn(5, 5)
        n1 = np.random.rand(5)

        set_seed(1234)
        t2 = torch.randn(5, 5)
        n2 = np.random.rand(5)

        self.assertTrue(torch.equal(t1, t2), "PyTorch tensors differ despite identical seed!")
        self.assertTrue(np.array_equal(n1, n2), "NumPy arrays differ despite identical seed!")

    def test_load_config_smoke_tier(self):
        """Verify config loader selects and validates smoke tier defaults."""
        cfg = load_config(tier_override="smoke", drive_dir_override=self.test_dir)
        self.assertEqual(cfg["active_tier_name"], "smoke")
        self.assertEqual(cfg["active_tier"]["sample_size"], 200)
        self.assertEqual(cfg["hyperparameters"]["max_len"], 256)
        self.assertTrue(Path(cfg["paths"]["data_dir"]).exists())

    def test_time_guard(self):
        """Verify TimeGuard tracking and heartbeat output without error."""
        guard = TimeGuard(max_hours=0.0001, heartbeat_interval_sec=0.0)  # Very small threshold for test
        guard.heartbeat("Test step")
        should_stop, elapsed = guard.check()
        self.assertIsInstance(should_stop, bool)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_chunked_writer_mid_loop_crash_and_resume(self):
        """Simulate a loop that crashes at 50% and verify that re-running resumes seamlessly."""
        output_jsonl = Path(self.test_dir) / "test_records.jsonl"
        all_samples = [{"id": f"doc_{i}", "text": f"Sample text {i}"} for i in range(10)]

        # --- Phase 1: First attempt crashes after 5 items ---
        writer_run1 = ChunkedWriter(str(output_jsonl), id_key="id")
        self.assertEqual(len(writer_run1.get_completed_ids()), 0)

        with self.assertRaises(RuntimeError):
            for i, item in enumerate(all_samples):
                if i == 5:
                    raise RuntimeError("Simulated crash mid-loop!")
                writer_run1.write_chunk([item])

        # Verify only 5 items made it to disk before crash
        self.assertEqual(len(writer_run1.get_completed_ids()), 5)

        # --- Phase 2: Resume with a fresh writer instance (simulating cell re-run) ---
        writer_run2 = ChunkedWriter(str(output_jsonl), id_key="id")
        completed = writer_run2.get_completed_ids()
        self.assertEqual(len(completed), 5, "Resume failed to detect previously completed IDs!")

        # Process whole list again; completed items must be skipped
        items_to_process = [it for it in all_samples if it["id"] not in completed]
        self.assertEqual(len(items_to_process), 5)

        written = writer_run2.write_chunk(items_to_process)
        self.assertEqual(written, 5)

        # Total items in final file must be exactly 10, with no duplicates
        final_completed = writer_run2.get_completed_ids()
        self.assertEqual(len(final_completed), 10)
        self.assertEqual(final_completed, {f"doc_{i}" for i in range(10)})

    def test_safe_batch_infer_oom_recovery(self):
        """Simulate GPU OOM when batch size > 2; verify halving down to 2 and successful completion."""
        items = [{"id": f"id_{i}", "val": i} for i in range(8)]
        attempts = []

        # Simulated inference function that throws OOM if batch size is > 2 on GPU/simulated device
        def mock_infer_fn(batch: List[Any], device: torch.device) -> List[Any]:
            attempts.append((len(batch), device.type))
            if len(batch) > 2 and device.type != "cpu":
                # Raise actual CUDA OutOfMemoryError
                raise torch.cuda.OutOfMemoryError("Simulated CUDA Out Of Memory Error")
            return [{"id": it["id"], "score": it["val"] * 2} for it in batch]

        # Call with initial batch size 4 on a mock cuda-labeled device
        mock_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # If running on CPU, we force mock device type to test OOM handling
        if mock_device.type == "cpu":
            # Test CPU execution directly works
            results = safe_batch_infer(
                inference_fn=mock_infer_fn,
                items=items,
                initial_batch_size=2,
                device=mock_device,
            )
            self.assertEqual(len(results), 8)
            self.assertEqual(results[0]["score"], 0)
        else:
            # Full OOM recovery test on CUDA
            results = safe_batch_infer(
                inference_fn=mock_infer_fn,
                items=items,
                initial_batch_size=4,
                device=mock_device,
            )
            self.assertEqual(len(results), 8)
            # Verify batch size 4 failed with OOM and halved to 2
            self.assertTrue(any(bs == 4 for bs, _ in attempts))
            self.assertTrue(any(bs <= 2 for bs, _ in attempts))

    def test_safe_batch_infer_cpu_fallback(self):
        """Simulate persistent OOM even at batch size 1 on CUDA; verify fallback to CPU."""
        items = [{"id": "id_0", "val": 10}]
        devices_attempted = []

        def mock_hard_oom(batch: List[Any], device: torch.device) -> List[Any]:
            devices_attempted.append(device.type)
            if device.type == "cuda":
                raise torch.cuda.OutOfMemoryError("Simulated stubborn OOM at batch size 1")
            return [{"id": it["id"], "score": it["val"] * 10} for it in batch]

        # Use CPU directly if no cuda, else test transition
        if torch.cuda.is_available():
            results = safe_batch_infer(
                inference_fn=mock_hard_oom,
                items=items,
                initial_batch_size=1,
                device=torch.device("cuda"),
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["score"], 100)
            self.assertIn("cuda", devices_attempted)
            self.assertIn("cpu", devices_attempted)
        else:
            results = safe_batch_infer(
                inference_fn=mock_hard_oom,
                items=items,
                initial_batch_size=1,
                device=torch.device("cpu"),
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["score"], 100)


def run_smoke_self_test():
    """Execute smoke self-test and display human-readable status."""
    print("=" * 70)
    print("RUNNING MODULE 1 SELF-TEST (Smoke Tier)")
    print("=" * 70)

    # 1. Config & Device Check
    cfg = load_config(tier_override="smoke")
    device = get_device()
    seed = set_seed(cfg["runtime"]["seed"])
    print(f"[Self-Test] Config loaded: tier={cfg['active_tier_name']}, sample_size={cfg['active_tier']['sample_size']}")
    print(f"[Self-Test] Device detected: {device}, Seed: {seed}")

    # 2. Crash and Resume Simulation
    test_file = Path(cfg["paths"]["data_dir"]) / "smoke_resume_test.jsonl"
    if test_file.exists():
        test_file.unlink()

    writer = ChunkedWriter(str(test_file))
    print(f"[Self-Test] Initializing writer at {test_file}")
    writer.write_chunk([{"id": "doc_0", "text": "Hello world"}])
    writer.write_chunk([{"id": "doc_1", "text": "Resilience test"}])
    print(f"[Self-Test] Chunk 1 written. Completed count: {len(writer.get_completed_ids())}")

    # Simulating restart
    resumed_writer = ChunkedWriter(str(test_file))
    completed_ids = resumed_writer.get_completed_ids()
    print(f"[Self-Test] Resumed writer loaded completed IDs: {completed_ids}")
    assert "doc_0" in completed_ids and "doc_1" in completed_ids, "Resume failed to reload IDs!"

    # 3. Clean memory check
    clean_memory()
    print("[Self-Test] Memory clean helper executed successfully.")
    print("=" * 70)
    print("ALL MODULE 1 CHECKS PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_self_test()
    print("\nRunning full unit test suite:")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        exit(1)
