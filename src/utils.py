"""Shared utilities for persistence, determinism, time guarding, and OOM-safe inference."""

import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import psutil
import torch
import yaml


def get_logger(name: str = "AITextDetect") -> logging.Logger:
    """Create and return a formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = get_logger("Utils")


def set_seed(seed: int = 42) -> int:
    """Set seeds across Python, NumPy, and PyTorch for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")
    return seed


def get_device(force_cpu: bool = False) -> torch.device:
    """Auto-detect available compute device (CUDA, Apple MPS, or CPU)."""
    if force_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        logger.info(f"Active device: CUDA ({props.name}, {vram_gb:.2f} GB VRAM)")
    else:
        logger.info(f"Active device: {device.type.upper()}")
    return device


def clean_memory(*objects: Any) -> None:
    """Explicitly delete variables, invoke Python GC, and release GPU memory cache."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_config(
    config_path: str = "configs/config.yaml",
    tier_override: Optional[str] = None,
    drive_dir_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Load configuration YAML, merge active tier parameters, and ensure persistence directories exist."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Active tier resolution
    active_tier = tier_override or config.get("tier", "smoke")
    if active_tier not in config.get("tiers", {}):
        valid = list(config.get("tiers", {}).keys())
        raise ValueError(f"Unknown tier '{active_tier}'. Available tiers: {valid}")
    config["active_tier_name"] = active_tier
    config["active_tier"] = config["tiers"][active_tier]

    # Resolve paths (environment variable DRIVE_DIR takes precedence if present)
    drive_dir = (
        drive_dir_override
        or os.environ.get("DRIVE_DIR")
        or config.get("paths", {}).get("drive_dir", "./outputs")
    )
    base_path = Path(drive_dir).resolve()
    config["paths"]["drive_dir"] = str(base_path)
    config["paths"]["cache_dir"] = str(base_path / "cache")
    config["paths"]["data_dir"] = str(base_path / "data")
    config["paths"]["checkpoints_dir"] = str(base_path / "checkpoints")
    config["paths"]["results_dir"] = str(base_path / "results")

    # Create directories immediately
    for p in config["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)

    return config


class TimeGuard:
    """Monitors session duration to stop gracefully before Colab/cloud cutoff limits."""

    def __init__(
        self,
        max_hours: float = 11.0,
        heartbeat_interval_sec: float = 120.0,
        custom_logger: Optional[logging.Logger] = None,
    ):
        self.max_seconds = max_hours * 3600.0
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.start_time = time.time()
        self.last_heartbeat_time = self.start_time
        self.logger = custom_logger or logger

    def check(self) -> Tuple[bool, float]:
        """Check if elapsed session time has exceeded safe threshold.

        Returns (should_stop, elapsed_hours).
        """
        elapsed = time.time() - self.start_time
        elapsed_hours = elapsed / 3600.0
        should_stop = elapsed >= self.max_seconds

        if should_stop:
            self.logger.warning(
                f"[TIME GUARD TRIGGERED] Elapsed {elapsed_hours:.2f} h exceeds limit of "
                f"{self.max_seconds / 3600.0:.2f} h. Stopping gracefully to persist state."
            )
        return should_stop, elapsed_hours

    def heartbeat(self, step_info: str = "") -> None:
        """Log a periodic heartbeat with resource usage (RAM and VRAM)."""
        now = time.time()
        if (now - self.last_heartbeat_time) >= self.heartbeat_interval_sec or step_info:
            elapsed = now - self.start_time
            remaining = max(0.0, self.max_seconds - elapsed)
            ram = psutil.virtual_memory()
            ram_info = f"RAM: {ram.used / (1024**3):.1f}/{ram.total / (1024**3):.1f} GB ({ram.percent}%)"

            gpu_info = "GPU: N/A"
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                gpu_info = f"VRAM Alloc/Res: {allocated:.2f}/{reserved:.2f} GB"

            info_str = f" | {step_info}" if step_info else ""
            self.logger.info(
                f"[HEARTBEAT] Elapsed: {elapsed/3600.0:.2f}h | Remaining: {remaining/3600.0:.2f}h | "
                f"{ram_info} | {gpu_info}{info_str}"
            )
            self.last_heartbeat_time = now


class ChunkedWriter:
    """Appends records to a JSONL file in atomic chunks with resumability tracking."""

    def __init__(self, output_path: str, id_key: str = "id"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.id_key = id_key
        self.completed_ids: Set[str] = self._load_completed_ids()

    def _load_completed_ids(self) -> Set[str]:
        """Scan existing JSONL file and return set of completed IDs, skipping malformed trailing lines."""
        ids: Set[str] = set()
        if not self.output_path.exists():
            return ids

        valid_lines: List[str] = []
        with open(self.output_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                    rec_id = str(record[self.id_key])
                    ids.add(rec_id)
                    valid_lines.append(stripped + "\n")
                except Exception as err:
                    logger.warning(
                        f"Skipping malformed record on line {line_no} in {self.output_path}: {err}"
                    )

        # Ensure file contains only intact records
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.writelines(valid_lines)

        logger.info(f"Loaded {len(ids)} already completed records from {self.output_path}")
        return ids

    def get_completed_ids(self) -> Set[str]:
        """Return the set of completed sample IDs."""
        return set(self.completed_ids)

    def write_chunk(self, records: List[Dict[str, Any]]) -> int:
        """Write a batch of records, filtering out duplicates, with atomic flush and fsync."""
        new_records = [r for r in records if str(r.get(self.id_key)) not in self.completed_ids]
        if not new_records:
            return 0

        with open(self.output_path, "a", encoding="utf-8") as f:
            for r in new_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                self.completed_ids.add(str(r[self.id_key]))
            f.flush()
            os.fsync(f.fileno())

        return len(new_records)


def safe_batch_infer(
    inference_fn: Callable[[List[Any], torch.device], List[Any]],
    items: List[Any],
    initial_batch_size: int = 16,
    device: Optional[torch.device] = None,
    time_guard: Optional[TimeGuard] = None,
    chunk_writer: Optional[ChunkedWriter] = None,
    id_extractor: Optional[Callable[[Any], str]] = None,
) -> List[Any]:
    """Execute batched inference with automatic OOM recovery (batch halving down to 1, then CPU fallback).

    Args:
        inference_fn: Callable accepting (batch_items, device) and returning output list of same length.
        items: List of input data items to process.
        initial_batch_size: Starting batch size.
        device: Target torch device (defaults to auto-detected device).
        time_guard: Optional TimeGuard instance to enforce runtime cutoff.
        chunk_writer: Optional ChunkedWriter for incremental output persistence.
        id_extractor: Optional function extracting item ID for tracking and resumption.
    """
    if device is None:
        device = get_device()

    # Skip already completed items if writer and id extractor are provided
    if chunk_writer is not None and id_extractor is not None:
        done_ids = chunk_writer.get_completed_ids()
        remaining_items = [it for it in items if id_extractor(it) not in done_ids]
        logger.info(
            f"Resuming inference: {len(done_ids)} items skipped, {len(remaining_items)} remaining."
        )
    else:
        remaining_items = list(items)

    results: List[Any] = []
    current_batch_size = max(1, initial_batch_size)
    current_device = device
    idx = 0

    while idx < len(remaining_items):
        if time_guard is not None:
            should_stop, elapsed = time_guard.check()
            if should_stop:
                logger.warning(f"TimeGuard triggered at item {idx}/{len(remaining_items)}. Exiting loop.")
                break

        batch_slice = remaining_items[idx : idx + current_batch_size]
        try:
            batch_output = inference_fn(batch_slice, current_device)
            results.extend(batch_output)

            # Incremental persistence if writer is configured
            if chunk_writer is not None:
                # Expecting output to be dicts or map to records
                if all(isinstance(out, dict) for out in batch_output):
                    chunk_writer.write_chunk(batch_output)

            idx += len(batch_slice)

            if time_guard is not None:
                time_guard.heartbeat(f"Processed {idx}/{len(remaining_items)} items")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            err_msg = str(err).lower()
            is_oom = isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in err_msg or "cuda oom" in err_msg
            if not is_oom:
                raise err

            logger.warning(
                f"[OOM DETECTED] GPU OOM encountered on device {current_device} with batch size {current_batch_size}."
            )
            clean_memory()

            if current_device.type == "cuda" and current_batch_size > 1:
                current_batch_size = max(1, current_batch_size // 2)
                logger.warning(f"Halving batch size to {current_batch_size} and retrying on {current_device}.")
            elif current_device.type == "cuda" and current_batch_size == 1:
                logger.warning(
                    "GPU OOM persisted even at batch size 1! Falling back to CPU execution with warning."
                )
                current_device = torch.device("cpu")
                current_batch_size = 1
            else:
                # Already on CPU and failed
                raise RuntimeError(f"Unrecoverable OOM error on CPU: {err}") from err

    return results
