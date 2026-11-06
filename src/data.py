# src/data.py
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from src.utils import clean_memory, get_logger, load_config, set_seed

logger = get_logger("Data")

SCHEMA_COLUMNS = [
    "id",
    "text",
    "label",
    "domain",
    "generator",
    "attack",
    "language",
    "source_dataset",
]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def inspect_dataset_stream(
    dataset_name: str,
    split: str = "train",
    num_samples: int = 5,
    streaming: bool = True,
    **kwargs: Any,
) -> Optional[List[Dict[str, Any]]]:
    """Inspect and stream top rows from a Hugging Face dataset."""
    if load_dataset is None:
        logger.warning(f"Inspection failed for '{dataset_name}': 'datasets' library is not installed.")
        return None
    logger.info(f"Inspecting stream from '{dataset_name}' (split={split})...")
    try:
        ds = load_dataset(dataset_name, split=split, streaming=streaming, **kwargs)
        rows: List[Dict[str, Any]] = []
        for i, row in enumerate(ds):
            rows.append(row)
            if i + 1 >= num_samples:
                break
        cols = list(rows[0].keys()) if rows else []
        logger.info(f"Dataset '{dataset_name}' columns: {cols}")
        return rows
    except Exception as err:
        logger.warning(f"Inspection failed for '{dataset_name}': {err}")
        return None


def _load_mock_stream(num_samples: int = 400) -> Generator[Dict[str, Any], None, None]:
    """Deterministic offline fallback generator for testing and smoke tier."""
    domains = ["news", "creative", "academic", "reviews"]
    generators = ["gpt2", "qwen", "llama3"]
    attacks = ["none", "paraphrase"]

    for i in range(num_samples):
        # Ensure every domain receives both human (label 0) and AI (label 1)
        dom = domains[i % len(domains)]
        is_ai = ((i // len(domains)) % 2) == 1
        gen = "human" if not is_ai else generators[(i // 2) % len(generators)]
        att = "none" if not is_ai else attacks[(i // 4) % len(attacks)]
        prefix = f"[{dom.upper()} - {gen.upper()} - {att}] "
        body = (
            "The quick brown fox jumps over the lazy dog repeatedly to test domain robust text detection. "
            f"Unique document payload token number {i} with experimental validation logic."
        )
        yield {
            "id": f"mock_{i:06d}",
            "text": prefix + body,
            "label": 1 if is_ai else 0,
            "domain": dom,
            "generator": gen,
            "attack": att,
            "language": "en",
            "source_dataset": "mock",
        }


def stream_raid_dataset(max_samples: int = 20000) -> Generator[Dict[str, Any], None, None]:
    """Stream and standardize 'liamdugan/raid' dataset."""
    if load_dataset is None:
        logger.warning("'datasets' package not installed; falling back to next source.")
        return
    try:
        # RAID splits include 'train', 'extra'
        ds = load_dataset("liamdugan/raid", split="train", streaming=True)
        count = 0
        for row in ds:
            text = row.get("generation") or row.get("text") or ""
            is_ai = int(row.get("label", 1 if row.get("model") != "human" else 0))
            gen = str(row.get("model", "human" if is_ai == 0 else "unknown"))
            domain = str(row.get("domain", "general"))
            attack = str(row.get("attack", "none"))
            rec_id = f"raid_{row.get('id', _hash_text(text))}"

            yield {
                "id": rec_id,
                "text": text,
                "label": is_ai,
                "domain": domain,
                "generator": gen,
                "attack": attack,
                "language": "en",
                "source_dataset": "raid",
            }
            count += 1
            if count >= max_samples:
                break
    except Exception as e:
        logger.warning(f"Failed to stream 'liamdugan/raid': {e}. Falling back to next source.")


def stream_m4_dataset(max_samples: int = 20000) -> Generator[Dict[str, Any], None, None]:
    """Stream and standardize M4 multi-generator multi-domain dataset."""
    if load_dataset is None:
        logger.warning("'datasets' package not installed; skipping M4.")
        return
    # Attempt known M4 repository identifiers
    m4_repos = ["yaful/M4", "baogorek/m4"]
    loaded = False
    for repo in m4_repos:
        try:
            ds = load_dataset(repo, split="train", streaming=True)
            count = 0
            for row in ds:
                text = row.get("text") or row.get("machine_text") or row.get("human_text") or ""
                label = 1 if "machine" in row or row.get("label") == 1 else 0
                domain = str(row.get("domain", "general"))
                gen = str(row.get("model", "machine" if label == 1 else "human"))
                rec_id = f"m4_{row.get('id', _hash_text(text))}"

                yield {
                    "id": rec_id,
                    "text": text,
                    "label": label,
                    "domain": domain,
                    "generator": gen,
                    "attack": "none",
                    "language": str(row.get("language", "en")),
                    "source_dataset": "m4",
                }
                count += 1
                if count >= max_samples:
                    break
            loaded = True
            break
        except Exception:
            continue

    if not loaded:
        logger.warning("Failed streaming M4 from standard repositories.")


def stream_hc3_fallback(max_samples: int = 20000) -> Generator[Dict[str, Any], None, None]:
    """Fallback loader for Hello-SimpleAI/HC3."""
    if load_dataset is None:
        logger.warning("'datasets' package not installed; skipping HC3.")
        return
    try:
        logger.info("Attempting fallback stream from 'Hello-SimpleAI/HC3'...")
        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train", streaming=True)
        count = 0
        for i, row in enumerate(ds):
            domain = str(row.get("source", "open_qa"))
            h_answers = row.get("human_answers", [])
            c_answers = row.get("chatgpt_answers", [])

            for h_text in h_answers:
                if h_text and len(h_text.strip()) > 30:
                    yield {
                        "id": f"hc3_h_{i}_{_hash_text(h_text)}",
                        "text": h_text,
                        "label": 0,
                        "domain": domain,
                        "generator": "human",
                        "attack": "none",
                        "language": "en",
                        "source_dataset": "hc3",
                    }
                    count += 1

            for c_text in c_answers:
                if c_text and len(c_text.strip()) > 30:
                    yield {
                        "id": f"hc3_ai_{i}_{_hash_text(c_text)}",
                        "text": c_text,
                        "label": 1,
                        "domain": domain,
                        "generator": "chatgpt",
                        "attack": "none",
                        "language": "en",
                        "source_dataset": "hc3",
                    }
                    count += 1

            if count >= max_samples:
                break
    except Exception as e:
        logger.warning(f"HC3 fallback stream failed: {e}. Defaulting to synthetic offline generator.")


def get_unified_stream(max_samples: int = 20000) -> Generator[Dict[str, Any], None, None]:
    """Yield standardized records from RAID -> M4 -> HC3 -> Synthetic fallback."""
    streamed = 0
    # Try RAID
    for rec in stream_raid_dataset(max_samples=max_samples):
        yield rec
        streamed += 1
        if streamed >= max_samples:
            return

    # Try M4
    if streamed < max_samples:
        for rec in stream_m4_dataset(max_samples=max_samples - streamed):
            yield rec
            streamed += 1
            if streamed >= max_samples:
                return

    # Try HC3
    if streamed < max_samples:
        for rec in stream_hc3_fallback(max_samples=max_samples - streamed):
            yield rec
            streamed += 1
            if streamed >= max_samples:
                return

    # Offline deterministic generator fallback
    if streamed < max_samples:
        needed = max_samples - streamed
        logger.info(f"Using deterministic offline stream fallback for {needed} samples.")
        for rec in _load_mock_stream(num_samples=needed):
            yield rec
            streamed += 1
            if streamed >= max_samples:
                return


def sample_and_clean_data(
    target_samples: int = 200,
    max_len_chars: int = 2048,
    min_len_chars: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Stream, clean, deduplicate, and perform chunked stratified sampling into memory."""
    set_seed(seed)
    seen_hashes: Set[str] = set()
    records: List[Dict[str, Any]] = []

    # Stream with buffer to allow stratified sampling
    buffer_target = target_samples * 2
    for item in get_unified_stream(max_samples=buffer_target):
        text = str(item.get("text", "")).strip()
        if len(text) < min_len_chars:
            continue
        # Truncate
        text = text[:max_len_chars]
        thash = _hash_text(text)
        if thash in seen_hashes:
            continue
        seen_hashes.add(thash)

        item["text"] = text
        records.append(item)
        if len(records) >= buffer_target:
            break

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Failed to retrieve any data records from streams and fallbacks.")

    # Stratified downsample by (domain, generator, label)
    df["strata"] = (
        df["domain"].astype(str)
        + "__"
        + df["generator"].astype(str)
        + "__"
        + df["label"].astype(str)
    )
    if len(df) > target_samples:
        # Proportional stratified sample
        sampled_indices = []
        for _, g in df.groupby("strata"):
            n_take = max(1, int(np.floor(len(g) / len(df) * target_samples)))
            sampled_indices.extend(g.sample(n=min(len(g), n_take), random_state=seed).index.tolist())

        # Fill deficit or trim to match exact target_samples without duplicates
        if len(sampled_indices) < target_samples:
            remaining_indices = list(set(df.index) - set(sampled_indices))
            deficit = target_samples - len(sampled_indices)
            extra_indices = pd.Series(remaining_indices).sample(
                n=min(len(remaining_indices), deficit), random_state=seed
            ).tolist()
            sampled_indices.extend(extra_indices)

        df = df.loc[sampled_indices[:target_samples]].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    df = df.drop(columns=["strata"])[SCHEMA_COLUMNS]
    clean_memory()
    return df


def plot_domain_class_distribution(df: pd.DataFrame, save_path: Path) -> None:
    """Plot single domain/class distribution bar chart and save to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    counts = df.groupby(["domain", "label"]).size().unstack(fill_value=0)
    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar", stacked=True, color=["#1f77b4", "#ff7f0e"], ax=plt.gca())
    plt.title("Sample Count by Domain and Class (0=Human, 1=AI)")
    plt.xlabel("Domain")
    plt.ylabel("Count")
    plt.legend(["Human", "AI"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved domain/class distribution chart to {save_path}")


def create_splits(
    df: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    holdout_domain: Optional[str] = None,
    holdout_generator: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate in-domain, leave-one-domain-out, and leave-one-generator-out splits with leakage assertions."""
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Automatically select holdouts if not provided
    unique_domains = sorted(df["domain"].unique().tolist())
    unique_gens = sorted(df[df["label"] == 1]["generator"].unique().tolist())

    sel_domain = holdout_domain or (unique_domains[-1] if len(unique_domains) > 1 else None)
    sel_gen = holdout_generator or (unique_gens[-1] if len(unique_gens) > 1 else None)

    # 1. Standard In-Domain Train / Val / Test (70 / 15 / 15) with stratified sampling
    df_sorted = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    train_indices = []
    val_indices = []
    test_indices = []

    # Stratify by (domain, label)
    for _, group in df_sorted.groupby(["domain", "label"]):
        g_len = len(group)
        g_indices = group.index.tolist()
        n_tr = max(1, int(round(0.70 * g_len)))
        n_va = max(1, int(round(0.15 * g_len))) if g_len >= 3 else 0
        
        tr = g_indices[:n_tr]
        va = g_indices[n_tr : n_tr + n_va]
        te = g_indices[n_tr + n_va :]
        if not te and len(tr) > 1:
            te = [tr.pop()]

        train_indices.extend(tr)
        val_indices.extend(va)
        test_indices.extend(te)

    train_df = df_sorted.loc[train_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = df_sorted.loc[val_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = df_sorted.loc[test_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Leakage assertions
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])
    assert train_ids.isdisjoint(val_ids), "Data leakage: train and val share sample IDs!"
    assert train_ids.isdisjoint(test_ids), "Data leakage: train and test share sample IDs!"
    assert val_ids.isdisjoint(test_ids), "Data leakage: val and test share sample IDs!"

    train_hashes = set(train_df["text"].apply(_hash_text))
    test_hashes = set(test_df["text"].apply(_hash_text))
    assert train_hashes.isdisjoint(test_hashes), "Data leakage: duplicate text in train & test!"

    # 2. Leave-One-Domain-Out Split
    lodo_test_df = df[df["domain"] == sel_domain].copy() if sel_domain else pd.DataFrame()
    lodo_train_df = df[df["domain"] != sel_domain].copy() if sel_domain else pd.DataFrame()

    # 3. Leave-One-Generator-Out Split
    logo_test_df = (
        df[(df["generator"] == sel_gen) & (df["label"] == 1)].copy() if sel_gen else pd.DataFrame()
    )
    logo_train_df = (
        df[~((df["generator"] == sel_gen) & (df["label"] == 1))].copy() if sel_gen else pd.DataFrame()
    )

    # Save to Parquet
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "lodo_train": lodo_train_df,
        "lodo_test": lodo_test_df,
        "logo_train": logo_train_df,
        "logo_test": logo_test_df,
    }
    paths: Dict[str, str] = {}
    for name, s_df in splits.items():
        p = output_dir / f"{name}.parquet"
        s_df.to_parquet(p, index=False)
        paths[name] = str(p)

    # EDA Plot
    plot_path = output_dir / "domain_class_dist.png"
    plot_domain_class_distribution(df, plot_path)

    # Manifest
    manifest = {
        "total_samples": len(df),
        "columns": SCHEMA_COLUMNS,
        "domains": unique_domains,
        "generators": sorted(df["generator"].unique().tolist()),
        "attacks": sorted(df["attack"].unique().tolist()),
        "holdout_domain": sel_domain,
        "holdout_generator": sel_gen,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "parquet_paths": paths,
        "plot_path": str(plot_path),
        "seed": seed,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Splits and manifest saved successfully to {output_dir}")
    return manifest
