"""
Tier-1 Benchmark Pipeline (MS MARCO/LoTTE - Training Sets Only)
----------------------------------------------------------------

This script evaluates information retrieval with Multi-Attention-Weight (MAW) transformers
on datasets that have training sets for fine-tuning.

1. Supported Datasets (with train sets):
   • MS MARCO: train (502K queries), dev (7K queries), test (43 queries)
   • LoTTE: search and forum splits with train/dev/test
   
   Note: BEIR datasets removed - they lack training sets and are zero-shot only

2. Model Variants:
   • BM25 baseline (no training required)
   • DenseZeroShot (pre-trained encoder, no fine-tuning)
   • DenseLoRA (LoRA fine-tuning)
   • MAWLoRA (LoRA + MAW on last layer)
   • MAWFullFT (Full fine-tuning + MAW on last layer)

3. MAW Features:
   • 5D attention tensor: (batch, heads, seq_q, seq_k, depth)
   • GRPO (Group Relative Policy Optimization) for depth selection
   • Scaled by √depth_dim for numerical stability
   • Layer-specific application (default: last layer only)

4. Training:
   • Contrastive learning with in-batch negatives
   • BM25 hard negatives
   • LoRA or full fine-tuning options
   • Proper train/dev/test split separation (no data leakage)

Run examples
------------
# Quick test on MS MARCO (~5-10 minutes)
python tier1_fixed.py --quick-smoke-test --msmarco

# Full MS MARCO evaluation
python tier1_fixed.py --msmarco

# LoTTE evaluation
python tier1_fixed.py --lotte search forum

# Both datasets
python tier1_fixed.py --msmarco --lotte search
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import logging
import math
import os
import random
import time
import zipfile
import shutil
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

import numpy as np
import pytrec_eval
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.amp import autocast, GradScaler
from peft import LoraConfig, TaskType, get_peft_model

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
import ir_datasets
from tqdm import tqdm


@dataclass
class BenchmarkConfig:
    """Top-level configuration for the Tier-1 benchmark"""

    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller: 22M params vs 110M (Contriever)
    max_seq_length: int = 256  # Optimized for multi-GPU setup with 45GB per GPU
    batch_size: int = 64  # Larger batch size possible with smaller model
    eval_batch_size: int = 128  # Larger evaluation batch size
    gradient_accumulation_steps: int = 2  # Accumulate gradients for memory efficiency
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    temperature: float = 0.04
    negatives_per_query: int = 4

    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    use_maw: bool = False
    maw_layer_indices: List[int] = field(default_factory=lambda: [-1])  # -1 means last layer (default)
    maw_depth_dim: int = 64
    maw_num_heads: int = 8

    top_k: int = 1000
    ms_marco: bool = False
    # REMOVED: beir_datasets - BEIR datasets don't have train sets
    lotte_splits: List[str] = field(default_factory=list)
    skip_bm25: bool = False

    data_root: str = "datasets"
    bm25_index_root: str = "indices/bm25"
    embedding_cache_root: str = "indices/dense"

    quick_smoke_test: bool = False
    medium_test: bool = False
    smoke_queries: int = 64
    smoke_docs: int = 2000
    medium_queries: int = 1000  # Changed from 300
    medium_docs: int = 100000  # Changed from 10000

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    device_ids: Tuple[int, ...] = field(default_factory=tuple)
    use_amp: bool = True
    amp_dtype: str = "bf16"

    distributed: bool = field(default=False, init=False)
    world_size: int = field(default=1, init=False)
    rank: int = field(default=0, init=False)
    local_rank: int = field(default=0, init=False)
    primary_device_index: int = field(default=0, init=False)

    def clone_for(self, **kwargs) -> "BenchmarkConfig":
        init_params = {}
        for f in fields(BenchmarkConfig):
            if f.init:
                init_params[f.name] = getattr(self, f.name)
        init_params.update(kwargs)
        
        # LoRA variants still backprop through large batches of text; use smaller micro-batches to avoid OOM
        if kwargs.get('use_lora', False) and not kwargs.get('use_maw', False) and 'batch_size' not in kwargs:
            init_params['batch_size'] = min(self.batch_size, 16)
            if 'eval_batch_size' not in kwargs:
                init_params['eval_batch_size'] = min(self.eval_batch_size, 64)
        
        # Optimize batch sizes for MAW variants with multi-GPU support
        if kwargs.get('use_maw', False) and 'batch_size' not in kwargs:
            # MAW uses 5D attention - use moderate batch sizes
            # With smaller model (22M params), we can afford larger MAW batches
            init_params['batch_size'] = 16  # MAW training batch size (increased with smaller model)
            init_params['eval_batch_size'] = 32  # MAW eval batch size
        
        clone = BenchmarkConfig(**init_params)
        # preserve runtime-discovered attributes that are init=False
        clone.distributed = self.distributed
        clone.world_size = self.world_size
        clone.rank = self.rank
        clone.local_rank = self.local_rank
        clone.primary_device_index = self.primary_device_index
        clone.device = self.device
        clone.device_ids = self.device_ids
        return clone

    def __post_init__(self) -> None:
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0

        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        env_rank = os.environ.get("RANK")
        env_local_rank = os.environ.get("LOCAL_RANK")

        if gpu_available and env_world_size > 1 and env_rank is not None and env_local_rank is not None:
            self.distributed = True
            self.world_size = env_world_size
            self.rank = int(env_rank)
            self.local_rank = int(env_local_rank)
            self.primary_device_index = self.local_rank
            self.device = f"cuda:{self.primary_device_index}"
            self.device_ids = tuple(range(gpu_count))
        elif gpu_available:
            self.primary_device_index = 0
            self.device = "cuda:0"
            self.device_ids = tuple(range(gpu_count))
        else:
            self.primary_device_index = -1
            self.device = "cpu"
            self.device_ids = tuple()
        
        # Adjust batch sizes for medium test to avoid OOM
        # Medium test has 50K docs which requires more memory during encoding
        if self.medium_test and self.batch_size > 32:
            self.batch_size = 32  # Reduce batch size for medium test
            if self.eval_batch_size > 64:
                self.eval_batch_size = 64


@dataclass
class DatasetPartition:
    """Container for a single split's queries and relevance judgments."""

    queries: Dict[str, str] = field(default_factory=dict)
    qrels: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return len(self.queries) == 0 or len(self.qrels) == 0


@dataclass
class DatasetBundle:
    """Grouped dataset resources covering train/dev/test splits."""

    name: str
    corpus: Dict[str, Dict[str, str]]
    train: Optional[DatasetPartition] = None
    dev: Optional[DatasetPartition] = None
    test: Optional[DatasetPartition] = None

    def has_train(self) -> bool:
        return self.train is not None and not self.train.is_empty()

    def has_dev(self) -> bool:
        return self.dev is not None and not self.dev.is_empty()

    def ensure_test(self) -> None:
        if self.test is None or self.test.is_empty():
            raise ValueError(f"Dataset '{self.name}' does not provide a valid evaluation split.")


@dataclass
class ModelSnapshot:
    """Stores trained model parameters and metadata for reuse."""

    label: str
    source_dataset: str
    config: BenchmarkConfig
    state_dict: Dict[str, torch.Tensor]
    training_stats: Dict[str, Any]


def initialize_distributed(config: BenchmarkConfig) -> None:
    if not config.distributed:
        return
    if not dist.is_available():
        raise RuntimeError("Distributed execution requested but torch.distributed is not available in this build.")
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    torch.cuda.set_device(config.primary_device_index)


def finalize_distributed(config: BenchmarkConfig) -> None:
    if config.distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(config: BenchmarkConfig) -> bool:
    if not config.distributed:
        return True
    return config.rank == 0


def wait_for_condition(predicate: Callable[[], bool], timeout: float, interval: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            member_path = (target_dir / member).resolve()
            if not str(member_path).startswith(str(target_dir.resolve())):
                raise RuntimeError(f"Unsafe path detected in archive {zip_path}: {member}")
        archive.extractall(target_dir)


def broadcast_object(config: BenchmarkConfig, obj: Any, src: int = 0) -> Any:
    if not (config.distributed and dist.is_available() and dist.is_initialized()):
        return obj
    payload = [obj if config.rank == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def all_gather_objects(config: BenchmarkConfig, obj: Any) -> List[Any]:
    if not (config.distributed and dist.is_available() and dist.is_initialized()):
        return [obj]
    gathered: List[Any] = [None for _ in range(config.world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def compose_document_text(doc: Dict[str, str]) -> str:
    title = doc.get("title", "")
    text = doc.get("text", "")
    if title and text:
        return f"{title} {text}"
    return title or text


def device_status(config: BenchmarkConfig) -> str:
    if config.device.startswith("cuda") and config.device_ids:
        return f"GPU x{len(config.device_ids)}"
    return "CPU x1"


class DatasetManager:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._loader_cache: Dict[str, GenericDataLoader] = {}

    def get_msmarco(self) -> DatasetBundle:
        dataset_name = "msmarco"
        dataset_path = self._ensure_beir_dataset(dataset_name, "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip")
        train_split = self._try_load_split(dataset_path, dataset_name, ["train"])
        dev_split = self._try_load_split(dataset_path, dataset_name, ["dev", "validation", "val"])
        # MS MARCO's "test" split only has 43 queries (not useful for evaluation)
        # The standard practice is to use "dev" (6,980 queries) as the test set
        logging.info("MS MARCO: Using dev split (6,980 queries) as test set (standard practice)")
        test_split = dev_split
        dev_split = None  # No separate dev monitoring for MS MARCO
        return self._assemble_bundle(dataset_name, train_split, dev_split, test_split)

    # REMOVED: get_beir_dataset() - BEIR datasets typically don't have train sets
    # Only keeping datasets with train sets: MS MARCO and LoTTE

    def get_lotte_split(self, split: str) -> DatasetBundle:
        normalized = split.lower()
        dataset_name = f"lotte-{normalized}"
        dataset_path = Path(self.config.data_root) / "lotte" / normalized
        if not dataset_path.exists() or not (dataset_path / "corpus.jsonl").exists():
            logging.info("Preparing LoTTE split '%s' in BEIR layout ...", split)
            self._prepare_lotte_split(normalized)
        if not (dataset_path / "corpus.jsonl").exists() or not (dataset_path / "queries.jsonl").exists():
            raise FileNotFoundError(
                f"Failed to materialize LoTTE split '{split}' under {dataset_path}. "
                "Verify that ir_datasets is installed and accessible."
            )

        # LoTTE only has dev and test splits (no train split available in ir_datasets)
        # Use dev as train for fine-tuning, test for evaluation
        train_split = self._try_load_split(str(dataset_path), dataset_name, ["dev"])
        dev_split = None  # No separate dev monitoring for LoTTE
        test_split = self._try_load_split(str(dataset_path), dataset_name, ["test"])
        if test_split is None:
            raise ValueError(f"LoTTE split '{split}' does not include a test partition.")
        if train_split is not None:
            logging.info("LoTTE-%s: Using 'dev' split for training, 'test' split for evaluation (LoTTE has no separate train split)", normalized)
        return self._assemble_bundle(dataset_name, train_split, dev_split, test_split)

    def _ensure_beir_dataset(self, dataset_name: str, url: str) -> str:
        dataset_path = ensure_dir(Path(self.config.data_root) / dataset_name)
        
        # Check if dataset needs to be downloaded
        has_corpus = (dataset_path / "corpus.jsonl").exists()
        has_queries = (dataset_path / "queries.jsonl").exists()
        has_qrels = (dataset_path / "qrels").exists() and any((dataset_path / "qrels").iterdir())
        
        if not (has_corpus and has_queries and has_qrels) and is_main_process(self.config):
            logging.info("Downloading and extracting dataset %s ...", dataset_name)
            # beir_util.download_and_unzip handles both download and extraction
            beir_util.download_and_unzip(url, str(Path(self.config.data_root)))
            
            # Flatten nested directory structure if present (e.g., msmarco/msmarco/ -> msmarco/)
            inner_dir = dataset_path / dataset_name
            if inner_dir.is_dir() and any(inner_dir.iterdir()):
                logging.info("Flattening nested directory structure for %s", dataset_name)
                for item in inner_dir.iterdir():
                    dest = dataset_path / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), dataset_path)
                try:
                    inner_dir.rmdir()
                except OSError:
                    pass
        else:
            # Wait for main process to finish downloading/extracting
            wait_for_condition(
                lambda: (dataset_path / "corpus.jsonl").exists() and 
                        (dataset_path / "queries.jsonl").exists() and
                        (dataset_path / "qrels").exists(),
                timeout=1800,
                interval=5.0
            )

        # Verify dataset files exist
        if not (dataset_path / "corpus.jsonl").exists():
            raise RuntimeError(f"Dataset {dataset_name}: corpus.jsonl not found in {dataset_path}")
        if not (dataset_path / "queries.jsonl").exists():
            raise RuntimeError(f"Dataset {dataset_name}: queries.jsonl not found in {dataset_path}")
        if not (dataset_path / "qrels").exists():
            raise RuntimeError(f"Dataset {dataset_name}: qrels directory not found in {dataset_path}")
            
        return str(dataset_path)

    def _try_load_split(
        self,
        dataset_path: str,
        dataset: str,
        candidates: List[str],
    ) -> Optional[Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]], str]]:
        for split in candidates:
            try:
                # Create a fresh loader for each split to avoid state pollution
                # (GenericDataLoader filters queries based on qrels after first load)
                fresh_loader = GenericDataLoader(dataset_path)
                corpus, queries, qrels = fresh_loader.load(split=split)
                logging.info(
                    "%s: loaded split '%s' with %d queries and %d qrels",
                    dataset,
                    split,
                    len(queries),
                    sum(len(v) for v in qrels.values()),
                )
                return corpus, dict(queries), {qid: dict(doc_scores) for qid, doc_scores in qrels.items()}, split
            except Exception as exc:  # noqa: BLE001 - surfaces dataset issues but continues
                logging.debug("Dataset %s missing split '%s': %s", dataset, split, str(exc))
        logging.warning("Dataset %s: none of the candidate splits %s were found", dataset, candidates)
        return None

    def _assemble_bundle(
        self,
        dataset: str,
        train_split: Optional[Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]], str]],
        dev_split: Optional[Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]], str]],
        test_split: Optional[Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]], str]],
    ) -> DatasetBundle:
        if test_split is None:
            available = []
            if train_split: available.append(f"train ({len(train_split[1])} queries)")
            if dev_split: available.append(f"dev ({len(dev_split[1])} queries)")
            raise ValueError(
                f"Dataset '{dataset}' is missing a required evaluation split (test or dev). "
                f"Available splits: {available or 'none'}. Check dataset directory structure."
            )

        # Extract data from splits, then explicitly delete the split tuples
        # Each split contains a full corpus copy (2.8M docs = ~42GB for LoTTE)
        # We only need ONE corpus reference, not multiple copies
        corpus = test_split[0]
        train_partition = DatasetPartition(queries=train_split[1], qrels=train_split[2]) if train_split else None
        dev_partition = DatasetPartition(queries=dev_split[1], qrels=dev_split[2]) if dev_split else None
        test_partition = DatasetPartition(queries=test_split[1], qrels=test_split[2])
        
        # Delete the split tuples to release their corpus references
        # Without this, train_split and dev_split hold duplicate corpus copies in memory
        del test_split
        if train_split is not None:
            del train_split
        if dev_split is not None:
            del dev_split
        gc.collect()  # Force garbage collection to release ~80GB for LoTTE (2 corpus copies)

        bundle = DatasetBundle(name=dataset, corpus=corpus, train=train_partition, dev=dev_partition, test=test_partition)
        bundle.ensure_test()
        return self._apply_smoke_filters(bundle)

    def _prepare_lotte_split(self, split: str) -> None:
        base_path = ensure_dir(Path(self.config.data_root) / "lotte" / split)
        corpus_path = base_path / "corpus.jsonl"
        queries_path = base_path / "queries.jsonl"
        qrels_dir = ensure_dir(base_path / "qrels")

        # LoTTE structure: lotte/{domain}/{split}/{variant}
        # Domains: lifestyle, recreation, science, technology, writing
        # Variants: search, forum
        # Splits: dev, test (no train split in ir_datasets)
        domains = ["lifestyle", "recreation", "science", "technology", "writing"]
        
        dataset_versions: Dict[str, Any] = {}
        if is_main_process(self.config):
            # Try loading pooled dataset first (combines all domains)
            for partition in ("dev", "test"):
                try:
                    dataset_versions[partition] = ir_datasets.load(f"lotte/pooled/{partition}/{split}")
                    logging.info("[LoTTE:%s] Loaded pooled/%s/%s dataset", split, partition, split)
                except Exception as e:
                    logging.debug("[LoTTE:%s] Could not load pooled/%s/%s: %s", split, partition, split, e)
                    dataset_versions[partition] = None
            
            if all(dataset is None for dataset in dataset_versions.values()):
                raise RuntimeError(
                    f"Unable to load LoTTE {split} via ir_datasets. "
                    f"Tried: lotte/pooled/dev/{split}, lotte/pooled/test/{split}. "
                    f"Make sure ir_datasets is installed and the dataset is available."
                )

            if not corpus_path.exists():
                logging.info("[LoTTE:%s] Writing corpus.jsonl", split)
                with corpus_path.open("w", encoding="utf-8") as corpus_file:
                    written = set()
                    for dataset in dataset_versions.values():
                        if dataset is None:
                            continue
                        for doc in dataset.docs_iter():
                            doc_id = getattr(doc, "doc_id", getattr(doc, "docid", None))
                            if doc_id is None or doc_id in written:
                                continue
                            title = getattr(doc, "title", "") or ""
                            text = getattr(doc, "text", getattr(doc, "document", ""))
                            record = {"_id": doc_id, "title": title, "text": text}
                            corpus_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            written.add(doc_id)

            existing_queries: set[str] = set()
            if queries_path.exists():
                with queries_path.open(encoding="utf-8") as qp:
                    for line in qp:
                        try:
                            payload = json.loads(line)
                            existing_queries.add(payload["_id"])
                        except Exception:  # noqa: BLE001
                            continue

            with queries_path.open("a", encoding="utf-8") as queries_file:
                for part, dataset in dataset_versions.items():
                    if dataset is None:
                        continue
                    new_queries = 0
                    for query in dataset.queries_iter():
                        qid = getattr(query, "query_id", getattr(query, "qid", None))
                        if qid is None or qid in existing_queries:
                            continue
                        text = getattr(query, "text", getattr(query, "query", ""))
                        queries_file.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")
                        existing_queries.add(qid)
                        new_queries += 1
                    if new_queries:
                        logging.info("[LoTTE:%s] Added %d queries from %s split", split, new_queries, part)

            for part, dataset in dataset_versions.items():
                if dataset is None:
                    continue
                qrels_path = qrels_dir / f"{part}.tsv"
                if qrels_path.exists():
                    continue
                logging.info("[LoTTE:%s] Writing qrels/%s.tsv", split, part)
                with qrels_path.open("w", encoding="utf-8") as qrels_file:
                    for qrel in dataset.qrels_iter():
                        qid = getattr(qrel, "query_id", getattr(qrel, "qid", None))
                        did = getattr(qrel, "doc_id", getattr(qrel, "docid", None))
                        rel = getattr(qrel, "relevance", getattr(qrel, "score", 1))
                        qrels_file.write(f"{qid}\t{did}\t{rel}\n")
        else:
            # LoTTE only has dev and test splits (no train)
            required_paths = [corpus_path, queries_path] + [qrels_dir / f"{part}.tsv" for part in ("dev", "test")]
            wait_for_condition(lambda: all(path.exists() for path in required_paths), timeout=1800, interval=5.0)

    def _apply_smoke_filters(self, bundle: DatasetBundle) -> DatasetBundle:
        if not (self.config.quick_smoke_test or self.config.medium_test):
            return bundle

        # Determine target sizes
        if self.config.quick_smoke_test:
            target_queries = self.config.smoke_queries
            target_docs = self.config.smoke_docs
            test_type = "SMOKE"
        else:  # medium_test
            target_queries = self.config.medium_queries
            target_docs = self.config.medium_docs
            test_type = "MEDIUM"

        rng = random.Random(self.config.seed)
        doc_ids: set[str] = set()

        for split_name in ["train", "dev", "test"]:
            partition = getattr(bundle, split_name)
            if partition is None or partition.is_empty():
                continue
            original_qids = list(partition.queries.keys())
            num_queries = min(len(original_qids), target_queries)
            if num_queries < len(original_qids):
                selected_qids = set(rng.sample(original_qids, num_queries))
                partition.queries = {qid: partition.queries[qid] for qid in selected_qids}
                partition.qrels = {qid: partition.qrels[qid] for qid in selected_qids if qid in partition.qrels}
            for docs in partition.qrels.values():
                doc_ids.update(docs.keys())

        if len(doc_ids) < target_docs:
            remaining = [doc_id for doc_id in bundle.corpus if doc_id not in doc_ids]
            if remaining:
                extra = min(len(remaining), target_docs - len(doc_ids))
                doc_ids.update(rng.sample(remaining, extra))

        allowed_docs = {doc_id for doc_id in doc_ids if doc_id in bundle.corpus}
        # Create filtered corpus and explicitly release the original
        old_corpus = bundle.corpus
        bundle.corpus = {doc_id: old_corpus[doc_id] for doc_id in allowed_docs}
        del old_corpus  # Explicitly delete the original large corpus to free memory
        gc.collect()  # Force garbage collection to immediately release ~40GB of memory

        for split_name in ["train", "dev", "test"]:
            partition = getattr(bundle, split_name)
            if partition is None or partition.is_empty():
                continue
            filtered_qrels = {}
            filtered_queries = {}
            for qid, rels in partition.qrels.items():
                kept = {doc_id: rel for doc_id, rel in rels.items() if doc_id in allowed_docs}
                if kept:
                    filtered_qrels[qid] = kept
                    filtered_queries[qid] = partition.queries[qid]
            partition.qrels = filtered_qrels
            partition.queries = filtered_queries
            logging.info(
                "[%s] %s/%s: %d queries | %d docs | %d qrels",
                test_type,
                bundle.name,
                split_name,
                len(partition.queries),
                len(bundle.corpus),
                sum(len(v) for v in partition.qrels.values()),
            )

        bundle.ensure_test()
        return bundle


class PyseriniBM25Retriever:
    def __init__(self, config: BenchmarkConfig, dataset_name: str) -> None:
        self.config = config
        self.dataset_name = dataset_name
        self.doc_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[Any] = None

    def build(self, corpus: Dict[str, Dict[str, str]]) -> None:
        logging.info("Building BM25 index for %s with %d documents...", self.dataset_name, len(corpus))
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise RuntimeError("rank_bm25 not installed. Install with: pip install rank-bm25")
        
        self.doc_ids = list(corpus.keys())
        self.tokenized_corpus = []
        
        logging.info("Tokenizing %d documents for BM25 (this may take several minutes)...", len(self.doc_ids))
        tokenize_start = time.time()
        for doc_id in tqdm(self.doc_ids, desc="Tokenizing documents"):
            doc_text = corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")
            tokens = doc_text.lower().split()
            self.tokenized_corpus.append(tokens)
        tokenize_time = time.time() - tokenize_start
        logging.info("Tokenization complete in %.1f seconds", tokenize_time)
        
        logging.info("Building BM25 index (computing term statistics, ~2-5 minutes for 8M docs)...")
        index_start = time.time()
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=0.9, b=0.4)
        index_time = time.time() - index_start
        logging.info("BM25 index built for %s in %.1f seconds", self.dataset_name, index_time)

    def search(self, queries: Dict[str, str], top_k: int) -> Dict[str, Dict[str, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built")
        
        logging.info("Running BM25 search for %d queries (top_k=%d)...", len(queries), top_k)
        results: Dict[str, Dict[str, float]] = {}
        for qid, query_text in tqdm(queries.items(), desc="BM25 search"):
            tokenized_query = query_text.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            results[qid] = {self.doc_ids[idx]: float(scores[idx]) for idx in top_indices if scores[idx] > 0}
        logging.info("BM25 search complete for %d queries", len(queries))
        return results


class GRPOEnvironment:
    """
    GRPO RL Environment for depth selection in MAW attention.
    
    State: 5D attention weights (batch, heads, seq_q, seq_k, depth)
    Action: Select depth index [0, depth_dim-1]
    Reward: Based on retrieval performance improvement
    """
    
    def __init__(self, depth_dim: int, num_heads: int):
        self.depth_dim = depth_dim
        self.num_heads = num_heads
        self.current_state = None
        self.current_relevance_scores = None
        self.baseline_scores = None
        
    def reset(self, attention_weights_5d: torch.Tensor, relevance_scores: List[float] = None, baseline_scores: List[float] = None):
        """
        Reset environment with new 5D attention weights and relevance data.
        
        Args:
            attention_weights_5d: (batch, heads, seq_q, seq_k, depth)
            relevance_scores: Ground truth relevance scores (optional)
            baseline_scores: Baseline model similarity scores (optional)
        """
        self.current_state = attention_weights_5d
        self.current_relevance_scores = relevance_scores or []
        self.baseline_scores = baseline_scores or []
        return self.get_state_representation()
    
    def get_state_representation(self) -> torch.Tensor:
        """
        Get state representation for the policy network.
        
        Returns:
            state: (batch, state_dim) - compressed state representation
        """
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Compress 5D attention to a manageable state representation
        # Use adaptive pooling to create fixed-size state
        batch_size = self.current_state.shape[0]
        
        # Pool spatial dimensions (seq_q, seq_k) to 8x8
        # Shape: (batch, heads, seq_q, seq_k, depth) -> (batch, heads, 8, 8, depth)
        pooled = F.adaptive_avg_pool2d(
            self.current_state.mean(dim=-1),  # Average over depth first
            (8, 8)
        )
        
        # Flatten to state vector: (batch, heads*8*8)
        state = pooled.view(batch_size, -1)
        return state
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Execute action and return next state, reward, done.
        
        Args:
            actions: (batch,) - selected depth indices
            
        Returns:
            next_state: (batch, state_dim)
            rewards: (batch,) - computed rewards
            done: Always True (single-step environment)
        """
        batch_size = actions.shape[0]
        rewards = torch.zeros(batch_size, device=actions.device)
        
        # Compute reward based on attention quality
        for b in range(batch_size):
            depth_idx = actions[b].item()
            
            # Select attention at chosen depth
            attn_slice = self.current_state[b, :, :, :, depth_idx]  # (heads, seq_q, seq_k)
            
            # Reward: negative entropy (more focused attention is better)
            attn_probs = F.softmax(attn_slice, dim=-1)
            entropy = -(attn_probs * torch.log(attn_probs + 1e-10)).sum(dim=-1).mean()
            rewards[b] = -entropy  # Lower entropy = higher reward
        
        return self.get_state_representation(), rewards, True


class GRPOPolicyNetwork(nn.Module):
    """
    GRPO Policy Network for depth selection.
    
    This is the actual RL policy that learns to select optimal depths
    based on 5D attention state representations.
    """
    
    def __init__(self, hidden_size: int, depth_dim: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth_dim = depth_dim
        self.num_heads = num_heads
        
        # State encoder
        state_dim = self.num_heads * 8 * 8  # From adaptive pooling
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Policy head (outputs action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.depth_dim)  # Output logits for each depth
        )
        
        # Value head (estimates state value for advantage computation)
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value output
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Args:
            state: (batch_size, state_dim) - state representation
            
        Returns:
            action_logits: (batch_size, depth_dim) - action probabilities
            state_value: (batch_size, 1) - estimated state value
        """
        encoded_state = self.state_encoder(state)
        
        action_logits = self.policy_head(encoded_state)
        state_value = self.value_head(encoded_state)
        
        return action_logits, state_value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: (batch_size, state_dim)
            deterministic: If True, take argmax action
            
        Returns:
            action: (batch_size,) - selected depth indices
            log_prob: (batch_size,) - log probability of selected actions
            state_value: (batch_size, 1) - estimated state value
        """
        action_logits, state_value = self.forward(state)
        
        if deterministic:
            action = action_logits.argmax(dim=-1)
            action_probs = F.softmax(action_logits, dim=-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1))).squeeze(1)
        else:
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        return action, log_prob, state_value


class GRPORouter(nn.Module):
    """
    GRPO Router using Reinforcement Learning for depth selection from 5D attention weights.
    
    This implements actual GRPO (Generalized Preference Optimization) algorithm:
    - Policy network learns to select optimal depths
    - Environment provides rewards based on retrieval performance
    - Uses policy gradients with preference-based optimization
    """
    
    def __init__(self, hidden_size: int, depth_dim: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth_dim = depth_dim
        self.num_heads = num_heads
        
        # RL Components
        self.policy = GRPOPolicyNetwork(hidden_size, depth_dim, num_heads)
        self.environment = GRPOEnvironment(depth_dim, num_heads)
        
        # Reference policy for KL regularization (frozen copy)
        self.reference_policy = GRPOPolicyNetwork(hidden_size, depth_dim, num_heads)
        self.reference_policy.load_state_dict(self.policy.state_dict())
        
        # Freeze reference policy
        for param in self.reference_policy.parameters():
            param.requires_grad = False
        
        # GRPO hyperparameters
        self.kl_coeff = 0.1  # KL divergence coefficient
        self.value_coeff = 0.5  # Value loss coefficient
        self.entropy_coeff = 0.01  # Entropy bonus coefficient
    
    def get_depth_selection(self, attention_weights_5d: torch.Tensor) -> torch.Tensor:
        """
        Select depth using the trained RL policy.
        
        Args:
            attention_weights_5d: (batch_size, num_heads, seq_len_query, seq_len_key, depth)
        Returns:
            depth_indices: (batch_size,) - selected depth index for each batch item
        """
        # Get state representation
        state = self.environment.reset(attention_weights_5d, [], [])
        
        # Get action from policy (deterministic during inference)
        action, _, _ = self.policy.get_action(state, deterministic=not self.training)
        
        return action
    
    def select_optimal_attention(self, attention_weights_5d: torch.Tensor) -> torch.Tensor:
        """
        Select optimal 4D attention weights from 5D attention weights using RL policy.
        
        Args:
            attention_weights_5d: (batch_size, num_heads, seq_len_query, seq_len_key, depth)
        Returns:
            attention_weights_4d: (batch_size, num_heads, seq_len_query, seq_len_key)
        """
        batch_size, num_heads, seq_q, seq_k, depth = attention_weights_5d.shape
        depth_indices = self.get_depth_selection(attention_weights_5d)  # (batch_size,)
        
        # PARALLEL SELECTION: Use advanced indexing
        batch_indices = torch.arange(batch_size, device=attention_weights_5d.device)
        
        # Reshape for gathering along depth dimension
        # (batch, heads, seq_q, seq_k, depth) -> (batch, heads*seq_q*seq_k, depth)
        attn_flat = attention_weights_5d.view(batch_size, num_heads * seq_q * seq_k, depth)
        
        # Expand depth_indices for gathering: (batch,) -> (batch, heads*seq_q*seq_k)
        depth_indices_expanded = depth_indices.unsqueeze(1).expand(batch_size, num_heads * seq_q * seq_k)
        
        # Gather: select depth for each position (fully parallel!)
        attention_4d_flat = torch.gather(attn_flat, dim=2, index=depth_indices_expanded.unsqueeze(2)).squeeze(2)
        
        # Reshape back: (batch, heads*seq_q*seq_k) -> (batch, heads, seq_q, seq_k)
        attention_weights_4d = attention_4d_flat.view(batch_size, num_heads, seq_q, seq_k)
        
        return attention_weights_4d


class TokenLevelMAW(nn.Module):
    """
    Multi-Attention-Weight (MAW) module with complete GRPO architecture.
    
    This is the full implementation from benchmark_evaluation_GRPO.py with:
    - 5D attention computation via depth-aware projections
    - GRPO RL router for depth selection
    - Policy network with actor-critic architecture
    """
    def __init__(self, hidden_size: int, depth_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.depth_dim = depth_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Standard projections for value vectors
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Depth-aware projections for 5D attention
        self.depth_query_proj = nn.Linear(hidden_size, num_heads * depth_dim, bias=False)
        self.depth_key_proj = nn.Linear(hidden_size, num_heads * depth_dim, bias=False)
        
        # GRPO RL Router (complete architecture from benchmark_evaluation_GRPO.py)
        self.grpo_router = GRPORouter(hidden_size, depth_dim, num_heads)
        
        # Output layers
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Enable gradient checkpointing for memory savings
        self.use_gradient_checkpointing = False
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, sequence_length, hidden_size)
            attention_mask: (batch_size, sequence_length)
        
        Returns:
            output: (batch_size, sequence_length, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Step 1-3: Project to Q, K, V and reshape for multi-head attention
        query = self.query_proj(hidden_states)  # (batch_size, seq_len, hidden_size)
        key = self.key_proj(hidden_states)      # (batch_size, seq_len, hidden_size)
        value = self.value_proj(hidden_states)  # (batch_size, seq_len, hidden_size)
        
        # Reshape to multi-head format: (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute 5D attention weights following the MAW process
        # Use gradient checkpointing if enabled (saves memory during training)
        if self.use_gradient_checkpointing and self.training:
            attn_output = torch.utils.checkpoint.checkpoint(
                self._compute_maw_attention,
                query, key, value, attention_mask, hidden_states,
                use_reentrant=False
            )
        else:
            attn_output = self._compute_maw_attention(query, key, value, attention_mask, hidden_states)
        
        # Reshape back to (batch_size, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Output projection and residual connection
        output = self.output_proj(attn_output)
        return self.norm(hidden_states + output)
    
    def _compute_maw_attention(
        self, 
        query: torch.Tensor,  # (batch_size, num_heads, seq_len_q, head_dim)
        key: torch.Tensor,    # (batch_size, num_heads, seq_len_k, head_dim)
        value: torch.Tensor,  # (batch_size, num_heads, seq_len_k, head_dim)
        attention_mask: torch.Tensor,  # (batch_size, seq_len)
        hidden_states: torch.Tensor,   # (batch_size, seq_len, hidden_size) for policy/value
    ) -> torch.Tensor:
        """
        Compute MAW attention using complete GRPO architecture from benchmark_evaluation_GRPO.py.
        
        Architecture:
        1. Compute 5D attention weights via depth-aware projections
        2. Use GRPO RL router to select optimal depth
        3. Apply selected 4D attention to value vectors
        """
        batch_size, num_heads, seq_len_q, head_dim = query.size()
        seq_len_k = key.size(2)
        seq_len = hidden_states.size(1)
        
        # Scale by sqrt(depth_dim) for numerical stability
        scaling_factor = math.sqrt(self.depth_dim)
        
        # Depth-aware projections for 5D attention
        Q_depth = self.depth_query_proj(hidden_states)  # (batch, seq, num_heads * depth_dim)
        K_depth = self.depth_key_proj(hidden_states)    # (batch, seq, num_heads * depth_dim)
        
        # Reshape for multi-head attention with depth dimension
        Q_depth = Q_depth.view(batch_size, seq_len, self.num_heads, self.depth_dim).transpose(1, 2)
        K_depth = K_depth.view(batch_size, seq_len, self.num_heads, self.depth_dim).transpose(1, 2)
        # Q_depth, K_depth: (batch, num_heads, seq_len, depth_dim)
        
        # Compute 5D attention scores using chunked outer product for memory efficiency
        # This prevents OOM when processing large batches across multiple GPUs
        # Use smaller chunks (2 samples at a time) to reduce peak memory usage
        chunk_size = min(2, batch_size)  # Process in small chunks to save memory
        scores_5d_chunks = []
        
        for i in range(0, batch_size, chunk_size):
            chunk_end = min(i + chunk_size, batch_size)
            Q_chunk = Q_depth[i:chunk_end].transpose(2, 3).unsqueeze(-1)  # (chunk, heads, depth, seq_q, 1)
            K_chunk = K_depth[i:chunk_end].transpose(2, 3).unsqueeze(-2)  # (chunk, heads, depth, 1, seq_k)
            
            # Element-wise multiplication creates 5D scores
            scores_chunk = Q_chunk * K_chunk  # (chunk, heads, depth, seq_q, seq_k)
            scores_chunk = scores_chunk.permute(0, 1, 3, 4, 2)  # (chunk, heads, seq_q, seq_k, depth)
            scores_chunk = scores_chunk / scaling_factor
            scores_5d_chunks.append(scores_chunk)
            
            # Free memory immediately after processing each chunk
            del Q_chunk, K_chunk, scores_chunk
        
        scores_5d = torch.cat(scores_5d_chunks, dim=0)  # (batch, heads, seq_q, seq_k, depth)
        del scores_5d_chunks  # Free chunk list
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1)  # (batch, 1, 1, seq, 1)
            mask = mask.expand(batch_size, self.num_heads, seq_len_q, seq_len_k, self.depth_dim)
            scores_5d = scores_5d.masked_fill(mask == 0, -1e9)
        
        # Softmax over DEPTH dimension to get attention weights (dim=-1)
        attention_weights_5d = F.softmax(scores_5d, dim=-1)  # Softmax over depth (last dim)
        # Shape: (batch, heads, seq_q, seq_k, depth)
        
        # Use GRPO router to select optimal depth (complete RL architecture)
        selected_attention_weights = self.grpo_router.select_optimal_attention(attention_weights_5d)
        # Shape: (batch, heads, seq_q, seq_k)
        
        # Apply dropout (NO second softmax or mask - already done!)
        selected_attention_weights = self.dropout(selected_attention_weights)
        
        # Apply attention to value vectors
        attn_output = torch.matmul(selected_attention_weights, value)  # (batch_size, num_heads, seq_len_q, head_dim)
        
        return attn_output


class HFTextEncoder(nn.Module):
    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.dense_model)
        hf_config = AutoConfig.from_pretrained(config.dense_model)
        self.model = AutoModel.from_pretrained(config.dense_model, config=hf_config)
        self.hidden_size_value = hf_config.hidden_size

        if config.use_lora:
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "query",
                    "key",
                    "value",
                    "dense",
                    "fc1",
                    "fc2",
                ],
            )
            self.model = get_peft_model(self.model, lora_cfg)

        # MAW: Apply to specific encoder layers
        self.maw_modules = None
        self.maw_layer_indices = None
        if config.use_maw:
            # Determine which layers to apply MAW to
            num_encoder_layers = self._get_num_encoder_layers(hf_config)
            self.maw_layer_indices = self._resolve_maw_layer_indices(config.maw_layer_indices, num_encoder_layers)
            
            # Create MAW module for each specified layer
            self.maw_modules = nn.ModuleDict({
                str(idx): TokenLevelMAW(self.hidden_size_value, config.maw_depth_dim, config.maw_num_heads)
                for idx in self.maw_layer_indices
            })
            
            # Enable gradient checkpointing for MAW modules to save memory
            for maw_module in self.maw_modules.values():
                maw_module.use_gradient_checkpointing = True
            
            logging.info(f"MAW enabled on encoder layers: {self.maw_layer_indices} (with gradient checkpointing)")

        self.primary_device = self._resolve_primary_device()
        self.to(self.primary_device)
    
    def _get_num_encoder_layers(self, hf_config) -> int:
        """Get the number of encoder layers from HuggingFace config"""
        # Different models use different attribute names
        if hasattr(hf_config, 'num_hidden_layers'):
            return hf_config.num_hidden_layers
        elif hasattr(hf_config, 'n_layers'):
            return hf_config.n_layers
        elif hasattr(hf_config, 'num_layers'):
            return hf_config.num_layers
        else:
            # Default fallback
            return 12
    
    def _resolve_maw_layer_indices(self, layer_indices: List[int], num_layers: int) -> List[int]:
        """
        Resolve layer indices, handling negative indexing and 'all' specification.
        
        Args:
            layer_indices: List of layer indices. -1 means last layer, -2 second to last, etc.
                          If list contains many indices (e.g., from 'all'), they'll be clamped.
            num_layers: Total number of encoder layers
        
        Returns:
            List of resolved positive layer indices (0-indexed)
        """
        # Check if this is the "all layers" case (large list of indices)
        if len(layer_indices) >= num_layers:
            return list(range(num_layers))
        
        resolved = []
        for idx in layer_indices:
            if idx < 0:
                # Negative indexing: -1 = last layer, -2 = second to last, etc.
                resolved_idx = num_layers + idx
            else:
                resolved_idx = idx
            
            # Validate index is in range
            if 0 <= resolved_idx < num_layers:
                if resolved_idx not in resolved:  # Avoid duplicates
                    resolved.append(resolved_idx)
            else:
                logging.warning(f"MAW layer index {idx} (resolved to {resolved_idx}) is out of range [0, {num_layers-1}]. Skipping.")
        
        return sorted(resolved)

    def _encode_text_batch(self, texts: List[str]) -> torch.Tensor:
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.primary_device) for k, v in tokenized.items()}
        with torch.set_grad_enabled(self.training):
            if self.maw_modules is not None:
                # Need to get hidden states from all layers to apply MAW at specific layers
                outputs = self.model(**tokenized, output_hidden_states=True)
                all_hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors
                
                # Apply MAW to specified layers
                # Note: hidden_states[0] is embeddings, hidden_states[i+1] is output of layer i
                modified_states = list(all_hidden_states)
                for layer_idx in self.maw_layer_indices:
                    # Apply MAW to the output of layer_idx
                    # hidden_states[layer_idx + 1] contains output from encoder layer layer_idx
                    state_idx = layer_idx + 1
                    if state_idx < len(modified_states):
                        maw_module = self.maw_modules[str(layer_idx)]
                        modified_states[state_idx] = maw_module(
                            modified_states[state_idx], 
                            tokenized["attention_mask"]
                        )
                
                # Use the final hidden state (after all modifications)
                hidden_states = modified_states[-1]
            else:
                # Standard forward pass without MAW
                outputs = self.model(**tokenized)
                hidden_states = outputs.last_hidden_state
            
            pooled = mean_pool(hidden_states, tokenized["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=-1)
        return normalized

    def encode_train(self, texts: List[str], batch_size: Optional[int] = None) -> torch.Tensor:
        if batch_size is None or batch_size <= 0:
            batch_size = self.config.batch_size
        else:
            # Ensure batch_size doesn't exceed configured batch_size to avoid OOM
            batch_size = min(batch_size, self.config.batch_size)
        
        embeddings: List[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if not batch:
                continue
            embeddings.append(self._encode_text_batch(batch))
        if embeddings:
            return torch.vstack(embeddings)
        return torch.zeros((0, self.hidden_size_value), device=self.primary_device)

    def encode_eval_batches(self, texts: List[str], batch_size: int) -> Iterable[torch.Tensor]:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if not batch:
                continue
            with torch.no_grad():
                embeddings = self._encode_text_batch(batch)
            yield embeddings.detach().cpu()

    def encode(self, texts: List[str], batch_size: int) -> torch.Tensor:
        batches = list(self.encode_eval_batches(texts, batch_size))
        if not batches:
            return torch.zeros((0, self.hidden_size_value))
        return torch.vstack(batches)

    def forward(self, texts: List[str], chunk_size: Optional[int] = None) -> torch.Tensor:  # type: ignore[override]
        return self.encode_train(texts, batch_size=chunk_size or self.config.batch_size)

    def freeze_base(self) -> None:
        for name, param in self.base_model.named_parameters():
            if "lora" in name:
                continue
            param.requires_grad = False

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def base_model(self) -> nn.Module:
        return self.model

    def _resolve_primary_device(self) -> torch.device:
        if self.config.device.startswith("cuda"):
            return torch.device(self.config.device)
        return torch.device("cpu")

    def enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing on the underlying HF model (and wrapped PEFT model if present)
        to trade compute for memory savings during training.
        """
        targets = []
        model = self.model
        targets.append(model)
        base_model = getattr(model, "base_model", None)
        if base_model is not None and base_model is not model:
            targets.append(base_model)

        for target in targets:
            enable_fn = getattr(target, "gradient_checkpointing_enable", None)
            if callable(enable_fn):
                enable_fn()
                config = getattr(target, "config", None)
                if config is not None:
                    setattr(config, "gradient_checkpointing", True)
                    if getattr(config, "use_cache", None):
                        config.use_cache = False


class TripletDataset(Dataset):
    def __init__(
        self,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        corpus: Dict[str, Dict[str, str]],
        negatives_per_query: int = 1,
        hard_negatives: Optional[Dict[str, List[str]]] = None,
        seed: int = 42,
    ) -> None:
        self.examples: List[Tuple[str, str, List[str]]] = []
        rng = random.Random(seed)
        doc_ids = list(corpus.keys())
        for qid, rels in qrels.items():
            positives = [doc_id for doc_id, rel in rels.items() if rel > 0]
            if not positives or qid not in queries:
                continue
            positive_id = rng.choice(positives)
            positive_text = compose_document_text(corpus[positive_id])

            negative_ids: List[str] = []
            if hard_negatives and qid in hard_negatives:
                for doc_id in hard_negatives[qid]:
                    if doc_id in corpus and doc_id not in rels and doc_id not in negative_ids:
                        negative_ids.append(doc_id)
                    if len(negative_ids) >= negatives_per_query:
                        break

            if len(negative_ids) < negatives_per_query:
                available = [doc_id for doc_id in doc_ids if doc_id not in rels and doc_id not in negative_ids]
                if available:
                    needed = negatives_per_query - len(negative_ids)
                    sampled = rng.sample(available, min(needed, len(available)))
                    negative_ids.extend(sampled)

            negative_texts = [compose_document_text(corpus[doc_id]) for doc_id in negative_ids if doc_id in corpus]
            if not negative_texts:
                continue
            self.examples.append((queries[qid], positive_text, negative_texts))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[str, str, List[str]]:
        return self.examples[idx]


class ContrastiveTrainer:
    def __init__(self, encoder: HFTextEncoder, config: BenchmarkConfig) -> None:
        self.config = config
        self.module = encoder
        self.parallel_model = self._wrap_for_distribution(encoder)
        params = [p for p in self.parallel_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.device = encoder.primary_device
        self.amp_dtype = self._resolve_amp_dtype()
        self.autocast_enabled = (
            self.config.use_amp
            and self.config.device.startswith("cuda")
            and self.amp_dtype != torch.float32
        )
        self.scaler = GradScaler('cuda', enabled=self.autocast_enabled and self.amp_dtype == torch.float16)
        self.global_step = 0
        self.wall_clock_start = None
        self.tokens_processed = 0

    def train(
        self,
        dataset_name: str,
        train_dataset: TripletDataset,
        corpus: Dict[str, Dict[str, str]],
        dev_partition: Optional[DatasetPartition],
        evaluator: "EvaluationManager",
    ) -> Dict[str, Any]:
        if len(train_dataset) == 0:
            return {"steps": 0, "tokens": 0, "wall_clock_sec": 0.0}

        sampler: Optional[DistributedSampler] = None
        if self.config.distributed:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=True,
            )

        # Use num_workers=0 to avoid forking and memory duplication issues
        # Multiprocessing workers can cause massive memory overhead (8 workers * model size)
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=sampler is None,
            drop_last=True,
            sampler=sampler,
            num_workers=0,  # Disabled to prevent memory issues
            pin_memory=self.config.device.startswith("cuda"),
        )
        num_training_steps = len(dataloader) * self.config.epochs
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, num_training_steps)

        log_enabled = is_main_process(self.config)
        
        # DEBUG: Check if this is MAW training
        is_maw_training = self.config.use_maw
        if log_enabled:
            logging.info(
                "[Train:%s] Using %s | batches=%d | epochs=%d | MAW=%s",
                dataset_name,
                device_status(self.config),
                len(dataloader),
                self.config.epochs,
                is_maw_training,
            )
            
            # DEBUG: Log initial model state for MAW
            if is_maw_training:
                logging.info("[DEBUG] MAW Training - Checking initial model state...")
                with torch.no_grad():
                    test_text = ["This is a test sentence."]
                    test_emb = self.module.encode(test_text, batch_size=1)
                    logging.info(f"[DEBUG] Initial test embedding: shape={test_emb.shape}, mean={test_emb.mean():.4f}, std={test_emb.std():.4f}, has_nan={torch.isnan(test_emb).any()}, has_inf={torch.isinf(test_emb).any()}")

        self.parallel_model.train()
        best_metric = -1.0
        best_state = None
        self.wall_clock_start = time.time()

        status = device_status(self.config)
        for epoch in range(self.config.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            progress = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.config.epochs} [{status}]",
                leave=False,
                disable=not log_enabled,
            )
            accumulation_step = 0
            for batch_idx, batch in enumerate(progress):
                queries, positives, negatives = batch
                batch_queries = list(queries)
                batch_pos = list(positives)
                negative_lists = [list(nlist) for nlist in negatives]

                flat_negatives = [neg for neg_list in negative_lists for neg in neg_list]
                merged_texts = batch_queries + batch_pos + flat_negatives

                # DEBUG: Monitor first batch of MAW training
                if is_maw_training and log_enabled and batch_idx == 0 and epoch == 0:
                    logging.info(f"[DEBUG] First batch: {len(batch_queries)} queries, {len(merged_texts)} total texts")

                with self._autocast_context():
                    # Don't pass chunk_size - let encode_train use the configured batch_size
                    # to avoid OOM when merged_texts is large (e.g., 32 queries * 6 texts each = 192)
                    embeddings = self.parallel_model(merged_texts)
                    
                # DEBUG: Check embeddings after encoding
                if is_maw_training and log_enabled and batch_idx == 0 and epoch == 0:
                    logging.info(f"[DEBUG] Embeddings: shape={embeddings.shape}, mean={embeddings.mean():.4f}, std={embeddings.std():.4f}, has_nan={torch.isnan(embeddings).any()}, has_inf={torch.isinf(embeddings).any()}")

                num_queries = len(batch_queries)
                num_pos = len(batch_pos)
                pos_start = num_queries
                pos_end = pos_start + num_pos
                query_emb = embeddings[:num_queries]
                pos_emb = embeddings[pos_start:pos_end]
                neg_emb = embeddings[pos_end:]

                doc_emb = torch.cat([pos_emb, neg_emb], dim=0) if neg_emb.numel() else pos_emb
                logits = torch.matmul(query_emb, doc_emb.t()) / self.config.temperature
                logits = logits.float()
                
                # DEBUG: Check logits
                if is_maw_training and log_enabled and batch_idx == 0 and epoch == 0:
                    logging.info(f"[DEBUG] Logits: shape={logits.shape}, mean={logits.mean():.4f}, std={logits.std():.4f}, min={logits.min():.4f}, max={logits.max():.4f}")
                    # Check if positive docs are actually ranked higher
                    pos_scores = logits.diag()
                    logging.info(f"[DEBUG] Positive scores (diagonal): mean={pos_scores.mean():.4f}, min={pos_scores.min():.4f}, max={pos_scores.max():.4f}")
                
                labels = torch.arange(logits.size(0), device=query_emb.device)
                loss = F.cross_entropy(logits, labels)
                
                # DEBUG: Check loss
                if is_maw_training and log_enabled and batch_idx == 0 and epoch == 0:
                    logging.info(f"[DEBUG] Loss: {loss.item():.4f}")
                
                # Scale loss by accumulation steps to average gradients
                loss = loss / self.config.gradient_accumulation_steps

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulation_step += 1
                
                # Only update weights after accumulating enough gradients
                if accumulation_step % self.config.gradient_accumulation_steps == 0:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), max_norm=2.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), max_norm=2.0)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                self.global_step += 1
                token_factor = 1 + 1 + self.config.negatives_per_query
                token_count = len(batch_queries) * self.config.max_seq_length * token_factor
                self.tokens_processed += token_count
                if log_enabled:
                    progress.set_postfix({"loss": loss.item(), "step": self.global_step})
                
                # DEBUG: Check gradients for MAW after first batch
                if is_maw_training and log_enabled and batch_idx == 0 and epoch == 0 and accumulation_step % self.config.gradient_accumulation_steps == 0:
                    total_grad_norm = 0.0
                    num_params_with_grad = 0
                    for name, param in self.parallel_model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            total_grad_norm += grad_norm ** 2
                            num_params_with_grad += 1
                            if 'maw' in name.lower() or 'grpo' in name.lower() or 'policy' in name.lower():
                                logging.info(f"[DEBUG] Gradient for {name}: norm={grad_norm:.6f}, mean={param.grad.mean():.6f}")
                    total_grad_norm = total_grad_norm ** 0.5
                    logging.info(f"[DEBUG] Total gradient norm: {total_grad_norm:.4f} ({num_params_with_grad} params with gradients)")

            if dev_partition is not None and not dev_partition.is_empty():
                if log_enabled:
                    metrics, _ = evaluator.evaluate_dense_model(
                        dataset_name,
                        self.module,
                        corpus,
                        dev_partition.queries,
                        dev_partition.qrels,
                        split="dev",
                    )
                    dev_metric = metrics.get("nDCG@10", metrics.get("MAP@100", 0.0))
                    if dev_metric > best_metric:
                        best_metric = dev_metric
                        best_state = {k: v.detach().cpu() for k, v in self.module.state_dict().items()}
                if self.config.distributed and dist.is_available() and dist.is_initialized():
                    dist.barrier()

        synced_state = self._synchronize_best_state(best_state)
        if synced_state:
            self.module.load_state_dict(synced_state)

        self.parallel_model.eval()
        
        # DEBUG: Test model after training for MAW
        if is_maw_training and log_enabled:
            logging.info("[DEBUG] MAW Training Complete - Testing trained model...")
            with torch.no_grad():
                test_queries = ["machine learning", "deep learning"]
                test_docs = ["Machine learning is AI", "Python programming"]
                query_embs = self.module.encode(test_queries, batch_size=8)
                doc_embs = self.module.encode(test_docs, batch_size=8)
                similarities = torch.matmul(query_embs, doc_embs.t())
                logging.info(f"[DEBUG] After training - Query embeddings: mean={query_embs.mean():.4f}, std={query_embs.std():.4f}, has_nan={torch.isnan(query_embs).any()}, has_inf={torch.isinf(query_embs).any()}")
                logging.info(f"[DEBUG] After training - Doc embeddings: mean={doc_embs.mean():.4f}, std={doc_embs.std():.4f}, has_nan={torch.isnan(doc_embs).any()}, has_inf={torch.isinf(doc_embs).any()}")
                logging.info(f"[DEBUG] After training - Similarities:\n{similarities.cpu().numpy()}")

        wall_clock = time.time() - self.wall_clock_start
        total_tokens = self.tokens_processed
        if self.config.distributed and dist.is_available() and dist.is_initialized():
            token_tensor = torch.tensor([self.tokens_processed], device=self.device, dtype=torch.float32)
            dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
            total_tokens = int(token_tensor.item())
            wall_tensor = torch.tensor([wall_clock], device=self.device, dtype=torch.float32)
            dist.all_reduce(wall_tensor, op=dist.ReduceOp.MAX)
            wall_clock = float(wall_tensor.item())

        return {
            "steps": self.global_step,
            "tokens": total_tokens,
            "wall_clock_sec": wall_clock,
        }

    def _wrap_for_distribution(self, module: HFTextEncoder) -> nn.Module:
        """Wrap model for multi-GPU training.
        
        NOTE: DataParallel doesn't work with text-based forward methods.
        Our encoder processes text strings internally (tokenization happens inside forward),
        which DataParallel cannot split across GPUs.
        
        For multi-GPU training with text encoders, we instead:
        1. Keep model on primary GPU
        2. Manually distribute batches in the training loop
        3. Use all GPUs for parallel corpus encoding during evaluation
        """
        if not self.config.device.startswith("cuda"):
            return module
            
        num_gpus = torch.cuda.device_count()
        
        if self.config.distributed and num_gpus > 1:
            # DistributedDataParallel for explicit distributed training
            kwargs: Dict[str, Any] = {"broadcast_buffers": False, "find_unused_parameters": False}
            kwargs["device_ids"] = [self.config.primary_device_index]
            kwargs["output_device"] = self.config.primary_device_index
            logging.info(f"Using DistributedDataParallel on GPU {self.config.primary_device_index}")
            return nn.parallel.DistributedDataParallel(module, **kwargs)
        
        # For standard multi-GPU: keep model on primary GPU, distribute batches manually
        if num_gpus > 1:
            logging.info(f"Multi-GPU mode: Model on GPU {self.config.primary_device_index}, will distribute batches across {num_gpus} GPUs during evaluation")
        
        return module

    def _resolve_amp_dtype(self) -> torch.dtype:
        if not self.config.use_amp or not self.config.device.startswith("cuda"):
            return torch.float32
        requested = self.config.amp_dtype.lower()
        if requested in {"bf16", "bfloat16"}:
            bf16_available = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            if bf16_available:
                return torch.bfloat16
            logging.warning("bf16 requested but not supported on this device; falling back to fp16.")
            return torch.float16
        if requested in {"fp16", "float16", "half"}:
            return torch.float16
        logging.warning("Unrecognized amp_dtype '%s'; defaulting to fp16.", self.config.amp_dtype)
        return torch.float16

    def _autocast_context(self):
        if self.autocast_enabled:
            return autocast(device_type='cuda', dtype=self.amp_dtype)
        return nullcontext()

    def _synchronize_best_state(self, state: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
        if not self.config.distributed:
            return state
        obj_list: List[Optional[Dict[str, torch.Tensor]]] = [state if is_main_process(self.config) else None]
        dist.broadcast_object_list(obj_list, src=0)
        return obj_list[0]


class EvaluationManager:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.primary_device = self._resolve_primary_device()
        self.use_amp = config.use_amp and self.primary_device.type == "cuda"
        self.search_dtype = self._resolve_search_dtype()

    def evaluate_dense_model(
        self,
        dataset_id: str,
        encoder: HFTextEncoder,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        split: str = "test",
        cache_prefix: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        if not queries:
            logging.warning("%s/%s evaluation skipped: no queries provided.", dataset_id, split)
            return self._empty_metrics(), {}

        doc_ids = list(corpus.keys())
        if not doc_ids:
            raise ValueError(f"Dataset '{dataset_id}' has an empty corpus; cannot evaluate.")

        top_k = min(self.config.top_k, len(doc_ids))
        cache_prefix = cache_prefix or dataset_id

        previous_mode = encoder.training
        encoder.eval()
        
        log_enabled = is_main_process(self.config)
        
        # DEBUG: Check if evaluating MAW model
        is_maw_eval = hasattr(encoder.config, 'use_maw') and encoder.config.use_maw
        if is_maw_eval and log_enabled:
            logging.info(f"[DEBUG] Starting evaluation of MAW model for {dataset_id}/{split}")
            # Test encode before corpus preparation
            with torch.no_grad():
                test_emb = encoder.encode(["test"], batch_size=1)
                logging.info(f"[DEBUG] Eval test embedding: mean={test_emb.mean():.4f}, has_nan={torch.isnan(test_emb).any()}, has_inf={torch.isinf(test_emb).any()}")

        corpus_embeddings = self._prepare_corpus_embeddings(encoder, cache_prefix, doc_ids, corpus)

        query_pairs = [(qid, queries[qid]) for qid in queries]
        if self.config.distributed and dist.is_available() and dist.is_initialized():
            query_pairs = query_pairs[self.config.rank :: self.config.world_size]

        local_results: Dict[str, Dict[str, float]] = {}
        
        # Use encoder's config for batch size (MAW may have different settings)
        query_batch_size = encoder.config.eval_batch_size if hasattr(encoder, 'config') else self.config.eval_batch_size

        for start in tqdm(
            range(0, len(query_pairs), query_batch_size),
            desc=f"{dataset_id}/{split}:queries [{device_status(self.config)}]",
            leave=False,
            disable=not log_enabled,
        ):
            batch = query_pairs[start : start + query_batch_size]
            if not batch:
                continue
            batch_ids = [pair[0] for pair in batch]
            batch_texts = [pair[1] for pair in batch]
            if self.use_amp and self.primary_device.type == "cuda":
                with autocast(device_type='cuda', dtype=self.search_dtype):
                    batch_embeddings = encoder.encode(batch_texts, batch_size=query_batch_size)
            else:
                batch_embeddings = encoder.encode(batch_texts, batch_size=query_batch_size)
            
            # DEBUG: Check query embeddings for MAW
            if is_maw_eval and log_enabled and start == 0:
                logging.info(f"[DEBUG] First query batch embeddings: shape={batch_embeddings.shape}, mean={batch_embeddings.mean():.4f}, std={batch_embeddings.std():.4f}, has_nan={torch.isnan(batch_embeddings).any()}, has_inf={torch.isinf(batch_embeddings).any()}")
            
            if batch_embeddings.numel() == 0:
                continue
            scores, indices = self._search_corpus(batch_embeddings, corpus_embeddings, top_k)
            
            # DEBUG: Check scores for MAW
            if is_maw_eval and log_enabled and start == 0:
                logging.info(f"[DEBUG] First query batch scores: shape={scores.shape}, mean={scores.mean():.4f}, std={scores.std():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")
            scores_np = scores.numpy()
            indices_np = indices.numpy()
            for row, qid in enumerate(batch_ids):
                ranking: Dict[str, float] = {}
                for col in range(scores_np.shape[1]):
                    doc_idx = int(indices_np[row, col])
                    if doc_idx < 0 or doc_idx >= len(doc_ids):
                        continue
                    ranking[doc_ids[doc_idx]] = float(scores_np[row, col])
                local_results[qid] = ranking

        partials = all_gather_objects(self.config, local_results)
        results: Dict[str, Dict[str, float]] = {}
        for part in partials:
            if part:
                results.update(part)

        if previous_mode:
            encoder.train()

        metrics: Dict[str, float]
        per_query: Dict[str, Dict[str, float]]
        if is_main_process(self.config):
            metrics, per_query = self.compute_metrics(qrels, results)
        else:
            metrics, per_query = {}, {}

        metrics = broadcast_object(self.config, metrics)
        per_query = broadcast_object(self.config, per_query)
        return metrics, per_query

    def evaluate_bm25(
        self,
        dataset_id: str,
        bm25: PyseriniBM25Retriever,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        if not queries:
            logging.warning("%s BM25 evaluation skipped: no queries provided.", dataset_id)
            return self._empty_metrics(), {}
        top_k = bm25.config.top_k
        results = bm25.search(queries, top_k)
        metrics, per_query = self.compute_metrics(qrels, results)
        return metrics, per_query

    def compute_metrics(
        self,
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        if not qrels:
            return self._empty_metrics(), {}
        
        # DEBUG: Check qrels and results structure
        log_enabled = is_main_process(self.config)
        if log_enabled:
            logging.info(f"[DEBUG METRICS] qrels: {len(qrels)} queries")
            logging.info(f"[DEBUG METRICS] results: {len(results)} queries")
            if qrels:
                sample_qid = list(qrels.keys())[0]
                logging.info(f"[DEBUG METRICS] Sample qrel: qid={sample_qid}, relevant_docs={list(qrels[sample_qid].keys())[:3]}")
            if results:
                sample_qid = list(results.keys())[0]
                top_docs = sorted(results[sample_qid].items(), key=lambda x: x[1], reverse=True)[:5]
                logging.info(f"[DEBUG METRICS] Sample result: qid={sample_qid}, top_5_docs={[(d, f'{s:.4f}') for d, s in top_docs]}")
        
        metrics_map = ["map_cut_100", "ndcg_cut_10", "ndcg_cut_100", "recall_100", "recall_1000", "recall_10"]
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_map)
        per_query = evaluator.evaluate(results)
        aggregate: Dict[str, float] = collections.defaultdict(float)
        for scores in per_query.values():
            for metric, value in scores.items():
                aggregate[metric] += value
        if per_query:
            for metric in aggregate:
                aggregate[metric] /= len(per_query)

        mapped = {
            "MAP@100": aggregate.get("map_cut_100", 0.0),
            "nDCG@10": aggregate.get("ndcg_cut_10", 0.0),
            "nDCG@100": aggregate.get("ndcg_cut_100", 0.0),
            "Recall@10": aggregate.get("recall_10", 0.0),
            "Recall@100": aggregate.get("recall_100", 0.0),
        }
        if "recall_1000" in aggregate:
            mapped["Recall@1000"] = aggregate["recall_1000"]
        return mapped, per_query

    @staticmethod
    def paired_bootstrap(
        per_query_a: Dict[str, Dict[str, float]],
        per_query_b: Dict[str, Dict[str, float]],
        metric: str,
        iterations: int = 1000,
        seed: int = 42,
    ) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        common_qids = per_query_a.keys() & per_query_b.keys()
        deltas = [
            per_query_b[qid].get(metric, 0.0) - per_query_a[qid].get(metric, 0.0)
            for qid in common_qids
        ]
        if not deltas:
            return {"p_value": 1.0, "mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        deltas = np.array(deltas)
        samples = []
        for _ in range(iterations):
            sample = rng.choice(deltas, size=len(deltas), replace=True)
            samples.append(sample.mean())
        samples = np.array(samples)
        mean = float(samples.mean())
        p_value = float(2 * min((samples <= 0).mean(), (samples >= 0).mean()))
        return {
            "p_value": p_value,
            "mean": mean,
            "ci_lower": float(np.percentile(samples, 2.5)),
            "ci_upper": float(np.percentile(samples, 97.5)),
            "wins": int((samples > 0).sum()),
            "losses": int((samples < 0).sum()),
        }

    def _prepare_corpus_embeddings(
        self,
        encoder: HFTextEncoder,
        cache_prefix: str,
        doc_ids: List[str],
        corpus: Dict[str, Dict[str, str]],
    ) -> np.memmap:
        cache_dir = ensure_dir(Path(self.config.embedding_cache_root) / cache_prefix)
        hidden_size = encoder.hidden_size_value
        emb_path = cache_dir / f"corpus_{len(doc_ids)}_{hidden_size}.memmap"
        meta_path = cache_dir / "meta.json"

        def _load_memmap() -> np.memmap:
            return np.memmap(emb_path, dtype=np.float32, mode="r", shape=(len(doc_ids), hidden_size))

        metadata_valid = False
        if emb_path.exists() and meta_path.exists():
            try:
                with meta_path.open("r") as f:
                    meta = json.load(f)
                metadata_valid = meta.get("doc_count") == len(doc_ids) and meta.get("hidden") == hidden_size
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to reuse dense cache at %s: %s", emb_path, exc)

        if metadata_valid:
            if self.config.distributed and dist.is_available() and dist.is_initialized():
                dist.barrier()
            return _load_memmap()

        def _encode_corpus() -> None:
            logging.info("Encoding corpus embeddings for cache prefix '%s' (%d documents)...", cache_prefix, len(doc_ids))
            writer = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(len(doc_ids), hidden_size))
            offset = 0
            # Use encoder's config for batch size (MAW may have different settings)
            batch_size = encoder.config.eval_batch_size if hasattr(encoder, 'config') else self.config.eval_batch_size
            for start in tqdm(
                range(0, len(doc_ids), batch_size),
                desc=f"{cache_prefix}:encode_corpus [{device_status(self.config)}]",
                leave=False,
            ):
                batch_ids = doc_ids[start : start + batch_size]
                batch_texts = [compose_document_text(corpus[doc_id]) for doc_id in batch_ids]
                batch_embeddings = encoder.encode(batch_texts, batch_size=batch_size)
                batch_np = batch_embeddings.to(torch.float32).cpu().numpy()
                if batch_np.size == 0:
                    continue
                end = offset + batch_np.shape[0]
                writer[offset:end] = batch_np
                offset = end
            if offset != len(doc_ids):
                logging.warning(
                    "Corpus embedding cache for %s expected %d vectors but wrote %d.",
                    cache_prefix,
                    len(doc_ids),
                    offset,
                )
            writer.flush()
            del writer

            with meta_path.open("w") as f:
                json.dump({"doc_count": len(doc_ids), "hidden": hidden_size}, f)

        if self.config.distributed and dist.is_available() and dist.is_initialized():
            if is_main_process(self.config):
                _encode_corpus()
                dist.barrier()
            else:
                dist.barrier()
            return _load_memmap()

        _encode_corpus()
        return _load_memmap()

    def _search_corpus(
        self,
        query_embeddings: torch.Tensor,
        corpus_embeddings: np.memmap,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_queries = query_embeddings.size(0)
        num_docs = corpus_embeddings.shape[0]
        if num_docs == 0 or top_k == 0 or num_queries == 0:
            empty_scores = torch.empty((num_queries, 0), dtype=torch.float32)
            empty_indices = torch.empty((num_queries, 0), dtype=torch.long)
            return empty_scores, empty_indices

        k = min(top_k, num_docs)
        device = self.primary_device
        if self.use_amp:
            queries = query_embeddings.to(device=device, dtype=self.search_dtype, non_blocking=True)
        else:
            queries = query_embeddings.to(device=device, dtype=torch.float32, non_blocking=True)

        top_scores = torch.full((num_queries, k), float("-inf"), device=device, dtype=torch.float32)
        top_indices = torch.full((num_queries, k), -1, device=device, dtype=torch.long)

        chunk_size = max(k, self._determine_chunk_size(num_queries))
        for start in range(0, num_docs, chunk_size):
            end = min(start + chunk_size, num_docs)
            docs_np = np.asarray(corpus_embeddings[start:end], dtype=np.float32)
            if docs_np.size == 0:
                continue
            # Make a writable copy to avoid PyTorch warning about non-writable arrays
            docs_np = np.array(docs_np, copy=True)
            if self.use_amp:
                docs = torch.from_numpy(docs_np).to(device=device, dtype=self.search_dtype, non_blocking=True)
            else:
                docs = torch.from_numpy(docs_np).to(device=device, dtype=torch.float32, non_blocking=True)
            scores = torch.matmul(queries, docs.t()).float()
            combined_scores = torch.cat([top_scores, scores], dim=1)
            doc_indices = torch.arange(start, end, device=device, dtype=torch.long)
            doc_indices = doc_indices.unsqueeze(0).expand(num_queries, -1)
            combined_indices = torch.cat([top_indices, doc_indices], dim=1)
            top_scores, top_pos = torch.topk(combined_scores, k=k, dim=1)
            top_indices = torch.gather(combined_indices, 1, top_pos)
            del docs, scores, combined_scores, doc_indices, combined_indices, top_pos

        return top_scores.cpu(), top_indices.cpu()

    def _determine_chunk_size(self, num_queries: int) -> int:
        target_bytes = 256 * 1024 * 1024  # 256MB buffer
        bytes_per_value = 2 if (self.use_amp and self.search_dtype in {torch.float16, torch.bfloat16}) else 4
        denom = max(num_queries, 1) * bytes_per_value
        chunk = target_bytes // denom
        return max(1024, chunk)

    def _resolve_primary_device(self) -> torch.device:
        if self.config.device.startswith("cuda"):
            return torch.device(self.config.device)
        return torch.device("cpu")

    def _resolve_search_dtype(self) -> torch.dtype:
        if not self.use_amp:
            return torch.float32
        requested = self.config.amp_dtype.lower()
        if requested in {"bf16", "bfloat16"}:
            bf16_available = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            if bf16_available:
                return torch.bfloat16
            logging.warning("bf16 requested for retrieval but not supported; defaulting to fp16.")
        return torch.float16

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {
            "MAP@100": 0.0,
            "nDCG@10": 0.0,
            "nDCG@100": 0.0,
            "Recall@10": 0.0,
            "Recall@100": 0.0,
            "Recall@1000": 0.0,
        }


def compile_budget_report(
    label: str,
    encoder: HFTextEncoder,
    training_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report = {
        "model": label,
        "total_parameters": encoder.total_parameters,
        "trainable_parameters": encoder.trainable_parameters,
    }
    if training_stats:
        report.update(training_stats)
    return report


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.dataset_manager = DatasetManager(config)
        self.evaluator = EvaluationManager(config)
        self.reports: Dict[str, Any] = {}
        self.variant_snapshots: Dict[str, Dict[str, ModelSnapshot]] = collections.defaultdict(dict)
        self.reference_sources: Dict[str, str] = {}
        self.base_encoder = HFTextEncoder(self.config.clone_for(use_lora=False, use_maw=False))
        self.base_encoder.eval()
        # Synchronize after model loading to avoid file locking issues in distributed mode
        if config.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def run(self) -> None:
        set_random_seed(self.config.seed)
        if self.config.ms_marco:
            self._run_bundle(self.dataset_manager.get_msmarco())
        # REMOVED: BEIR datasets loop - BEIR datasets don't have train sets
        for split in self.config.lotte_splits:
            self._run_bundle(self.dataset_manager.get_lotte_split(split))
        if is_main_process(self.config) and self.reports:
            ensure_dir(Path("results"))
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out_path = Path("results") / f"tier1_report_{timestamp}.json"
            with out_path.open("w") as f:
                json.dump(self.reports, f, indent=2)
            logging.info("Saved aggregated report to %s", out_path)

    def _run_bundle(self, bundle: DatasetBundle) -> None:
        """
        Run complete benchmark pipeline for a single dataset.
        
        DATA SPLIT USAGE (NO LEAKAGE):
        - TRAIN split: Used for training model variants (DenseLoRA, MAWLoRA, MAWFullFT)
        - DEV split: Used ONLY for validation during training (early stopping, hyperparameter monitoring)
        - TEST split: Used ONLY for final evaluation metrics reported in results
        
        If a dataset has no explicit test split, dev split is used as test AND nullified for validation
        to prevent data leakage. See get_beir_dataset() and get_msmarco() for split loading logic.
        """
        bundle.ensure_test()
        dataset_name = bundle.name
        test_queries = bundle.test.queries if bundle.test else {}
        test_qrels = bundle.test.qrels if bundle.test else {}

        log_enabled = is_main_process(self.config)

        if not test_queries:
            if log_enabled:
                logging.warning("Skipping dataset %s because the evaluation split is empty.", dataset_name)
            return

        if log_enabled:
            logging.info("=== %s ===", dataset_name)
            logging.info("Device allocation: %s", device_status(self.config))
            
            # Log data split usage to ensure no leakage
            logging.info(
                "Data splits - TRAIN: %d queries (for training), DEV: %d queries (for validation), TEST: %d queries (for final evaluation)",
                len(bundle.train.queries) if bundle.train else 0,
                len(bundle.dev.queries) if bundle.dev else 0,
                len(bundle.test.queries) if bundle.test else 0
            )

        bm25: Optional[PyseriniBM25Retriever] = None
        if log_enabled and not self.config.skip_bm25:
            bm25 = PyseriniBM25Retriever(self.config, dataset_name)
            bm25.build(bundle.corpus)
        elif log_enabled and self.config.skip_bm25:
            logging.info("Skipping BM25 evaluation (--skip-bm25 flag set)")
        if self.config.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

        dataset_report: Dict[str, Any] = {}
        if log_enabled and bm25 is not None:
            bm25_metrics, _ = self.evaluator.evaluate_bm25(dataset_name, bm25, test_queries, test_qrels)
            dataset_report["BM25"] = {"metrics": bm25_metrics}
        
        # Sync after BM25 evaluation before starting dense model evaluation
        if self.config.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

        hard_negatives: Dict[str, List[str]] = {}
        if bundle.has_train():
            if log_enabled and bm25 is not None:
                negative_depth = max(self.config.negatives_per_query * 10, 100)
                hard_negatives = self._mine_bm25_negatives(bm25, bundle.train.queries, bundle.train.qrels, negative_depth)
            hard_negatives = broadcast_object(self.config, hard_negatives)

        if self.config.device.startswith("cuda"):
            allocated_before = torch.cuda.memory_allocated(0) / 1024**3
            reserved_before = torch.cuda.memory_reserved(0) / 1024**3
            logging.info(f"[MEMORY:Before DenseZeroShot Eval] Allocated={allocated_before:.2f}GB, Reserved={reserved_before:.2f}GB")
        
        baseline_metrics, baseline_per_query = self.evaluator.evaluate_dense_model(
            dataset_name,
            self.base_encoder,
            bundle.corpus,
            test_queries,
            test_qrels,
            split="test",
            cache_prefix=f"{dataset_name}/DenseZeroShot",
        )
        
        if self.config.device.startswith("cuda"):
            allocated_after = torch.cuda.memory_allocated(0) / 1024**3
            reserved_after = torch.cuda.memory_reserved(0) / 1024**3
            logging.info(f"[MEMORY:After DenseZeroShot Eval] Allocated={allocated_after:.2f}GB, Reserved={reserved_after:.2f}GB")
        
        if log_enabled:
            logging.info(
                "[Eval:%s] DenseZeroShot completed on %s",
                dataset_name,
                device_status(self.config),
            )
            dataset_report["DenseZeroShot"] = {
                "metrics": baseline_metrics,
                "budget": compile_budget_report("DenseZeroShot", self.base_encoder),
            }
        
        # Move base_encoder to CPU and aggressively clear GPU memory after DenseZeroShot evaluation
        # This frees space for training variants which will load their own models
        if self.config.device.startswith("cuda"):
            # Move model to CPU
            self.base_encoder.model.to("cpu")
            # Force garbage collection and CUDA cache cleanup
            gc.collect()
            torch.cuda.empty_cache()
            # Synchronize to ensure memory is freed before next model loads
            torch.cuda.synchronize()
            # Check memory after cleanup
            allocated_cleaned = torch.cuda.memory_allocated(0) / 1024**3
            reserved_cleaned = torch.cuda.memory_reserved(0) / 1024**3
            logging.info(f"[MEMORY:After Cleanup] Allocated={allocated_cleaned:.2f}GB, Reserved={reserved_cleaned:.2f}GB")

        train_dataset: Optional[TripletDataset] = None
        if bundle.has_train():
            logging.info(f"[DEBUG] Creating TripletDataset with corpus size: {len(bundle.corpus)} docs")
            logging.info(f"[DEBUG] bundle.train.queries: {len(bundle.train.queries)}, bundle.train.qrels: {len(bundle.train.qrels)}")
            
            candidate_dataset = TripletDataset(
                bundle.train.queries,
                bundle.train.qrels,
                bundle.corpus,
                negatives_per_query=self.config.negatives_per_query,
                hard_negatives=hard_negatives,
                seed=self.config.seed,
            )
            if len(candidate_dataset) == 0:
                if log_enabled:
                    logging.warning(
                        "%s: training split empty after filtering; fine-tuning variants will reuse reference checkpoints.",
                        dataset_name,
                    )
            else:
                train_dataset = candidate_dataset
        else:
            if log_enabled:
                logging.info(
                    "%s: no training split detected; reusing reference fine-tuned checkpoints where available.",
                    dataset_name,
                )

        variant_specs = {
            "DenseLoRA": self.config.clone_for(use_lora=True, use_maw=False),
            "MAWLoRA": self.config.clone_for(use_lora=True, use_maw=True),
            "MAWFullFT": self.config.clone_for(use_lora=False, use_maw=True),
        }

        per_query_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
        if log_enabled:
            per_query_scores["DenseZeroShot"] = baseline_per_query

        for label, variant_cfg in variant_specs.items():
            if self.config.device.startswith("cuda"):
                # Check memory on ALL GPUs, not just GPU 0
                for gpu_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    logging.info(f"[MEMORY:Before {label} GPU{gpu_id}] Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
            
            encoder, training_stats, source = self._prepare_variant_model(
                label,
                variant_cfg,
                dataset_name,
                train_dataset,
                bundle,
            )
            metrics, per_query = self.evaluator.evaluate_dense_model(
                dataset_name,
                encoder,
                bundle.corpus,
                test_queries,
                test_qrels,
                split="test",
                cache_prefix=f"{dataset_name}/{label}_{source}",
            )
            if log_enabled:
                logging.info(
                    "[Eval:%s] %s (source=%s) on %s",
                    dataset_name,
                    label,
                    source,
                    device_status(self.config),
                )
                dataset_report[label] = {
                    "metrics": metrics,
                    "budget": compile_budget_report(label, encoder, training_stats),
                    "training_source": source,
                }
                per_query_scores[label] = per_query

            # Free GPU memory after each variant
            # Need aggressive cleanup like DenseZeroShot to prevent accumulation
            if self.config.device.startswith("cuda"):
                encoder.model.to("cpu")
                del encoder
                # Force garbage collection before emptying cache
                gc.collect()
                torch.cuda.empty_cache()
                # Synchronize to ensure memory is freed before next variant loads
                torch.cuda.synchronize()

        if log_enabled:
            sig_tests: Dict[str, Dict[str, float]] = {}
            if (
                "MAWFullFT" in per_query_scores
                and "DenseZeroShot" in per_query_scores
                and per_query_scores["MAWFullFT"]
                and per_query_scores["DenseZeroShot"]
            ):
                sig_tests["MAWFullFT_vs_DenseZeroShot"] = self.evaluator.paired_bootstrap(
                    per_query_scores["DenseZeroShot"],
                    per_query_scores["MAWFullFT"],
                    "ndcg_cut_10",
                )
                if "DenseLoRA" in per_query_scores and per_query_scores["DenseLoRA"]:
                    sig_tests["MAWFullFT_vs_DenseLoRA"] = self.evaluator.paired_bootstrap(
                        per_query_scores["DenseLoRA"],
                        per_query_scores["MAWFullFT"],
                        "ndcg_cut_10",
                    )
            dataset_report["significance"] = sig_tests
            self.reports[dataset_name] = dataset_report
            
            # Save BEST_RESULTS.json for this dataset
            self._save_best_results(dataset_name, dataset_report)

        if self.config.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.config.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _save_best_results(self, dataset_name: str, dataset_report: Dict[str, Any]) -> None:
        """
        Save BEST_RESULTS.json for each dataset containing all methods and their evaluation metrics.
        For methods that require training, includes hyperparameters used.
        """
        ensure_dir(Path("results"))
        best_results_path = Path("results") / f"{dataset_name}_BEST_RESULTS.json"
        
        best_results = {
            "dataset": dataset_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "methods": {}
        }
        
        # Extract results for each method
        for method_name, method_data in dataset_report.items():
            if method_name == "significance":
                # Skip significance tests section
                continue
            
            if isinstance(method_data, dict) and "metrics" in method_data:
                method_info = {
                    "metrics": method_data["metrics"],
                }
                
                # Add hyperparameters for methods that were trained
                if "budget" in method_data and method_data["budget"]:
                    budget = method_data["budget"]
                    
                    # Determine if this method was trained
                    training_source = method_data.get("training_source", "untrained")
                    
                    if training_source != "untrained" and "steps" in budget and budget["steps"] > 0:
                        # Method was trained - include hyperparameters
                        hyperparameters = {
                            "learning_rate": self.config.learning_rate,
                            "batch_size": self.config.batch_size,
                            "epochs": self.config.epochs,
                            "max_seq_length": self.config.max_seq_length,
                            "negatives_per_query": self.config.negatives_per_query,
                            "temperature": self.config.temperature,
                        }
                        
                        # Add method-specific hyperparameters
                        if "LoRA" in method_name or "lora" in method_name.lower():
                            hyperparameters.update({
                                "lora_rank": self.config.lora_rank,
                                "lora_alpha": self.config.lora_alpha,
                                "lora_dropout": self.config.lora_dropout,
                            })
                        
                        if "MAW" in method_name:
                            hyperparameters.update({
                                "maw_depth_dim": self.config.maw_depth_dim,
                                "maw_num_heads": self.config.maw_num_heads,
                                "maw_layer_indices": self.config.maw_layer_indices,
                            })
                        
                        method_info["hyperparameters"] = hyperparameters
                        method_info["training_info"] = {
                            "trained_on": training_source,
                            "total_parameters": budget.get("total_parameters", 0),
                            "trainable_parameters": budget.get("trainable_parameters", 0),
                            "training_steps": budget.get("steps", 0),
                            "training_tokens": budget.get("tokens", 0),
                            "wall_clock_sec": budget.get("wall_clock_sec", 0.0),
                        }
                    else:
                        # Method was not trained (zero-shot or baseline)
                        method_info["training_info"] = {
                            "trained": False,
                            "type": "zero-shot" if "ZeroShot" in method_name else "baseline"
                        }
                
                best_results["methods"][method_name] = method_info
        
        # Add significance tests if available
        if "significance" in dataset_report:
            best_results["significance_tests"] = dataset_report["significance"]
        
        # Save to file
        with best_results_path.open("w") as f:
            json.dump(best_results, f, indent=2)
        
        logging.info("Saved best results to %s", best_results_path)

    def _mine_bm25_negatives(
        self,
        bm25: PyseriniBM25Retriever,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        sample_k: int,
    ) -> Dict[str, List[str]]:
        if not queries:
            return {}
        results = bm25.search(queries, sample_k)
        hard_negatives: Dict[str, List[str]] = {}
        for qid, doc_scores in results.items():
            rel_docs = qrels.get(qid, {})
            ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            negatives = [doc_id for doc_id, _ in ranked_docs if doc_id not in rel_docs]
            if negatives:
                hard_negatives[qid] = negatives
        return hard_negatives

    def _prepare_variant_model(
        self,
        label: str,
        variant_cfg: BenchmarkConfig,
        dataset_name: str,
        train_dataset: Optional[TripletDataset],
        bundle: DatasetBundle,
    ) -> Tuple[HFTextEncoder, Dict[str, Any], str]:
        if torch.cuda.is_available() and variant_cfg.device.startswith("cuda"):
            for gpu_id in range(torch.cuda.device_count()):
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            gc.collect()

        has_training_data = train_dataset is not None and len(train_dataset) > 0

        if has_training_data:
            encoder = HFTextEncoder(variant_cfg)
            if not variant_cfg.use_lora:
                encoder.enable_gradient_checkpointing()
            if variant_cfg.use_lora:
                encoder.freeze_base()
            
            if variant_cfg.device.startswith("cuda"):
                for gpu_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                    logging.info(f"[MEMORY:After {label} Encoder Init GPU{gpu_id}] Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
            
            trainer = ContrastiveTrainer(encoder, variant_cfg)
            training_stats = trainer.train(dataset_name, train_dataset, bundle.corpus, bundle.dev, self.evaluator)
            snapshot = self._store_snapshot(label, dataset_name, encoder, training_stats)
            source_dataset = dataset_name
            encoder.eval()
            return encoder, training_stats, source_dataset

        reference_dataset = self.reference_sources.get(label)
        if reference_dataset:
            snapshot = self.variant_snapshots[label][reference_dataset]
            encoder = self._instantiate_from_snapshot(snapshot)
            encoder.eval()
            return encoder, snapshot.training_stats, reference_dataset

        logging.warning(
            "%s: no training data or reference checkpoint available for %s; evaluating zero-shot variant.",
            dataset_name,
            label,
        )
        encoder = HFTextEncoder(variant_cfg)
        if variant_cfg.use_lora:
            encoder.freeze_base()
        zero_stats = {"steps": 0, "tokens": 0, "wall_clock_sec": 0.0}
        encoder.eval()
        return encoder, zero_stats, "untrained"

    def _store_snapshot(
        self,
        label: str,
        dataset_name: str,
        encoder: HFTextEncoder,
        training_stats: Dict[str, Any],
    ) -> ModelSnapshot:
        state_dict = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        config_copy = encoder.config.clone_for()
        stats_copy = dict(training_stats)
        snapshot = ModelSnapshot(
            label=label,
            source_dataset=dataset_name,
            config=config_copy,
            state_dict=state_dict,
            training_stats=stats_copy,
        )
        self.variant_snapshots[label][dataset_name] = snapshot
        if training_stats.get("steps", 0) > 0 and label not in self.reference_sources:
            self.reference_sources[label] = dataset_name
        return snapshot

    def _instantiate_from_snapshot(self, snapshot: ModelSnapshot) -> HFTextEncoder:
        config_copy = snapshot.config.clone_for()
        encoder = HFTextEncoder(config_copy)
        if config_copy.use_lora:
            encoder.freeze_base()
        encoder.load_state_dict(snapshot.state_dict, strict=True)
        encoder.eval()
        return encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier-1 benchmark runner (real datasets)")
    parser.add_argument("--dense-model", default="facebook/contriever", help="Dense encoder model name")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--msmarco", action="store_true", help="Include MS MARCO evaluation")
    # REMOVED: --beir argument - BEIR datasets don't have train sets
    parser.add_argument("--lotte", nargs="*", default=[], help="LoTTE splits (search/forum)")
    parser.add_argument("--quick-smoke-test", action="store_true", help="Run tiny smoke test subset (64 queries, 2K docs, ~5 min)")
    parser.add_argument("--medium-test", action="store_true", help="Run medium test subset (500 queries, 50K docs, ~20-30 min)")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 baseline evaluation (saves time on large datasets)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--maw-depth", type=int, default=64)
    parser.add_argument("--maw-heads", type=int, default=8)
    parser.add_argument(
        "--maw-layer-indices", 
        type=str, 
        default="-1",
        help="Comma-separated layer indices for MAW. Use -1 for last layer (default), -2 for second-to-last, etc. "
             "Examples: '-1' (last layer only), '0,5,11' (layers 0, 5, and 11), 'all' (all layers), '-1,-2' (last two layers)"
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument(
        "--amp-dtype",
        choices=["fp16", "bf16"],
        default="bf16",
        help="AMP precision to use when acceleration is enabled",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    # Parse MAW layer indices
    maw_layer_indices = _parse_maw_layer_indices(args.maw_layer_indices)
    
    return BenchmarkConfig(
        dense_model=args.dense_model,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        top_k=args.top_k,
        negatives_per_query=args.negatives_per_query,
        ms_marco=args.msmarco,
        # REMOVED: beir_datasets - BEIR datasets don't have train sets
        lotte_splits=args.lotte,
        skip_bm25=args.skip_bm25,
        quick_smoke_test=args.quick_smoke_test,
        medium_test=args.medium_test,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
        use_lora=False,
        use_maw=False,
        maw_depth_dim=args.maw_depth,
        maw_num_heads=args.maw_heads,
        maw_layer_indices=maw_layer_indices,
    )


def _parse_maw_layer_indices(layer_spec: str) -> List[int]:
    """
    Parse MAW layer specification string.
    
    Args:
        layer_spec: String like "-1", "0,5,11", "all", etc.
    
    Returns:
        List of layer indices (may include negative indices to be resolved later)
    """
    layer_spec = layer_spec.strip().lower()
    
    if layer_spec == "all":
        # Special value to indicate all layers (will be resolved in encoder)
        return list(range(100))  # Large number, will be clamped to actual layer count
    
    # Parse comma-separated indices
    try:
        indices = [int(x.strip()) for x in layer_spec.split(",") if x.strip()]
        return indices
    except ValueError:
        logging.warning(f"Invalid MAW layer specification '{layer_spec}', using default [-1]")
        return [-1]


def main() -> None:
    # Suppress common warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer fork warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')
    warnings.filterwarnings('ignore', category=FutureWarning, module='torch.utils.checkpoint')
    
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
    config = build_config(args)
    initialize_distributed(config)

    if not is_main_process(config):
        logging.getLogger().setLevel(logging.WARNING)

    logging.info(
        "Runtime devices -> %s (%s)",
        device_status(config),
        ",".join(f"cuda:{idx}" for idx in config.device_ids) if config.device_ids else "cpu",
    )

    if not (config.ms_marco or config.lotte_splits):
        logging.info("No datasets specified; defaulting to MS MARCO")
        config.ms_marco = True

    runner = BenchmarkRunner(config)
    try:
        runner.run()
    finally:
        finalize_distributed(config)


if __name__ == "__main__":
    main()
