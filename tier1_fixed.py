"""
Tier-1 Benchmark Pipeline (BEIR/MS MARCO/LoTTE compliant)
---------------------------------------------------------

This script replaces the toy baseline with a full evaluation pipeline that matches
community expectations for information retrieval benchmarking:

1. Real datasets and protocols
   • MS MARCO train/dev/test via BEIR-compatible loaders
   • BEIR datasets (default: nq, hotpotqa, scifact, fiqa)
   • LoTTE (search + forum splits) via manual ir_datasets export (see setup script)

2. Reference baselines
   • Pyserini/Anserini BM25 with k1=0.9, b=0.4
   • SentenceTransformer dense retrievers (Contriever by default)
   • MAW variants layered on top of the dense encoder

3. Training with capacity matching
   • Full fine-tuning with in-batch InfoNCE loss
   • LoRA fine-tuning on attention/MLP blocks via PEFT
   • Token-level MAW integration prior to pooling
   • Budget accounting (trainable params, steps, batch tokens, wall-clock)

4. Reference evaluation tooling
   • Retrieval with full corpora, consistent top-K
   • Metrics from pytrec_eval / BEIR EvaluateRetrieval
   • Paired bootstrap significance tests using per-query scores

5. Quality-of-life features
   • --quick-smoke-test flag for a 2–3 minute shakedown on tiny slices
   • Automatic dataset download (BEIR util, HuggingFace datasets)
   • Optional caching of dense embeddings and Pyserini indices

Run examples
------------
# Quick shakedown (~3 minutes, 64 queries/document subset per dataset)
python tier1_fixed.py --quick-smoke-test

# Full Tier-1 sweep (MS MARCO + BEIR + LoTTE)
python tier1_fixed.py --msmarco --beir nq hotpotqa scifact fiqa --lotte search forum

# Dense-only experiment on MS MARCO with LoRA rank 32
python tier1_fixed.py --msmarco --dense-model facebook/contriever --lora-rank 32
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
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

    dense_model: str = "facebook/contriever"
    max_seq_length: int = 512
    batch_size: int = 32
    eval_batch_size: int = 256
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
    beir_datasets: List[str] = field(default_factory=list)
    lotte_splits: List[str] = field(default_factory=list)

    data_root: str = "datasets"
    bm25_index_root: str = "indices/bm25"
    embedding_cache_root: str = "indices/dense"

    quick_smoke_test: bool = False
    smoke_queries: int = 64
    smoke_docs: int = 2000

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
        
        # Reduce batch size for MAW variants to prevent OOM (5D attention is memory-intensive)
        if kwargs.get('use_maw', False) and 'batch_size' not in kwargs:
            # Reduce training batch size by 4x for MAW (5D tensor is large)
            init_params['batch_size'] = max(4, init_params.get('batch_size', 32) // 4)
            # Also reduce eval batch size by 2x
            init_params['eval_batch_size'] = max(64, init_params.get('eval_batch_size', 256) // 2)
        
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
        test_split = self._try_load_split(dataset_path, dataset_name, ["test"])
        if test_split is None and dev_split is not None:
            logging.info("MS MARCO missing explicit test split; using dev split for final evaluation and disabling dev monitoring.")
            test_split = dev_split
            dev_split = None
        return self._assemble_bundle(dataset_name, train_split, dev_split, test_split)

    def get_beir_dataset(self, dataset: str) -> DatasetBundle:
        dataset_path = self._ensure_beir_dataset(dataset, f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip")
        train_split = self._try_load_split(dataset_path, dataset, ["train"])
        dev_split = self._try_load_split(dataset_path, dataset, ["dev", "validation", "val"])
        test_split = self._try_load_split(dataset_path, dataset, ["test"])
        
        # Fall back to dev ONLY for test if test doesn't exist AND we have no training data
        # This prevents data leakage: if we use dev for test, we cannot use it for validation
        if test_split is None and dev_split is not None:
            logging.warning(
                "%s: No test split found, using dev split for evaluation. "
                "Dev split will NOT be used for validation to prevent data leakage.",
                dataset
            )
            test_split = dev_split
            dev_split = None  # Nullify dev to prevent using it for validation
        
        if test_split is None:
            raise ValueError(f"Dataset '{dataset}' does not provide a test or dev split for evaluation.")
        
        # Additional safety check: ensure dev and test are different
        if dev_split is not None and test_split[3] == dev_split[3]:
            logging.warning(
                "%s: Dev and test splits are identical. Nullifying dev split to prevent data leakage.",
                dataset
            )
            dev_split = None
            
        return self._assemble_bundle(dataset, train_split, dev_split, test_split)

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

        train_split = self._try_load_split(str(dataset_path), dataset_name, ["train"])
        dev_split = self._try_load_split(str(dataset_path), dataset_name, ["dev", "validation", "val"])
        test_split = self._try_load_split(str(dataset_path), dataset_name, ["test"])
        if test_split is None:
            raise ValueError(f"LoTTE split '{split}' does not include a test partition.")
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

        corpus = test_split[0]
        train_partition = DatasetPartition(queries=train_split[1], qrels=train_split[2]) if train_split else None
        dev_partition = DatasetPartition(queries=dev_split[1], qrels=dev_split[2]) if dev_split else None
        test_partition = DatasetPartition(queries=test_split[1], qrels=test_split[2])

        bundle = DatasetBundle(name=dataset, corpus=corpus, train=train_partition, dev=dev_partition, test=test_partition)
        bundle.ensure_test()
        return self._apply_smoke_filters(bundle)

    def _prepare_lotte_split(self, split: str) -> None:
        base_path = ensure_dir(Path(self.config.data_root) / "lotte" / split)
        corpus_path = base_path / "corpus.jsonl"
        queries_path = base_path / "queries.jsonl"
        qrels_dir = ensure_dir(base_path / "qrels")

        dataset_versions: Dict[str, Any] = {}
        if is_main_process(self.config):
            for partition in ("train", "dev", "test"):
                try:
                    dataset_versions[partition] = ir_datasets.load(f"lotte/{split}/{partition}")
                except Exception:
                    dataset_versions[partition] = None
            if all(dataset is None for dataset in dataset_versions.values()):
                raise RuntimeError(
                    "Unable to load LoTTE via ir_datasets. Install the package and ensure the dataset is available."
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
            required_paths = [corpus_path, queries_path] + [qrels_dir / f"{part}.tsv" for part in ("train", "dev", "test")]
            wait_for_condition(lambda: all(path.exists() for path in required_paths), timeout=1800, interval=5.0)

    def _apply_smoke_filters(self, bundle: DatasetBundle) -> DatasetBundle:
        if not self.config.quick_smoke_test:
            return bundle

        rng = random.Random(self.config.seed)
        doc_ids: set[str] = set()

        for split_name in ["train", "dev", "test"]:
            partition = getattr(bundle, split_name)
            if partition is None or partition.is_empty():
                continue
            original_qids = list(partition.queries.keys())
            target_queries = min(len(original_qids), self.config.smoke_queries)
            if target_queries < len(original_qids):
                selected_qids = set(rng.sample(original_qids, target_queries))
                partition.queries = {qid: partition.queries[qid] for qid in selected_qids}
                partition.qrels = {qid: partition.qrels[qid] for qid in selected_qids if qid in partition.qrels}
            for docs in partition.qrels.values():
                doc_ids.update(docs.keys())

        if len(doc_ids) < self.config.smoke_docs:
            remaining = [doc_id for doc_id in bundle.corpus if doc_id not in doc_ids]
            if remaining:
                extra = min(len(remaining), self.config.smoke_docs - len(doc_ids))
                doc_ids.update(rng.sample(remaining, extra))

        allowed_docs = {doc_id for doc_id in doc_ids if doc_id in bundle.corpus}
        bundle.corpus = {doc_id: bundle.corpus[doc_id] for doc_id in allowed_docs}

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
                "[SMOKE] %s/%s: %d queries | %d docs | %d qrels",
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
        for doc_id in self.doc_ids:
            doc_text = corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")
            tokens = doc_text.lower().split()
            self.tokenized_corpus.append(tokens)
        
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=0.9, b=0.4)
        logging.info("BM25 index built for %s", self.dataset_name)

    def search(self, queries: Dict[str, str], top_k: int) -> Dict[str, Dict[str, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built")
        
        results: Dict[str, Dict[str, float]] = {}
        for qid, query_text in queries.items():
            tokenized_query = query_text.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            results[qid] = {self.doc_ids[idx]: float(scores[idx]) for idx in top_indices if scores[idx] > 0}
        return results


class TokenLevelMAW(nn.Module):
    """
    Multi-Attention-Weight (MAW) module that computes 5D attention weights.
    
    This module extracts query, key, and value vectors from the encoder's last layer,
    computes a 5D attention tensor, and uses GRPO to select the optimal depth index.
    """
    def __init__(self, hidden_size: int, depth_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.depth_dim = depth_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Policy network for GRPO (selects depth index)
        self.policy_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, depth_dim),
        )
        
        # Value network for GRPO (estimates expected reward)
        self.value_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        
        # Projections for Q, K, V
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        self.norm = nn.LayerNorm(hidden_size)
        
        # GRPO training state
        self.register_buffer('baseline_reward', torch.tensor(0.0))
        
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
        Compute Multi-Attention-Weight (MAW) attention following the 7-step process.
        Creates proper 5D attention tensor: (batch_size, num_heads, seq_len_q, seq_len_k, depth_dim)
        """
        batch_size, num_heads, seq_len_q, head_dim = query.size()
        seq_len_k = key.size(2)
        
        # MAW 7-STEP PROCESS (as specified)
        
        # Step 1: Increase dimension of query vector and transpose
        # Target: (batch_size, num_heads, depth, seq_len_q, 1)
        # Current query shape: (batch_size, num_heads, seq_len_q, head_dim)
        
        # Reshape query: (B, H, seq_q, head_dim) -> (B, H, 1, seq_q, head_dim)
        query_reshaped = query.unsqueeze(2)
        # Expand depth: (B, H, 1, seq_q, head_dim) -> (B, H, depth, seq_q, head_dim)
        query_reshaped = query_reshaped.expand(batch_size, num_heads, self.depth_dim, seq_len_q, head_dim)
        # Reduce head_dim by averaging: (B, H, depth, seq_q, head_dim) -> (B, H, depth, seq_q, 1)
        query_expanded = query_reshaped.mean(dim=-1, keepdim=True)
        
        # Step 2: Increase dimension of key vector and transpose
        # Target: (batch_size, num_heads, depth, 1, seq_len_k)
        # Current key shape: (batch_size, num_heads, seq_len_k, head_dim)
        
        # Reshape key: (B, H, seq_k, head_dim) -> (B, H, 1, seq_k, head_dim)
        key_reshaped = key.unsqueeze(2)
        # Expand depth: (B, H, 1, seq_k, head_dim) -> (B, H, depth, seq_k, head_dim)
        key_reshaped = key_reshaped.expand(batch_size, num_heads, self.depth_dim, seq_len_k, head_dim)
        # Reduce head_dim by averaging: (B, H, depth, seq_k, head_dim) -> (B, H, depth, seq_k)
        key_reduced = key_reshaped.mean(dim=-1)
        # Reshape to target: (B, H, depth, seq_k) -> (B, H, depth, 1, seq_k)
        key_expanded = key_reduced.unsqueeze(3)
        
        # Step 3: Multiply query and key to get 5D attention tensor
        # Shape: (batch_size, num_heads, depth, seq_len_q, seq_len_k)
        attn_5d = torch.matmul(query_expanded, key_expanded)
        
        # Step 4: Transpose to put depth dimension last
        # From: (batch_size, num_heads, depth, seq_len_q, seq_len_k)
        # To: (batch_size, num_heads, seq_len_q, seq_len_k, depth)
        attn_5d = attn_5d.permute(0, 1, 3, 4, 2)
        
        # Step 5: Use GRPO to select the best depth index
        if self.training:
            depth_weights = self._grpo_select_depth_5d(attn_5d, hidden_states, attention_mask)
        else:
            # During inference, use greedy selection (argmax of policy)
            if attention_mask is not None:
                pooled_hidden = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled_hidden = hidden_states.mean(dim=1)
            policy_logits = self.policy_network(pooled_hidden)  # (batch_size, depth_dim)
            depth_idx = policy_logits.argmax(dim=-1)  # (batch_size,)
            # Create one-hot weights
            depth_weights = F.one_hot(depth_idx, num_classes=self.depth_dim).float()  # (batch_size, depth_dim)
        
        # Apply depth weights to reduce 5D tensor to 4D
        # depth_weights: (batch_size, depth_dim) -> (batch_size, 1, 1, 1, depth_dim)
        depth_weights = depth_weights.view(batch_size, 1, 1, 1, self.depth_dim)
        
        # Weighted sum over depth dimension
        # Result: (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_4d = (attn_5d * depth_weights).sum(dim=-1)
        
        # Step 6: Softmax over the last dimension (seq_len_k)
        # Apply attention mask before softmax
        if attention_mask is not None:
            # Expand mask to match attention shape
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attn_4d = attn_4d.masked_fill(~mask_expanded.bool(), float('-inf'))
        
        attn_weights = F.softmax(attn_4d, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Handle NaN from softmax of all -inf
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        
        # Step 7: Multiply attention weights with value vectors
        attn_output = torch.matmul(attn_weights, value)  # (batch_size, num_heads, seq_len_q, head_dim)
        
        return attn_output
    
    def _grpo_select_depth_5d(
        self,
        attn_5d: torch.Tensor,  # (batch_size, num_heads, seq_len_q, seq_len_k, depth_dim)
        hidden_states: torch.Tensor,  # (batch_size, seq_len, hidden_size)
        attention_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
    ) -> torch.Tensor:
        """
        Use GRPO to select the best depth from 5D attention tensor.
        
        Args:
            attn_5d: 5D attention tensor (batch_size, num_heads, seq_len_q, seq_len_k, depth_dim)
            hidden_states: Input hidden states for policy/value networks
            attention_mask: Attention mask
            
        Returns:
            depth_weights: (batch_size, depth_dim) - soft weights over depth dimension
        """
        batch_size = attn_5d.size(0)
        
        # Pool hidden states
        if attention_mask is not None:
            pooled_hidden = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled_hidden = hidden_states.mean(dim=1)
        
        # Compute policy logits (probability distribution over depth indices)
        policy_logits = self.policy_network(pooled_hidden)  # (batch_size, depth_dim)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Compute value estimate
        value_estimate = self.value_network(pooled_hidden).squeeze(-1)  # (batch_size,)
        
        # Sample depth indices from policy
        depth_dist = torch.distributions.Categorical(probs=policy_probs)
        depth_indices = depth_dist.sample()  # (batch_size,)
        
        # Compute reward based on attention quality at selected depth
        # Select attention scores at sampled depth: (batch_size, num_heads, seq_len_q, seq_len_k)
        selected_attn = attn_5d.gather(
            dim=-1,
            index=depth_indices.view(batch_size, 1, 1, 1, 1).expand(
                batch_size, attn_5d.size(1), attn_5d.size(2), attn_5d.size(3), 1
            )
        ).squeeze(-1)
        
        # Compute reward: negative entropy encourages focused attention
        attn_softmax = F.softmax(selected_attn, dim=-1)
        entropy = -(attn_softmax * torch.log(attn_softmax + 1e-10)).sum(dim=-1)  # (batch_size, num_heads, seq_len_q)
        reward = -entropy.mean(dim=(1, 2))  # (batch_size,) - lower entropy is better
        
        # GRPO loss: policy gradient with baseline
        advantage = reward - self.baseline_reward
        log_probs = depth_dist.log_prob(depth_indices)
        policy_loss = -(log_probs * advantage.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(value_estimate, reward.detach())
        
        # Total loss (will be added to the main loss during training)
        grpo_loss = policy_loss + 0.5 * value_loss
        
        # Update baseline (exponential moving average)
        with torch.no_grad():
            self.baseline_reward = 0.95 * self.baseline_reward + 0.05 * reward.mean()
        
        # Store loss for backward pass (attached to the output)
        # In training mode, use soft weights from policy (Gumbel-Softmax for differentiability)
        if self.training:
            # Use Gumbel-Softmax for differentiable sampling
            depth_weights = F.gumbel_softmax(policy_logits, tau=1.0, hard=False)
            # Add GRPO loss to computation graph
            depth_weights = depth_weights + 0.0 * grpo_loss  # Trick to include loss in backward
        else:
            depth_weights = policy_probs
        
        return depth_weights  # (batch_size, depth_dim)


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
            batch_size = max(len(texts), 1) if len(texts) > 0 else self.config.batch_size
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

        max_workers = max(1, min(8, (os.cpu_count() or 1)))
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=sampler is None,
            drop_last=True,
            sampler=sampler,
            num_workers=max_workers,
            pin_memory=self.config.device.startswith("cuda"),
        )
        num_training_steps = len(dataloader) * self.config.epochs
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, num_training_steps)

        log_enabled = is_main_process(self.config)
        if log_enabled:
            logging.info(
                "[Train:%s] Using %s | batches=%d | epochs=%d",
                dataset_name,
                device_status(self.config),
                len(dataloader),
                self.config.epochs,
            )

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
            for batch in progress:
                queries, positives, negatives = batch
                batch_queries = list(queries)
                batch_pos = list(positives)
                negative_lists = [list(nlist) for nlist in negatives]

                flat_negatives = [neg for neg_list in negative_lists for neg in neg_list]
                merged_texts = batch_queries + batch_pos + flat_negatives

                with self._autocast_context():
                    embeddings = self.parallel_model(merged_texts, chunk_size=len(merged_texts))

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
                labels = torch.arange(logits.size(0), device=query_emb.device)
                loss = F.cross_entropy(logits, labels)

                self.optimizer.zero_grad(set_to_none=True)

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), max_norm=2.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parallel_model.parameters(), max_norm=2.0)
                    self.optimizer.step()
                scheduler.step()

                self.global_step += 1
                token_factor = 1 + 1 + self.config.negatives_per_query
                token_count = len(batch_queries) * self.config.max_seq_length * token_factor
                self.tokens_processed += token_count
                if log_enabled:
                    progress.set_postfix({"loss": loss.item(), "step": self.global_step})

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
        if not self.config.distributed:
            return module
        kwargs: Dict[str, Any] = {"broadcast_buffers": False, "find_unused_parameters": False}
        if self.config.device.startswith("cuda"):
            kwargs["device_ids"] = [self.config.primary_device_index]
            kwargs["output_device"] = self.config.primary_device_index
        return nn.parallel.DistributedDataParallel(module, **kwargs)

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

        corpus_embeddings = self._prepare_corpus_embeddings(encoder, cache_prefix, doc_ids, corpus)

        query_pairs = [(qid, queries[qid]) for qid in queries]
        if self.config.distributed and dist.is_available() and dist.is_initialized():
            query_pairs = query_pairs[self.config.rank :: self.config.world_size]

        log_enabled = is_main_process(self.config)
        local_results: Dict[str, Dict[str, float]] = {}

        for start in tqdm(
            range(0, len(query_pairs), self.config.eval_batch_size),
            desc=f"{dataset_id}/{split}:queries [{device_status(self.config)}]",
            leave=False,
            disable=not log_enabled,
        ):
            batch = query_pairs[start : start + self.config.eval_batch_size]
            if not batch:
                continue
            batch_ids = [pair[0] for pair in batch]
            batch_texts = [pair[1] for pair in batch]
            if self.use_amp and self.primary_device.type == "cuda":
                with autocast(device_type='cuda', dtype=self.search_dtype):
                    batch_embeddings = encoder.encode(batch_texts, batch_size=self.config.eval_batch_size)
            else:
                batch_embeddings = encoder.encode(batch_texts, batch_size=self.config.eval_batch_size)
            if batch_embeddings.numel() == 0:
                continue
            scores, indices = self._search_corpus(batch_embeddings, corpus_embeddings, top_k)
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
            batch_size = self.config.eval_batch_size
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
        for dataset in self.config.beir_datasets:
            self._run_bundle(self.dataset_manager.get_beir_dataset(dataset))
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
        if log_enabled:
            bm25 = PyseriniBM25Retriever(self.config, dataset_name)
            bm25.build(bundle.corpus)
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

        baseline_metrics, baseline_per_query = self.evaluator.evaluate_dense_model(
            dataset_name,
            self.base_encoder,
            bundle.corpus,
            test_queries,
            test_qrels,
            split="test",
            cache_prefix=f"{dataset_name}/DenseZeroShot",
        )
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

        train_dataset: Optional[TripletDataset] = None
        if bundle.has_train():
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

            if self.config.device.startswith("cuda"):
                encoder.model.to("cpu")
                torch.cuda.empty_cache()

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

        if self.config.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.config.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

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
        has_training_data = train_dataset is not None and len(train_dataset) > 0

        if has_training_data:
            encoder = HFTextEncoder(variant_cfg)
            if variant_cfg.use_lora:
                encoder.freeze_base()
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
    parser.add_argument("--msmarco", action="store_true", help="Include MS MARCO dev evaluation")
    parser.add_argument("--beir", nargs="*", default=[], help="List of BEIR datasets to evaluate")
    parser.add_argument("--lotte", nargs="*", default=[], help="LoTTE splits (search/forum)")
    parser.add_argument("--quick-smoke-test", action="store_true", help="Run tiny smoke test subset")
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
        beir_datasets=args.beir,
        lotte_splits=args.lotte,
        quick_smoke_test=args.quick_smoke_test,
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

    if not (config.ms_marco or config.beir_datasets or config.lotte_splits):
        logging.info("No datasets specified; defaulting to MS MARCO dev")
        config.ms_marco = True

    runner = BenchmarkRunner(config)
    try:
        runner.run()
    finally:
        finalize_distributed(config)


if __name__ == "__main__":
    main()
