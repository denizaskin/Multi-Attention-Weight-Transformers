from __future__ import annotations

import os
# --- HARD disable Apple Malloc Stack Logging for this and all children ---
# If the parent shell or launchctl has MallocStackLogging enabled, we must *unset* it
# before any heavy libs spawn (wandb-core, tokenizers, etc.). Setting "0" is not enough.
if "MallocStackLogging" in os.environ:
    try:
        os.unsetenv("MallocStackLogging")
    except Exception:
        pass
    os.environ.pop("MallocStackLogging", None)
if "MallocStackLoggingNoCompact" in os.environ:
    try:
        os.unsetenv("MallocStackLoggingNoCompact")
    except Exception:
        pass
    os.environ.pop("MallocStackLoggingNoCompact", None)

os.environ["PYTHONMALLOC"] = "malloc"

# Silence W&B console chatter
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")

# --- Filter noisy malloc/wandb-core lines from BOTH stdout and stderr (without touching tqdm) ---
import sys, re
class _FilterStream:
    __slots__ = ("_s", "_patterns")
    def __init__(self, stream, patterns):
        self._s = stream
        self._patterns = patterns
    def write(self, text):
        # Drop lines that include noisy malloc stack logging chatter
        if any(p in text for p in self._patterns):
            return
        return self._s.write(text)
    def flush(self):
        return self._s.flush()
    def isatty(self):
        return getattr(self._s, "isatty", lambda: False)()
    def fileno(self):
        return getattr(self._s, "fileno", lambda: -1)()
    def writelines(self, lines):
        for line in lines:
            self.write(line)

_NOISE_PATTERNS = [
    "MallocStackLogging",
    "stack logs being written",
    "recording malloc and VM allocation stacks",
    "turning off stack logging",
]
sys.stderr = _FilterStream(sys.stderr, _NOISE_PATTERNS)
sys.stdout = _FilterStream(sys.stdout, _NOISE_PATTERNS)

print("Starting script...")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import math
import platform
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, TrainingArguments, Trainer

from datasets import load_dataset
# Optional: use ir_datasets (Python API) for BEIR loading to avoid HF script loaders (datasets>=4)
try:
    import ir_datasets  # pip install ir_datasets
    HAS_IR_DATASETS = True
except Exception:
    HAS_IR_DATASETS = False

try:
    import ir_measures
    from ir_measures import nDCG, MRR, R
    HAS_IR_MEASURES = True
except Exception:
    ir_measures = None
    nDCG = MRR = R = None
    HAS_IR_MEASURES = False

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    stats = None
    HAS_SCIPY = False

from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import numpy as np
import json
import time
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import inspect
import sys
import io
import contextlib
from contextlib import redirect_stdout, redirect_stderr

# --- Monkeypatch subprocess.Popen to silence wandb-core child process ---
import subprocess

_orig_popen = subprocess.Popen
def _quiet_popen(*args, **kwargs):
    # Get command string for inspection
    cmd = kwargs.get("args", args[0] if args else None)
    cmd_str = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
    if cmd_str and "wandb-core" in cmd_str:
        # Silence the wandb-core service process completely
        kwargs.setdefault("stdout", subprocess.DEVNULL)
        kwargs.setdefault("stderr", subprocess.DEVNULL)
    return _orig_popen(*args, **kwargs)
subprocess.Popen = _quiet_popen

# Quiet toggles
QUIET_WANDB = True          # keep W&B completely silent in the terminal
LOG_EVAL_EXAMPLES = False   # disable per-example wandb logging during evaluation (prevents progress bar breaks)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Config
BACKBONE = "mixedbread-ai/mxbai-rerank-xsmall-v1"  # State-of-the-art cross-encoder for reranking
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3  # Short for demo; increase for better results
LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_ACCUM = 2
ENABLE_WANDB = os.environ.get("ENABLE_WANDB", "").lower() in {"1", "true", "yes"}

# LoRA config for efficient fine-tuning
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["query_proj", "key_proj", "value_proj", "dense"]

# MAW hyperparams
HEADS = 8
DEPTH = 8

# Data
NUM_TRAIN = 10000  # Subset for faster training
NUM_TEST = 2000

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if HAS_IR_MEASURES:
    METRIC_MRR10 = MRR @ 10
    METRIC_R1000 = R @ 1000
    METRIC_NDCG10 = nDCG @ 10
else:
    METRIC_MRR10 = METRIC_R1000 = METRIC_NDCG10 = None


@dataclass
class DatasetSpec:
    key: str
    irds_id: str
    doc_irds_id: Optional[str]
    pyserini_prebuilt: Optional[str] = None
    scoreddocs_irds_id: Optional[str] = None
    topk: int = 1000
    run_filename: str = "bm25_top1000.trec"
    run_tag: str = "bm25"
    primary_metrics: Sequence = field(default_factory=tuple)
    secondary_metrics: Sequence = field(default_factory=tuple)
    description: str = ""
    requires_ir_measures: bool = True


RUNS_DIR = Path("runs")


def _metric_list(*metrics):
    return tuple(m for m in metrics if m is not None)


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "MSMARCO/dev-small": DatasetSpec(
        key="MSMARCO/dev-small",
        irds_id="msmarco-passage/dev/small",
        doc_irds_id="msmarco-passage",
        pyserini_prebuilt="msmarco-passage",
        scoreddocs_irds_id="msmarco-passage/dev/small",
        topk=1000,
        primary_metrics=_metric_list(METRIC_MRR10),
        secondary_metrics=_metric_list(METRIC_R1000),
        description="MS MARCO passage dev small (official 6980 queries)",
    ),
    "TREC-DL-2019-passage": DatasetSpec(
        key="TREC-DL-2019-passage",
        irds_id="msmarco-passage/trec-dl-2019/judged",
        doc_irds_id="msmarco-passage",
        pyserini_prebuilt="msmarco-passage",
        scoreddocs_irds_id="msmarco-passage/trec-dl-2019/judged",
        topk=1000,
        primary_metrics=_metric_list(METRIC_NDCG10),
        secondary_metrics=_metric_list(METRIC_MRR10, METRIC_R1000),
        description="TREC Deep Learning 2019 passsage (judged queries)",
    ),
    "TREC-DL-2020-passage": DatasetSpec(
        key="TREC-DL-2020-passage",
        irds_id="msmarco-passage/trec-dl-2020/judged",
        doc_irds_id="msmarco-passage",
        pyserini_prebuilt="msmarco-passage",
        scoreddocs_irds_id="msmarco-passage/trec-dl-2020/judged",
        topk=1000,
        primary_metrics=_metric_list(METRIC_NDCG10),
        secondary_metrics=_metric_list(METRIC_MRR10, METRIC_R1000),
        description="TREC Deep Learning 2020 passsage (judged queries)",
    ),
    "BeIR/scifact": DatasetSpec(
        key="BeIR/scifact",
        irds_id="beir/scifact/test",
        doc_irds_id="beir/scifact",
        pyserini_prebuilt="beir-v1.0.0-scifact",
        scoreddocs_irds_id="beir/scifact/test",
        topk=1000,
        primary_metrics=_metric_list(METRIC_NDCG10),
        description="Scientific fact-checking (BeIR)",
    ),
    "BeIR/trec-covid": DatasetSpec(
        key="BeIR/trec-covid",
        irds_id="beir/trec-covid/test",
        doc_irds_id="beir/trec-covid",
        pyserini_prebuilt="beir-v1.0.0-trec-covid",
        scoreddocs_irds_id="beir/trec-covid/test",
        topk=1000,
        primary_metrics=_metric_list(METRIC_NDCG10),
        description="TREC-COVID (BeIR)",
    ),
    "BeIR/fiqa": DatasetSpec(
        key="BeIR/fiqa",
        irds_id="beir/fiqa/test",
        doc_irds_id="beir/fiqa",
        pyserini_prebuilt="beir-v1.0.0-fiqa",
        scoreddocs_irds_id="beir/fiqa/test",
        topk=1000,
        primary_metrics=_metric_list(METRIC_NDCG10),
        description="Financial QA (BeIR)",
    ),
    "BeIR/nfcorpus": DatasetSpec(
        key="BeIR/nfcorpus",
        irds_id="beir/nfcorpus/test",
        doc_irds_id="beir/nfcorpus",
        pyserini_prebuilt="beir-v1.0.0-nfcorpus",
        scoreddocs_irds_id="beir/nfcorpus/test",
        topk=1000,
        primary_metrics=_metric_list(METRIC_NDCG10),
        description="NFCorpus medical QA (BeIR)",
    ),
    "mteb/scidocs-reranking": DatasetSpec(
        key="mteb/scidocs-reranking",
        irds_id="mteb/scidocs-reranking",  # placeholder; will fallback to HF loader
        doc_irds_id=None,
        pyserini_prebuilt=None,
        scoreddocs_irds_id=None,
        topk=1000,
        primary_metrics=tuple(),
        description="MTEB SciDocs reranking (pairwise) — sanity check only",
        requires_ir_measures=False,
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[dataset_name]
    except KeyError as exc:
        raise KeyError(f"No dataset spec registered for '{dataset_name}'.") from exc


def _require_ir_datasets(dataset_id: str):
    if not HAS_IR_DATASETS:
        raise RuntimeError(
            f"ir_datasets is required to load '{dataset_id}'. "
            "Install with `pip install ir_datasets`."
        )
    return ir_datasets.load(dataset_id)


def _write_run_from_scoreddocs(spec: DatasetSpec, run_path: Path, topk: Optional[int] = None) -> None:
    ds = _require_ir_datasets(spec.scoreddocs_irds_id or spec.irds_id)
    limit = topk or spec.topk
    with run_path.open("w") as fh:
        current_qid = None
        rank = 0
        for scored in ds.scoreddocs_iter():
            qid = str(getattr(scored, "query_id"))
            if current_qid != qid:
                current_qid = qid
                rank = 0
            if rank >= limit:
                continue
            doc_id = str(getattr(scored, "doc_id"))
            score = float(getattr(scored, "score", 0.0))
            rank += 1
            fh.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {spec.run_tag}\n")


def _write_run_with_pyserini(spec: DatasetSpec, run_path: Path, topk: Optional[int] = None, k1: float = 0.9, b: float = 0.4) -> None:
    if spec.pyserini_prebuilt is None:
        raise RuntimeError(f"Dataset '{spec.key}' does not define a Pyserini index.")
    try:
        from pyserini.search.lucene import LuceneSearcher
    except Exception as exc:
        raise RuntimeError(
            "Pyserini is required for BM25 candidate generation. Install with `pip install pyserini`."
        ) from exc

    searcher = LuceneSearcher.from_prebuilt_index(spec.pyserini_prebuilt)
    searcher.set_bm25(k1, b)

    ds_queries = _require_ir_datasets(spec.irds_id)
    limit = topk or spec.topk

    with run_path.open("w") as fh:
        for q in ds_queries.queries_iter():
            qid = str(getattr(q, "query_id"))
            query_text = getattr(q, "text", "")
            if not query_text:
                continue
            hits = searcher.search(query_text, k=limit)
            for rank, hit in enumerate(hits, start=1):
                fh.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} {spec.run_tag}\n")


def ensure_candidate_runfile(dataset_name: str, force: bool = False, topk: Optional[int] = None) -> Path:
    spec = get_dataset_spec(dataset_name)
    run_dir = RUNS_DIR / spec.key
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / spec.run_filename

    if run_path.exists() and not force:
        return run_path

    if spec.scoreddocs_irds_id is not None:
        _write_run_from_scoreddocs(spec, run_path, topk)
    elif spec.pyserini_prebuilt is not None:
        _write_run_with_pyserini(spec, run_path, topk)
    else:
        raise RuntimeError(
            f"Dataset '{dataset_name}' does not provide scoreddocs or a Pyserini index for BM25 pools."
        )
    return run_path


def read_trec_run(run_path: Path, limit: Optional[int] = None) -> Dict[str, List[Tuple[str, float]]]:
    results: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    with run_path.open("r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, doc_id, rank_str, score_str, *_rest = parts
            try:
                rank = int(rank_str)
            except ValueError:
                rank = len(results[qid]) + 1
            if limit is not None and len(results[qid]) >= limit:
                continue
            try:
                score = float(score_str)
            except ValueError:
                score = 0.0
            results[qid].append((doc_id, score, rank))
    for qid, entries in results.items():
        entries.sort(key=lambda tup: tup[2])
    return {qid: [(doc_id, score) for doc_id, score, _ in entries] for qid, entries in results.items()}


def build_candidate_pools(dataset_name: str, force: bool = False) -> Tuple[Dict[str, List[str]], Path]:
    spec = get_dataset_spec(dataset_name)
    run_path = ensure_candidate_runfile(dataset_name, force=force, topk=spec.topk)
    run = read_trec_run(run_path, limit=spec.topk)
    pools = {qid: [doc_id for doc_id, _ in docs] for qid, docs in run.items()}
    return pools, run_path


class DocumentLookup:
    def __init__(self, dataset):
        self._dataset = dataset
        try:
            self._store = dataset.docs_store()
        except Exception:
            self._store = None
        self._cache: Optional[Dict[str, str]] = None

    def _build_cache(self) -> None:
        if self._cache is not None:
            return
        self._cache = {}
        for doc in self._dataset.docs_iter():
            doc_id = str(getattr(doc, "doc_id", ""))
            if not doc_id:
                continue
            text = self._combine(doc)
            if text:
                self._cache[doc_id] = text

    def get(self, doc_id: str) -> Optional[str]:
        if not doc_id:
            return None
        if self._store is not None:
            doc = self._store.get(doc_id)
            if doc is None:
                return None
            return self._combine(doc)
        self._build_cache()
        return self._cache.get(doc_id) if self._cache else None

    def get_many(self, doc_ids: Sequence[str]) -> Dict[str, str]:
        if self._store is not None and hasattr(self._store, "get_many"):
            docs = self._store.get_many(doc_ids)
            collected: Dict[str, str] = {}
            for did in doc_ids:
                doc = docs.get(did) if isinstance(docs, dict) else None
                if doc is None and hasattr(docs, "get"):
                    doc = docs.get(did)
                if doc is None and isinstance(docs, Iterable):
                    # Some implementations return list/iter of docs; fallback to single get
                    doc = self._store.get(did)
                if doc is None:
                    continue
                text = self._combine(doc)
                if text:
                    collected[did] = text
            return collected
        # fallback: fetch individually (cache-enabled for non-store case)
        result: Dict[str, str] = {}
        for did in doc_ids:
            text = self.get(did)
            if text:
                result[did] = text
        return result

    @staticmethod
    def _combine(doc) -> str:
        title = getattr(doc, "title", "") or ""
        text = getattr(doc, "text", "") or ""
        return (title + " " + text).strip() if title else text.strip()


@dataclass
class DatasetResources:
    spec: DatasetSpec
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    doc_lookup: DocumentLookup
    candidate_pools: Dict[str, List[str]]
    run_path: Path


def _load_queries(ds) -> Dict[str, str]:
    queries = {}
    for q in ds.queries_iter():
        qid = str(getattr(q, "query_id", ""))
        text = getattr(q, "text", "") or ""
        if qid and text:
            queries[qid] = text
    return queries


def _load_qrels(ds) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    if not ds.has_qrels():
        return qrels
    for qr in ds.qrels_iter():
        qid = str(getattr(qr, "query_id", ""))
        doc_id = str(getattr(qr, "doc_id", ""))
        if not qid or not doc_id:
            continue
        try:
            rel = int(getattr(qr, "relevance", 0))
        except Exception:
            rel = 0
        qrels[qid][doc_id] = rel
    return qrels


def load_dataset_resources(dataset_name: str, force_pools: bool = False) -> DatasetResources:
    spec = get_dataset_spec(dataset_name)
    if spec.requires_ir_measures and not HAS_IR_MEASURES:
        raise RuntimeError(
            "ir_measures is required for official evaluation metrics. Install with `pip install ir_measures`."
        )

    ds = _require_ir_datasets(spec.irds_id)
    queries = _load_queries(ds)
    qrels = _load_qrels(ds)

    doc_ds = _require_ir_datasets(spec.doc_irds_id or spec.irds_id)
    doc_lookup = DocumentLookup(doc_ds)

    pools, run_path = build_candidate_pools(dataset_name, force=force_pools)

    return DatasetResources(
        spec=spec,
        queries=queries,
        qrels=qrels,
        doc_lookup=doc_lookup,
        candidate_pools=pools,
        run_path=run_path,
    )


@dataclass
class VariantConfig:
    name: str
    use_maw: bool
    maw_strength: float = 0.03
    depth_dim: int = DEPTH
    inject_last_k: int = 1
    gating_mode: str = "stat"
    maw_random_seed: Optional[int] = None
    lora_finetune: bool = False
    lora_learning_rate: float = 1e-4
    lora_epochs: int = 1
    lora_rank: int = LORA_RANK
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    lora_batch_size: int = 8
    lora_max_queries: Optional[int] = 512
    zero_shot: bool = True


@dataclass
class SeedRunResult:
    seed: int
    metrics: Dict[str, float]
    per_query: Dict[str, Dict[str, float]]
    run_path: str
    depth_log: Optional[str]
    scores: Dict[str, List[Tuple[str, float]]]


DEFAULT_SEEDS: Sequence[int] = (13, 17, 29)

DEFAULT_VARIANTS: Sequence[VariantConfig] = (
    VariantConfig(name="non_maw", use_maw=False),
    VariantConfig(name="maw_default", use_maw=True, maw_strength=0.03, depth_dim=DEPTH, inject_last_k=1),
)

ABLATION_VARIANTS: Sequence[VariantConfig] = (
    VariantConfig(name="maw_depth1", use_maw=True, depth_dim=1),
    VariantConfig(name="maw_depth4", use_maw=True, depth_dim=4),
    VariantConfig(name="maw_depth8", use_maw=True, depth_dim=8),
    VariantConfig(name="maw_gating_uniform", use_maw=True, gating_mode="uniform"),
    VariantConfig(name="maw_gating_argmax", use_maw=True, gating_mode="argmax"),
    VariantConfig(name="maw_gating_random", use_maw=True, gating_mode="random"),
    VariantConfig(name="maw_inject_last2", use_maw=True, inject_last_k=2),
    VariantConfig(
        name="maw_lora_finetune",
        use_maw=True,
        lora_finetune=True,
        lora_epochs=2,
        lora_learning_rate=5e-5,
        lora_batch_size=4,
        lora_max_queries=256,
    ),
)

DEV_SWEEP_VARIANTS: Sequence[VariantConfig] = tuple(
    VariantConfig(
        name=f"maw_strength_{strength:.2f}_depth_{depth}",
        use_maw=True,
        maw_strength=strength,
        depth_dim=depth,
    )
    for strength in (0.02, 0.03, 0.05)
    for depth in (4, 8)
)


def unique_variants(*groups: Sequence[VariantConfig]) -> List[VariantConfig]:
    registry: Dict[str, VariantConfig] = {}
    for group in groups:
        for variant in group:
            registry[variant.name] = variant
    return list(registry.values())


def get_git_commit() -> Optional[str]:
    try:
        repo_dir = Path(__file__).resolve().parent
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir)
        return commit.decode().strip()
    except Exception:
        return None


def get_system_info() -> Dict[str, str]:
    info = platform.uname()
    return {
        "system": info.system,
        "node": info.node,
        "release": info.release,
        "version": info.version,
        "machine": info.machine,
        "processor": info.processor,
    }

PRIMARY_BENCHMARKS: Sequence[str] = (
    "MSMARCO/dev-small",
    "TREC-DL-2019-passage",
    "TREC-DL-2020-passage",
)

SECONDARY_DATASETS: Sequence[str] = (
    "BeIR/scifact",
    "BeIR/trec-covid",
    "BeIR/fiqa",
    "BeIR/nfcorpus",
    "mteb/scidocs-reranking",
)


class MAWDepthLogger:
    def __init__(self, path: Path, flush_every: int = 256):
        self.path = path
        self.flush_every = flush_every
        self._buffer: List[Dict[str, object]] = []
        self._current_meta: Optional[Dict[str, object]] = None
        self.enabled = True
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def capture(self, query_id: str, doc_ids: Sequence[str]):
        prev = self._current_meta
        self._current_meta = {
            "query_id": query_id,
            "doc_ids": list(doc_ids),
        }
        try:
            yield
        finally:
            self._current_meta = prev
            if len(self._buffer) >= self.flush_every:
                self.flush()

    def record(self, depth_weights: torch.Tensor, best_idx: torch.Tensor) -> None:
        if not self.enabled or self._current_meta is None:
            return

        weights = depth_weights.detach().cpu()
        best = best_idx.detach().cpu()
        doc_ids = self._current_meta.get("doc_ids", [])
        query_id = self._current_meta.get("query_id")

        for row_idx in range(weights.size(0)):
            w = weights[row_idx].numpy()
            entropy = float(-np.sum(w * np.log(np.clip(w, 1e-9, 1.0))))
            entry = {
                "query_id": query_id,
                "doc_id": doc_ids[row_idx] if row_idx < len(doc_ids) else None,
                "weights": w.tolist(),
                "argmax_depth": int(best[row_idx].item()),
                "entropy": entropy,
                "max_weight": float(w.max()),
                "min_weight": float(w.min()),
            }
            self._buffer.append(entry)

        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        with self.path.open("a") as fh:
            for row in self._buffer:
                fh.write(json.dumps(row) + "\n")
        self._buffer.clear()

    def close(self):
        self.flush()


_ACTIVE_MAW_LOGGER: Optional[MAWDepthLogger] = None


def register_maw_logger(logger: Optional[MAWDepthLogger]) -> None:
    global _ACTIVE_MAW_LOGGER
    _ACTIVE_MAW_LOGGER = logger


def get_active_maw_logger() -> Optional[MAWDepthLogger]:
    return _ACTIVE_MAW_LOGGER


def _tensor_to_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 0:
        return logits.unsqueeze(0)
    if logits.dim() == 1:
        return logits
    if logits.size(-1) == 1:
        return logits.view(-1)
    return logits[..., 0]


def _chunk(sequence: Sequence, size: int):
    for idx in range(0, len(sequence), size):
        yield sequence[idx: idx + size]


def rerank(model, tokenizer, resources: DatasetResources, batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH, depth_logger: Optional[MAWDepthLogger] = None) -> Dict[str, List[Tuple[str, float]]]:
    model.eval()
    scores_by_query: Dict[str, List[Tuple[str, float]]] = {}
    missing_docs = 0
    total_pairs = 0

    pools_items = list(resources.candidate_pools.items())
    use_bar = len(pools_items) > 1
    iterator = tqdm(pools_items, desc="Reranking", unit="query") if use_bar else pools_items

    for qid, doc_ids in iterator:
        query_text = resources.queries.get(qid)
        if not query_text:
            continue

        docs_map = resources.doc_lookup.get_many(doc_ids)
        pairs = []
        for did in doc_ids:
            text = docs_map.get(did)
            if not text:
                missing_docs += 1
                continue
            pairs.append((did, text))

        if not pairs:
            continue

        doc_scores: List[Tuple[str, float]] = []
        for chunk in _chunk(pairs, max(1, batch_size)):
            c_doc_ids = [doc_id for doc_id, _ in chunk]
            passages = [text for _, text in chunk]
            tokenized = tokenizer(
                [query_text] * len(chunk),
                passages,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            with torch.no_grad():
                if depth_logger is not None:
                    with depth_logger.capture(qid, c_doc_ids):
                        outputs = model(**tokenized)
                else:
                    outputs = model(**tokenized)
            scores_tensor = _tensor_to_scores(outputs)
            scores_tensor = scores_tensor.detach().cpu()

            for did, score in zip(c_doc_ids, scores_tensor.tolist()):
                doc_scores.append((did, float(score)))
                total_pairs += 1

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        scores_by_query[qid] = doc_scores

    if use_bar:
        iterator.close()

    if missing_docs:
        print(f"[RERANK] Missing {missing_docs} documents out of {total_pairs} scored pairs.")

    if depth_logger is not None:
        depth_logger.flush()

    return scores_by_query


def write_trec_run(run: Dict[str, List[Tuple[str, float]]], path: Path, run_tag: str = "MAW") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for qid in sorted(run.keys()):
            ranked = sorted(run[qid], key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                fh.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")
    return path


def evaluate_with_ir_measures(
    resources: DatasetResources,
    run: Dict[str, List[Tuple[str, float]]],
    metrics: Optional[Sequence] = None,
    per_query: bool = True,
):
    if ir_measures is None:
        raise RuntimeError("ir_measures is required but not installed.")

    metrics = tuple(metrics or resources.spec.primary_metrics)
    if not metrics:
        raise ValueError("No metrics provided for evaluation.")

    qrels_obj = ir_measures.Qrels(resources.qrels)
    run_obj = ir_measures.ScoredDocs(run)

    aggregate_results = ir_measures.calc_aggregate(metrics, qrels_obj, run_obj)
    aggregated = {str(metric): float(value) for metric, value in aggregate_results.items()}

    per_query_results: Dict[str, Dict[str, float]] = defaultdict(dict)
    if per_query:
        for metric, qid, value in ir_measures.iter_calc(metrics, qrels_obj, run_obj):
            per_query_results[str(qid)][str(metric)] = float(value)

    return aggregated, per_query_results


def compute_run(
    model,
    tokenizer,
    resources: DatasetResources,
    output_dir: Path,
    run_name: str,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    depth_logger: Optional[MAWDepthLogger] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = rerank(
        model,
        tokenizer,
        resources,
        batch_size=batch_size,
        max_length=max_length,
        depth_logger=depth_logger,
    )
    run_path = output_dir / f"{run_name}.trec"
    write_trec_run(scores, run_path, run_tag=run_name)
    return scores, run_path


def run_rerank_and_evaluate(
    model,
    tokenizer,
    resources: DatasetResources,
    output_dir: Path,
    run_name: str,
    metrics: Optional[Sequence] = None,
    depth_log_path: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
):
    depth_logger: Optional[MAWDepthLogger] = None
    if depth_log_path is not None:
        depth_logger = MAWDepthLogger(Path(depth_log_path))

    register_maw_logger(depth_logger)
    try:
        scores, run_path = compute_run(
            model,
            tokenizer,
            resources,
            output_dir=output_dir,
            run_name=run_name,
            batch_size=batch_size,
            max_length=max_length,
            depth_logger=depth_logger,
        )
    finally:
        register_maw_logger(None)
        if depth_logger is not None:
            depth_logger.close()

    aggregated, per_query = evaluate_with_ir_measures(
        resources,
        scores,
        metrics=metrics,
        per_query=True,
    )

    return {
        "metrics": aggregated,
        "per_query": per_query,
        "run_path": str(run_path),
        "depth_log": str(depth_log_path) if depth_log_path else None,
        "scores": scores,
    }


def write_per_query_scores(path: Path, scores: Dict[str, List[Tuple[str, float]]]) -> None:
    with path.open("w") as fh:
        for qid, doc_scores in scores.items():
            fh.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "scores": [[doc_id, float(score)] for doc_id, score in doc_scores],
                    }
                )
                + "\n"
            )


def summarize_seed_results(seed_results: Sequence[SeedRunResult]) -> Dict[str, Dict[str, float]]:
    metrics_map: Dict[str, List[float]] = defaultdict(list)
    for result in seed_results:
        for metric_name, value in result.metrics.items():
            metrics_map[metric_name].append(float(value))

    summary: Dict[str, Dict[str, float]] = {}
    for metric_name, values in metrics_map.items():
        arr = np.array(values, dtype=np.float32)
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        summary[metric_name] = {
            "mean": float(arr.mean()),
            "std": std,
            "values": [float(v) for v in arr],
        }
    return summary


def average_per_query(seed_results: Sequence[SeedRunResult], metric_name: str) -> Dict[str, float]:
    per_query_scores: Dict[str, List[float]] = defaultdict(list)
    for result in seed_results:
        for qid, metrics in result.per_query.items():
            if metric_name in metrics:
                per_query_scores[qid].append(float(metrics[metric_name]))

    averaged: Dict[str, float] = {}
    for qid, values in per_query_scores.items():
        if values:
            averaged[qid] = float(np.mean(values))
    return averaged


def paired_t_test(baseline_values: Sequence[float], variant_values: Sequence[float]):
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for paired t-tests. Install with `pip install scipy`.")

    baseline_arr = np.array(baseline_values, dtype=np.float64)
    variant_arr = np.array(variant_values, dtype=np.float64)
    if baseline_arr.size != variant_arr.size:
        raise ValueError("Paired t-test requires arrays of equal length.")
    if baseline_arr.size < 2:
        return float("nan"), float("nan")
    t_stat, p_value = stats.ttest_rel(variant_arr, baseline_arr)
    return float(p_value), float(t_stat)


def compute_significance(
    baseline_results: Sequence[SeedRunResult],
    variant_results: Sequence[SeedRunResult],
    metric_name: str,
):
    baseline_avg = average_per_query(baseline_results, metric_name)
    variant_avg = average_per_query(variant_results, metric_name)
    common_qids = sorted(set(baseline_avg.keys()) & set(variant_avg.keys()))
    if len(common_qids) < 2:
        return {
            "p_value": float("nan"),
            "t_stat": float("nan"),
            "queries": len(common_qids),
        }

    baseline_vals = [baseline_avg[qid] for qid in common_qids]
    variant_vals = [variant_avg[qid] for qid in common_qids]
    p_value, t_stat = paired_t_test(baseline_vals, variant_vals)
    return {
        "p_value": p_value,
        "t_stat": t_stat,
        "queries": len(common_qids),
    }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_variant_on_dataset(
    variant: VariantConfig,
    dataset_name: str,
    tokenizer,
    resources: DatasetResources,
    seeds: Sequence[int],
    output_root: Path,
) -> List[SeedRunResult]:
    results: List[SeedRunResult] = []
    for seed in seeds:
        run_dir = Path(output_root) / resources.spec.key / variant.name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_name = f"{variant.name}_seed{seed}"
        depth_log_path = run_dir / "maw_depth.jsonl" if variant.use_maw else None

        effective_seed = seed if variant.maw_random_seed is None else (variant.maw_random_seed + seed)
        set_global_seed(effective_seed)

        model = CrossEncoderWithMAW(
            BACKBONE,
            use_maw=variant.use_maw,
            maw_strength=variant.maw_strength,
            depth_dim=variant.depth_dim,
            inject_last_k=variant.inject_last_k,
            gating_mode=variant.gating_mode,
        ).to(device)

        model = apply_variant_adaptations(model, tokenizer, resources, variant)

        evaluation = run_rerank_and_evaluate(
            model,
            tokenizer,
            resources,
            output_dir=run_dir,
            run_name=run_name,
            metrics=tuple(resources.spec.primary_metrics or []),
            depth_log_path=depth_log_path,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )

        write_run_artifacts(
            run_dir,
            dataset_name,
            variant,
            effective_seed,
            evaluation,
            resources,
            base_seed=seed,
        )

        results.append(
            SeedRunResult(
                seed=seed,
                metrics=evaluation["metrics"],
                per_query=evaluation["per_query"],
                run_path=evaluation["run_path"],
                depth_log=evaluation["depth_log"],
                scores=evaluation["scores"],
            )
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def seed_result_to_public_dict(result: SeedRunResult, include_scores: bool = False) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "seed": result.seed,
        "metrics": result.metrics,
        "run_path": result.run_path,
        "depth_log": result.depth_log,
    }
    if include_scores:
        payload["scores"] = result.scores
    return payload


def run_dataset_suite(
    dataset_name: str,
    variants: Sequence[VariantConfig],
    seeds: Sequence[int],
    tokenizer,
    output_root: Path,
    include_scores: bool = False,
):
    try:
        resources = load_dataset_resources(dataset_name)
    except Exception as exc:
        print(f"[ERROR] Failed to load {dataset_name}: {exc}")
        return {
            "dataset": dataset_name,
            "error": str(exc),
        }
    ensure_dataset_metadata(resources, output_root)
    dataset_output: Dict[str, object] = {
        "dataset": dataset_name,
        "description": resources.spec.description,
        "candidates": str(resources.run_path),
        "variants": {},
    }

    baseline_name: Optional[str] = None
    baseline_results: Optional[List[SeedRunResult]] = None
    variant_results_map: Dict[str, List[SeedRunResult]] = {}

    for variant in variants:
        seed_results = run_variant_on_dataset(
            variant,
            dataset_name,
            tokenizer,
            resources,
            seeds,
            output_root=output_root,
        )
        variant_results_map[variant.name] = seed_results

        summary = summarize_seed_results(seed_results)
        dataset_output["variants"][variant.name] = {
            "config": asdict(variant),
            "summary": summary,
            "seeds": [seed_result_to_public_dict(res, include_scores=include_scores) for res in seed_results],
        }

        if baseline_results is None and not variant.use_maw:
            baseline_name = variant.name
            baseline_results = seed_results

    metrics_for_significance = [str(m) for m in resources.spec.primary_metrics[:1]] if resources.spec.primary_metrics else []

    if baseline_results is not None and metrics_for_significance:
        for variant in variants:
            if variant.name == baseline_name:
                continue
            seed_results = variant_results_map.get(variant.name)
            if not seed_results:
                continue
            significance_report = {}
            for metric_name in metrics_for_significance:
                significance_report[metric_name] = compute_significance(baseline_results, seed_results, metric_name)
            dataset_output["variants"][variant.name]["significance_vs_{}".format(baseline_name)] = significance_report

    return dataset_output


def run_experiment_suite(
    datasets: Sequence[str],
    variants: Sequence[VariantConfig],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    output_root: Path = Path("experiments"),
    include_scores: bool = False,
    dev_sweeps: Optional[Dict[str, Sequence[VariantConfig]]] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    suite_results: Dict[str, object] = {}
    for dataset_name in datasets:
        print(f"\n>>> Running suite for {dataset_name} ...")
        suite_results[dataset_name] = run_dataset_suite(
            dataset_name,
            variants,
            seeds,
            tokenizer,
            output_root=output_root,
            include_scores=include_scores,
        )
        if dev_sweeps and dataset_name in dev_sweeps:
            print(f"  Performing dev sweep for {dataset_name} ...")
            suite_results[dataset_name]["dev_sweep"] = run_dataset_suite(
                dataset_name,
                dev_sweeps[dataset_name],
                seeds,
                tokenizer,
                output_root=output_root / "dev_sweeps",
                include_scores=False,
            )
    return suite_results


def build_lora_training_pairs(
    resources: DatasetResources,
    max_negatives_per_pos: int = 4,
    topk_limit: Optional[int] = 200,
    max_queries: Optional[int] = None,
):
    pairs: List[Tuple[str, str, float]] = []
    for idx, (qid, doc_ids) in enumerate(resources.candidate_pools.items()):
        if max_queries is not None and idx >= max_queries:
            break
        query_text = resources.queries.get(qid)
        if not query_text:
            continue
        judged = resources.qrels.get(qid, {})
        positives = [did for did in doc_ids[:topk_limit or len(doc_ids)] if judged.get(did, 0) > 0]
        if not positives:
            continue
        negatives = [
            did for did in doc_ids[:topk_limit or len(doc_ids)] if judged.get(did, 0) == 0
        ]
        if max_negatives_per_pos is not None and max_negatives_per_pos > 0:
            negatives = negatives[: max_negatives_per_pos * len(positives)]

        doc_texts = resources.doc_lookup.get_many(list(set(positives + negatives)))
        for did in positives:
            text = doc_texts.get(did)
            if text:
                pairs.append((query_text, text, 1.0))
        for did in negatives:
            text = doc_texts.get(did)
            if text:
                pairs.append((query_text, text, 0.0))
    return pairs


class LoraFineTuneDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, str, float]]):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def make_lora_collate_fn(tokenizer, max_length: int):
    def _collate(batch):
        queries, docs, labels = zip(*batch)
        encoded = tokenizer(
            list(queries),
            list(docs),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.float32)
        return encoded

    return _collate


def finetune_with_lora(
    model: CrossEncoderWithMAW,
    tokenizer,
    resources: DatasetResources,
    variant: VariantConfig,
    device=device,
):
    pairs = build_lora_training_pairs(
        resources,
        max_negatives_per_pos=4,
        topk_limit=200,
        max_queries=variant.lora_max_queries,
    )
    if not pairs:
        print("[LoRA] No training pairs constructed; skipping fine-tuning.")
        return model

    dataset = LoraFineTuneDataset(pairs)
    collate_fn = make_lora_collate_fn(tokenizer, MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, variant.lora_batch_size),
        shuffle=True,
        collate_fn=collate_fn,
    )

    lora_config = LoraConfig(
        r=variant.lora_rank,
        lora_alpha=variant.lora_alpha,
        lora_dropout=variant.lora_dropout,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="SEQ_CLS",
    )
    model.backbone = get_peft_model(model.backbone, lora_config)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=variant.lora_learning_rate, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()

    for epoch in range(max(1, variant.lora_epochs)):
        epoch_losses = []
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(**inputs).squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if epoch_losses:
            print(f"[LoRA] Epoch {epoch+1}: loss={np.mean(epoch_losses):.4f}")

    model.eval()
    return model


def apply_variant_adaptations(
    model: CrossEncoderWithMAW,
    tokenizer,
    resources: DatasetResources,
    variant: VariantConfig,
):
    if variant.lora_finetune:
        return finetune_with_lora(model, tokenizer, resources, variant)
    return model


def write_run_artifacts(
    run_dir: Path,
    dataset_name: str,
    variant: VariantConfig,
    seed: int,
    evaluation: Dict[str, object],
    resources: DatasetResources,
    base_seed: Optional[int] = None,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": dataset_name,
        "variant": asdict(variant),
        "seed": seed,
        "base_seed": base_seed,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(),
        "system_info": get_system_info(),
        "candidate_run": str(resources.run_path),
        "run_path": evaluation.get("run_path"),
        "depth_log": evaluation.get("depth_log"),
    }

    with (run_dir / "config.json").open("w") as fh:
        json.dump(config, fh, indent=2)

    per_query = evaluation.get("per_query", {})
    with (run_dir / "per_query_metrics.json").open("w") as fh:
        json.dump(per_query, fh, indent=2)

    scores = evaluation.get("scores", {})
    write_per_query_scores(run_dir / "per_query_scores.jsonl", scores)


def build_pools_for_datasets(datasets: Sequence[str], force: bool = False) -> Dict[str, str]:
    built = {}
    for dataset in datasets:
        path = ensure_candidate_runfile(dataset, force=force)
        built[dataset] = str(path)
        print(f"Candidate pool ready: {dataset} -> {path}")
    return built


def ensure_dataset_metadata(resources: DatasetResources, output_root: Path) -> None:
    dataset_dir = Path(output_root) / resources.spec.key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    meta_path = dataset_dir / "dataset_meta.json"
    if not meta_path.exists():
        metadata = {
            "dataset": resources.spec.key,
            "description": resources.spec.description,
            "candidate_run": str(resources.run_path),
            "topk": resources.spec.topk,
        }
        with meta_path.open("w") as fh:
            json.dump(metadata, fh, indent=2)

    qrels_path = dataset_dir / "qrels.trec"
    if not qrels_path.exists():
        with qrels_path.open("w") as fh:
            for qid, docs in resources.qrels.items():
                for doc_id, rel in docs.items():
                    fh.write(f"{qid} 0 {doc_id} {rel}\n")

    queries_path = dataset_dir / "queries.json"
    if not queries_path.exists():
        with queries_path.open("w") as fh:
            json.dump(resources.queries, fh, indent=2)

# Helper functions to run wandb calls silently
def _silent_call(fn, *args, **kwargs):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            return fn(*args, **kwargs)
    except Exception:
        # swallow any benign initialization noise
        return None

def wandb_init_quiet(**kwargs):
    if QUIET_WANDB:
        return _silent_call(wandb.init, **kwargs)
    return wandb.init(**kwargs)

def wandb_login_quiet(**kwargs):
    if QUIET_WANDB:
        return _silent_call(wandb.login, **kwargs)
    return wandb.login(**kwargs)

# Initialize wandb for experiment tracking
def setup_wandb(use_maw):
    wandb_init_quiet(
        project="maw-reranker",
        entity="enverdenizaskin-ibm",
        name=f"scidocs-reranking-{'maw' if use_maw else 'baseline'}",
        config={
            "backbone": BACKBONE,
            "use_maw": use_maw,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "lora_rank": LORA_RANK,
            "max_length": MAX_LENGTH,
        },
        mode="online",
        resume="allow",
        sync_tensorboard=False,
        settings=wandb.Settings(console="off", _disable_stats=True, _disable_service=True, start_method="thread")
    )

# GRPO-based Depth Selector (unchanged)
class GRPODepthSelector(nn.Module):
    def __init__(self, depth_dim, hidden_dim=128, use_grpo=False):
        super().__init__()
        self.fc1 = nn.Linear(depth_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, depth_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.use_grpo = use_grpo

    def forward(self, attn_5d):
        batch_size = attn_5d.shape[0]
        x = attn_5d.mean(dim=(-2, -3)).view(batch_size, -1)
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        if self.use_grpo:
            depth_probs = self.softmax(logits)
            depth_index = torch.multinomial(depth_probs, num_samples=1).squeeze(-1)
            depth_routing = F.one_hot(depth_index, num_classes=logits.shape[-1]).float()
        else:
            depth_probs = F.gumbel_softmax(logits, tau=1.0, hard=False)
            depth_index = depth_probs.argmax(dim=-1)
            depth_routing = depth_probs
        return depth_routing, depth_probs, depth_index


# Depthwise MAW: 5D attention + statistics-based selection/gating
class DepthwiseMAWSelfAttention(nn.Module):
    """
    MAW: depth-wise attention for the *last* self-attention layer.
    Transforms attention from (B, H, Lq, Lk) -> (B, H, D, Lq, Lk), where D = depth_dim.
    Implements a non-learning selection/gating over depths using variance, max, entropy & HHI.
    Falls back to the original attention if required attributes aren't available.
    """
    def __init__(self, original_attention, depth_dim=8, maw_strength=0.15, eps: float = 1e-6, gating_mode: str = "stat"):
        super().__init__()
        self.original_attention = original_attention
        # Try to mirror BERT-style attributes; use safe defaults if missing
        self.num_attention_heads = getattr(original_attention, "num_attention_heads", 12)
        self.attention_head_size = getattr(original_attention, "attention_head_size", None)
        if self.attention_head_size is None and hasattr(original_attention, "all_head_size"):
            # common in some implementations
            self.attention_head_size = getattr(original_attention, "all_head_size") // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size if self.attention_head_size is not None else None

        self.depth_dim = int(depth_dim)
        self.maw_strength = float(maw_strength)
        self.eps = eps
        self.gating_mode = gating_mode

        # Cached helpers if original module doesn't expose them
        self._has_proj = (
            (all(hasattr(original_attention, n) for n in ["query", "key", "value"]) or
             all(hasattr(original_attention, n) for n in ["query_proj", "key_proj", "value_proj"]))
            and self.attention_head_size is not None
        )
        # DeBERTa-style detection
        self._use_deberta_proj = all(
            hasattr(original_attention, n) for n in ["query_proj", "key_proj", "value_proj"]
        )
        self._orig_forward = getattr(original_attention, "forward", None)
    def _call_original_attention(
        self,
        hidden_states,
        attention_mask=None,
        **kwargs,
    ):
        """Call the wrapped attention's forward with only supported kwargs.
        This avoids errors like `unexpected keyword argument 'head_mask'` on modules
        (e.g., DeBERTa's DisentangledSelfAttention) that don't accept BERT-style args.
        """
        fwd = self._orig_forward if hasattr(self, "_orig_forward") else getattr(self.original_attention, "forward", None)
        if fwd is None:
            # As a last resort, try the bound module directly with minimal args
            if attention_mask is not None:
                return self.original_attention(hidden_states, attention_mask)
            return self.original_attention(hidden_states)

        sig = inspect.signature(fwd)

        # Build a candidate kwarg set from known attention interfaces across models
        candidates = {
            # BERT-style
            "attention_mask": attention_mask,
            "head_mask": kwargs.get("head_mask", None),
            "encoder_hidden_states": kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": kwargs.get("encoder_attention_mask", None),
            "past_key_value": kwargs.get("past_key_value", None),
            "output_attentions": kwargs.get("output_attentions", None),
            # DeBERTa-style
            "return_att": kwargs.get("return_att", None),
            "query_states": kwargs.get("query_states", None),
            "relative_pos": kwargs.get("relative_pos", None),
            "rel_embeddings": kwargs.get("rel_embeddings", None),
            # Misc common flags
            "use_cache": kwargs.get("use_cache", None),
        }

        filtered = {k: v for k, v in candidates.items() if (v is not None and k in sig.parameters)}

        # First try with all filtered kwargs passed by name
        try:
            return fwd(hidden_states, **filtered)
        except TypeError:
            # Some modules expect attention_mask as positional arg
            if "attention_mask" in sig.parameters and attention_mask is not None:
                filtered_no_mask = dict(filtered)
                filtered_no_mask.pop("attention_mask", None)
                return fwd(hidden_states, attention_mask, **filtered_no_mask)
            # Otherwise, re-raise the original TypeError
            raise

    # -------- Utility helpers --------
    @staticmethod
    def _transpose_for_scores(x, num_heads, head_dim):
        # x: [B, L, H*Dh] -> [B, H, L, Dh]
        new_shape = x.size()[:-1] + (num_heads, head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def _min_max_norm(vals, eps=1e-12):
        vmin = vals.min(dim=-1, keepdim=True).values
        vmax = vals.max(dim=-1, keepdim=True).values
        denom = (vmax - vmin).clamp_min(eps)
        return (vals - vmin) / denom

    def _score_depths(self, attn_5d):
        """
        attn_5d: [B, H, D, Lq, Lk] (softmax-normalized)
        Returns:
          weights: [B, D] soft selection weights per depth
          best_idx: [B] argmax index per batch
        """
        B, H, D, Lq, Lk = attn_5d.shape
        mode = self.gating_mode

        if mode == "uniform":
            weights = torch.full((B, D), 1.0 / max(D, 1), device=attn_5d.device, dtype=attn_5d.dtype)
            best_idx = torch.zeros(B, dtype=torch.long, device=attn_5d.device)
            return weights, best_idx

        if mode == "random":
            rand = torch.rand((B, D), device=attn_5d.device, dtype=attn_5d.dtype)
            rand_sum = rand.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            weights = rand / rand_sum
            best_idx = torch.argmax(weights, dim=-1)
            return weights, best_idx

        # Variance across keys (focus); higher is sharper
        var_k = attn_5d.var(dim=-1)  # [B,H,D,Lq]
        var_k = var_k.mean(dim=(1, 3))  # [B,D]
        # Max over keys per query; higher max suggests spiky focus
        max_k = attn_5d.max(dim=-1).values.mean(dim=(1, 3))  # [B,D]
        # Entropy (lower is better); add eps for stability
        p = attn_5d.clamp_min(self.eps)
        ent = (-p * p.log()).sum(dim=-1).mean(dim=(1, 3))  # [B,D]
        # Herfindahl-Hirschman Index (HHI) = sum(p^2), higher => concentrated
        hhi = (attn_5d.pow(2).sum(dim=-1)).mean(dim=(1, 3))  # [B,D]

        # Normalize each feature per batch across depths
        var_n = self._min_max_norm(var_k)
        max_n = self._min_max_norm(max_k)
        ent_n = self._min_max_norm(ent)
        hhi_n = self._min_max_norm(hhi)

        # Weighted ensemble; favor spiky/peaky distributions
        # Increase sharpness with maw_strength
        alpha = 1.0 + 10.0 * max(self.maw_strength, 0.0)
        score = (0.5 * var_n + 0.3 * max_n + 0.2 * hhi_n) - (0.4 * ent_n)
        weights = torch.softmax(alpha * score, dim=-1)  # [B,D]
        best_idx = torch.argmax(weights, dim=-1)  # [B]

        if mode == "argmax":
            one_hot = F.one_hot(best_idx, num_classes=D).to(weights.dtype)
            return one_hot, best_idx

        return weights, best_idx

    def _apply_mask(self, scores_5d, attention_mask):
        # scores_5d: [B,H,D,Lq,Lk]; attention_mask usually [B,1,1,Lk]
        if attention_mask is None:
            return scores_5d
        mask = attention_mask
        if mask.dim() == 4:  # [B,1,1,Lk]
            mask = mask.unsqueeze(2)  # [B,1,1,1,Lk] to broadcast over D
        elif mask.dim() == 3:  # [B,1,Lk]
            mask = mask.unsqueeze(1).unsqueeze(1)  # -> [B,1,1,1,Lk]
        elif mask.dim() == 2:  # [B,Lk]
            mask = mask.view(mask.size(0), 1, 1, 1, mask.size(1))
        return scores_5d + mask

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=True,
        **kwargs,
    ):
        """Return (context_layer, attention_probs_4d).
        context_layer: [B, L, H*Dh] matching the original API
        attention_probs_4d: [B, H, Lq, Lk] (depth-compressed by our gating)
        """
        # If we cannot compute Q/K/V safely, fall back to original module
        if not self._has_proj or self.attention_head_size is None:
            try:
                return self._call_original_attention(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    **kwargs,
                )
            except TypeError as e:
                # Final minimal fallback with only mandatory args
                print(f"[MAW] Filtered fallback failed; retrying minimal call: {e}")
                if attention_mask is not None:
                    return self.original_attention(hidden_states, attention_mask)
                return self.original_attention(hidden_states)
            except Exception as e:
                print(f"[MAW] Fallback to original attention failed: {e}")
                raise

        B, Lq, _ = hidden_states.shape
        H = self.num_attention_heads
        Dh = self.attention_head_size
        D = int(self.depth_dim)

        # --- Q, K, V ---
        if self._use_deberta_proj:
            q_lin = self.original_attention.query_proj
            k_lin = self.original_attention.key_proj
            v_lin = self.original_attention.value_proj
        else:
            q_lin = getattr(self.original_attention, "query", None)
            k_lin = getattr(self.original_attention, "key", None)
            v_lin = getattr(self.original_attention, "value", None)

        q = q_lin(hidden_states)
        kv_source = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        k = k_lin(kv_source)
        v = v_lin(kv_source)

        # [B,H,L,Dh]
        q = self._transpose_for_scores(q, H, Dh)
        k = self._transpose_for_scores(k, H, Dh)
        v = self._transpose_for_scores(v, H, Dh)

        # Split head_dim into D slices (allow uneven last chunk)
        q_slices = torch.tensor_split(q, D, dim=-1)  # list of [B,H,L,dh_i]
        k_slices = torch.tensor_split(k, D, dim=-1)
        v_slices = torch.tensor_split(v, D, dim=-1)

        # Compute per-depth attention scores: [B,H,D,Lq,Lk]
        scores = []
        for qi, ki in zip(q_slices, k_slices):
            d_k = max(qi.size(-1), 1)
            s = torch.matmul(qi, ki.transpose(-1, -2)) / math.sqrt(d_k)
            scores.append(s)
        scores_5d = torch.stack(scores, dim=2)  # [B,H,D,Lq,Lk]
        scores_5d = self._apply_mask(scores_5d, attention_mask)

        # Softmax over keys per depth
        attn_5d = F.softmax(scores_5d, dim=-1)  # [B,H,D,Lq,Lk]

        # Non-learning depth selection/gating
        depth_weights, best_idx = self._score_depths(attn_5d)  # [B,D], [B]
        self._log_depth(depth_weights, best_idx)
        # Weighted sum of attn over depths for returning 4D attentions
        attn_4d = torch.sum(attn_5d * depth_weights.view(B, 1, D, 1, 1), dim=2)  # [B,H,Lq,Lk]

        # Compute per-depth contexts then gate & concatenate back to Dh
        ctx_slices = []
        for d, (attn_d, vi) in enumerate(zip(attn_5d.unbind(dim=2), v_slices)):
            # attn_d: [B,H,Lq,Lk], vi: [B,H,Lk,dh_i]
            ctx_d = torch.matmul(attn_d, vi)  # [B,H,Lq,dh_i]
            # Apply depth gating (broadcast over heads & tokens)
            w_d = depth_weights[:, d].view(B, 1, 1, 1)
            ctx_slices.append(ctx_d * w_d)
        # Concatenate gated depth slices along feature dim to reconstruct Dh
        context = torch.cat(ctx_slices, dim=-1)  # [B,H,Lq,Dh]

        # head mask (if any)
        if head_mask is not None:
            context = context * head_mask.view(1, -1, 1, 1)

        # Merge heads -> [B,L, H*Dh]
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = (B, Lq, H * context.size(-1))
        context = context.view(*new_context_shape)

        return context, attn_4d

    def _log_depth(self, depth_weights: torch.Tensor, best_idx: torch.Tensor) -> None:
        logger = get_active_maw_logger()
        if logger is not None:
            try:
                logger.record(depth_weights, best_idx)
            except Exception as exc:
                print(f"[MAW] Depth logging failed: {exc}")

# CrossEncoderWithMAW using pretrained classification head and MAW injection
class CrossEncoderWithMAW(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        use_maw: bool = False,
        maw_strength: float = 0.0,
        depth_dim: int = DEPTH,
        inject_last_k: int = 1,
        gating_mode: str = "stat",
    ):
        super().__init__()
        # Use the task-appropriate head; this preserves the pretrained scoring head
        self.backbone = AutoModelForSequenceClassification.from_pretrained(backbone_name)
        # Ensure attentions are available (for any downstream inspection)
        self.backbone.config.output_attentions = True

        self.use_maw = use_maw
        self.maw_layers = []

        inject_last_k = max(1, int(inject_last_k))

        # Locate the base encoder (bert/roberta/deberta/etc.) to inject MAW into the last attention layer
        base = None
        for attr in ("bert", "roberta", "deberta", "electra", "albert", "model", "backbone", "base_model"):
            if hasattr(self.backbone, attr):
                base = getattr(self.backbone, attr)
                break

        if use_maw and base is not None and hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
            num_layers = len(base.encoder.layer)
            target_indices = list(range(max(0, num_layers - inject_last_k), num_layers))
            for layer_idx in target_indices:
                original_attn = base.encoder.layer[layer_idx].attention.self
                base.encoder.layer[layer_idx].attention.self = DepthwiseMAWSelfAttention(
                    original_attn,
                    depth_dim=depth_dim,
                    maw_strength=maw_strength,
                    gating_mode=gating_mode,
                )
                self.maw_layers.append(layer_idx)
                if not (hasattr(original_attn, "query") or hasattr(original_attn, "query_proj")):
                    print("[MAW] Warning: Original attention lacks q/k/v projections; using compatibility fallback.")
            if self.maw_layers:
                print(
                    f"Injected Depthwise MAW into layers {self.maw_layers} "
                    f"(depth={depth_dim}, strength={maw_strength})"
                )

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
        )
        # Return logits as scores (shape [B, 1] or [B])
        return outputs.logits

# Dataset for reranking
class RerankingDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=MAX_LENGTH):
        self.data = []
        for ex in dataset:
            query = ex["query"]
            passages = ex["passages"]["passage_text"]
            labels = ex["passages"]["is_selected"]
            for passage, label in zip(passages, labels):
                self.data.append((query, passage, label))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, passage, label = self.data[idx]
        enc = self.tokenizer(query, passage, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(float(label))
        return enc

# Dataset for MTEB-style reranking (e.g., mteb/scidocs-reranking)
class MTEBRerankingDataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_length=MAX_LENGTH, max_negatives=20):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        print("Creating dataset from MTEB split...")
        
        # Add progress bar for dataset preparation
        update_frequency = max(1, len(hf_split) // 100)  # Only update ~100 times total
        for ex in tqdm(hf_split, desc="Preparing dataset", miniters=update_frequency):
            query = ex["query"]
            pos = ex["positive"]
            neg = ex["negative"]
            pos_list = pos if isinstance(pos, list) else [pos]
            neg_list = neg if isinstance(neg, list) else [neg]
            # Limit negatives for faster CPU training
            neg_list = neg_list[:max_negatives]
            for passage in pos_list:
                self.data.append((query, passage, 1.0))
            for passage in neg_list:
                self.data.append((query, passage, 0.0))
        
        print(f"Created dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, passage, label = self.data[idx]
        enc = self.tokenizer(query, passage, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(float(label))
        return enc

# Metrics for reranking
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits > 0).astype(int)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

def evaluate_reranking(model, tokenizer, test_dataset, ks=[1, 3, 5, 10]):
    model.eval()
    results = {f"Hit@{k}": 0 for k in ks}
    results.update({f"MRR@{k}": 0 for k in ks})
    results.update({f"NDCG@{k}": 0 for k in ks})
    count = 0
    with torch.no_grad():
        for ex in test_dataset:
            query = ex["query"]
            passages = ex["passages"]["passage_text"]
            labels = ex["passages"]["is_selected"]
            scores = []
            for passage in passages:
                inputs = tokenizer(query, passage, return_tensors="pt", truncation=True, padding=True).to(device)
                out = model(**inputs)
                score = out.squeeze().detach().cpu().item()
                scores.append(score)
            # Sort by score
            ranked_indices = np.argsort(scores)[::-1]
            ranked_labels = [labels[i] for i in ranked_indices]
            for k in ks:
                results[f"Hit@{k}"] += 1 if 1 in ranked_labels[:k] else 0
                results[f"MRR@{k}"] += mrr_at_k(ranked_labels, k)
                results[f"NDCG@{k}"] += ndcg_at_k(ranked_labels, k)
            count += 1
    for key in results:
        results[key] /= count
    return results

def evaluate_reranking_mteb(model, tokenizer, hf_split, ks=[1, 3, 5, 10]):
    model.eval()
    results = {f"Hit@{k}": 0 for k in ks}
    results.update({f"MRR@{k}": 0 for k in ks})
    results.update({f"NDCG@{k}": 0 for k in ks})
    count = 0
    
    # Update progress bar to show full dataset
    progress_bar = tqdm(
        total=len(hf_split),
        desc="Evaluating",
        leave=True,                # keep a single finished line
        dynamic_ncols=True,
        miniters=max(1, len(hf_split)//100),
        position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    with torch.no_grad():
        for ex_idx, ex in enumerate(hf_split):
            # Skip examples that don't match expected format
            if not all(k in ex for k in ["query", "positive", "negative"]) and not all(k in ex for k in ["query", "pos", "neg"]):
                # Check for alternative BeIR format
                if "query" in ex and "corpus" in ex and "relevant_docs" in ex:
                    query = ex["query"]
                    # Extract positive passages from corpus based on relevant_docs
                    pos_list = [ex["corpus"][doc_id] for doc_id in ex.get("relevant_docs", [])]
                    # Get negative passages (some from corpus that aren't in relevant_docs)
                    neg_ids = [doc_id for doc_id in list(ex["corpus"].keys())[:20]
                               if doc_id not in ex.get("relevant_docs", [])]
                    neg_list = [ex["corpus"][doc_id] for doc_id in neg_ids]
                else:
                    # silently skip incompatible formats to avoid breaking progress bar lines
                    continue
            else:
                # Handle both possible field naming conventions
                query = ex["query"]
                pos = ex.get("positive", ex.get("pos", []))
                neg = ex.get("negative", ex.get("neg", []))
                pos_list = pos if isinstance(pos, list) else [pos]
                neg_list = neg if isinstance(neg, list) else [neg]

            try:
                start_time = time.time()

                if len(neg_list) > 20:
                    neg_list = neg_list[:20]

                passages = pos_list + neg_list
                labels = [1] * len(pos_list) + [0] * len(neg_list)
                if sum(labels) == 0:
                    print(f"[DEBUG] No positive labels for query {ex_idx}")
                if len(pos_list) == 0:
                    print(f"[DEBUG] No positive passages for query {ex_idx}")
                # Get model scores
                scores = []
                model.eval()
                with torch.no_grad():
                    for passage in passages:
                        inputs = tokenizer(query, passage, return_tensors="pt", truncation=True, padding=True).to(device)
                        out = model(**inputs)
                        score = out.squeeze().detach().cpu().item()
                        scores.append(score)

                # Sort by scores
                ranked_indices = np.argsort(scores)[::-1]
                ranked_passages = [passages[i] for i in ranked_indices]
                ranked_labels = [labels[i] for i in ranked_indices]
                ranked_scores = [scores[i] for i in ranked_indices]

                # accumulate metrics
                for K in ks:
                    results[f"Hit@{K}"] += 1 if 1 in ranked_labels[:K] else 0
                    results[f"MRR@{K}"] += mrr_at_k(ranked_labels, K)
                    results[f"NDCG@{K}"] += ndcg_at_k(ranked_labels, K)
                count += 1

                # Optionally log per-example tables to wandb, but only if enabled
                if ENABLE_WANDB and LOG_EVAL_EXAMPLES:
                    columns = ["Rank", "Passage", "Score", "Relevant"]
                    table = wandb.Table(columns=columns)
                    for rank, (passage, score, label) in enumerate(zip(ranked_passages[:k], ranked_scores[:k], ranked_labels[:k])):
                        display_passage = passage[:300] + "..." if len(passage) > 300 else passage
                        table.add_data(rank+1, display_passage, f"{score:.4f}", "✓" if label == 1 else "✗")
                    wandb.log({f"examples/query_{ex_idx}": wandb.Html(f"<h3>Query: {query}</h3>")})
                    wandb.log({f"examples/results_{ex_idx}": table})
            except Exception as e:
                # silently skip errors to preserve single-line progress bar
                continue
            finally:
                # Update with current metrics in description
                if count > 0 and ex_idx % 25 == 0:
                    hit1 = results["Hit@1"] / count
                    ndcg10 = results["NDCG@10"] / count
                    progress_bar.set_postfix({"Hit@1": f"{hit1:.3f}", "NDCG@10": f"{ndcg10:.3f}"})
                progress_bar.update(1)
    
    progress_bar.close()
    
    # Calculate final metrics
    for key in results:
        results[key] /= max(count, 1)
    
    print(f"\nEvaluated {count} examples out of {len(hf_split)}")
    return results

k = 10  # Add this line before using 'k'

def mrr_at_k(labels, k):
    for i, label in enumerate(labels[:k]):
        if label == 1:
            return 1 / (i + 1)
    return 0

def dcg_at_k(labels, k):
    dcg = 0
    for i in range(min(k, len(labels))):
        dcg += labels[i] / math.log2(i + 2)
    return dcg

def ndcg_at_k(labels, k):
    dcg = dcg_at_k(labels, k)
    ideal = dcg_at_k(sorted(labels, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0

# Cleanup function for old checkpoints
def cleanup_old_checkpoints(checkpoint_dir, max_to_keep=5):
    """Keeps only the most recent 'max_to_keep' checkpoints, deleting older ones."""
    import shutil
    import glob
    import os
    
    # Get all subdirectories that match our checkpoint pattern
    pattern = os.path.join(checkpoint_dir, "epoch_*_p*")
    checkpoint_dirs = glob.glob(pattern)
    
    if len(checkpoint_dirs) <= max_to_keep:
        return  # No cleanup needed
    
    # Sort checkpoints by creation time (newest first)
    checkpoint_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Keep the top 'max_to_keep' and delete the rest
    for old_dir in checkpoint_dirs[max_to_keep:]:
        print(f"Removing old checkpoint: {old_dir}")
        try:
            shutil.rmtree(old_dir)
        except Exception as e:
            print(f"Error removing {old_dir}: {e}")
    
    print(f"Kept {min(max_to_keep, len(checkpoint_dirs))} most recent checkpoints")

# Add to your evaluation function
def log_sample_predictions(model, tokenizer, test_examples, k=5):
    """Log some sample predictions to WandB"""
    if not ENABLE_WANDB:
        return
        
    # Select a few random examples
    import random
    samples = random.sample(list(test_examples), min(5, len(test_examples)))
    
    for i, ex in enumerate(samples):
        query = ex["query"]
        pos = ex["positive"]
        neg = ex["negative"]
        
        pos_list = pos if isinstance(pos, list) else [pos]
        neg_list = neg if isinstance(neg, list) else [neg]
        passages = pos_list + neg_list[:min(len(neg_list), 9)]  # Limit to 10 total
        labels = [1] * len(pos_list) + [0] * len(passages[len(pos_list):])
        
        # Get model scores
        scores = []
        model.eval()
        with torch.no_grad():
            for passage in passages:
                inputs = tokenizer(query, passage, return_tensors="pt", truncation=True, padding=True).to(device)
                out = model(**inputs)
                score = out.squeeze().detach().cpu().item()
                scores.append(score)
        
        # Sort by scores
        ranked_indices = np.argsort(scores)[::-1]
        ranked_passages = [passages[i] for i in ranked_indices]
        ranked_labels = [labels[i] for i in ranked_indices]
        ranked_scores = [scores[i] for i in ranked_indices]
        
        # Create a WandB Table
        columns = ["Rank", "Passage", "Score", "Relevant"]
        table = wandb.Table(columns=columns)
        
        for rank, (passage, score, label) in enumerate(zip(ranked_passages[:k], ranked_scores[:k], ranked_labels[:k])):
            # Truncate very long passages for display
            display_passage = passage[:300] + "..." if len(passage) > 300 else passage
            table.add_data(rank+1, display_passage, f"{score:.4f}", "✓" if label == 1 else "✗")
        
        wandb.log({f"examples/query_{i}": wandb.Html(f"<h3>Query: {query}</h3>")})
        wandb.log({f"examples/results_{i}": table})

# Add this right after your imports to ensure proper authentication
import wandb
if ENABLE_WANDB:
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        wandb_login_quiet(key=wandb_api_key, relogin=False)

# -----------------------------
# Helpers: Robust dataset loaders
# -----------------------------

def _safe_len(ds):
    try:
        return len(ds)
    except Exception:
        return sum(1 for _ in ds)


def load_beir_via_irds(base_name: str, max_negatives: int = 40, seed: int = SEED):
    """Build a reranking-friendly test split for a BEIR dataset using the ir_datasets **Python API**.
    Returns a list of dicts, each with keys: {"query", "corpus", "relevant_docs"}.
    - corpus: a small per-query dict {doc_id: text} containing all positives + sampled negatives.
    """
    import random as _random

    if not HAS_IR_DATASETS:
        raise RuntimeError(
            "ir_datasets is required for loading BeIR datasets on datasets>=4.\n"
            "Install it with: pip install ir_datasets"
        )

    _random.seed(seed)

    # Try to use the explicit /test split; if it doesn't exist (e.g., trec-covid),
    # fall back to the base dataset key which corresponds to the BEIR test topics.
    dataset_key_try = f"beir/{base_name}/test"
    try:
        ds = ir_datasets.load(dataset_key_try)
    except KeyError:
        base_key = f"beir/{base_name}"
        print(f"[IRDS] '{dataset_key_try}' not found. Falling back to '{base_key}'.")
        ds = ir_datasets.load(base_key)

    # Build doc_id -> text lookup (prefer title + text if available)
    doc_lookup = {}
    for d in ds.docs_iter():
        doc_id = str(getattr(d, "doc_id"))
        title = getattr(d, "title", "") or ""
        text = getattr(d, "text", "") or ""
        combined = (title + " " + text).strip() if title else text.strip()
        if doc_id and combined:
            doc_lookup[doc_id] = combined

    all_doc_ids = list(doc_lookup.keys())

    # Aggregate qrels: query_id -> [relevant doc ids]
    qrels_by_q = defaultdict(list)
    for qr in ds.qrels_iter():
        try:
            rel = int(getattr(qr, "relevance", 0))
        except Exception:
            rel = 0
        if rel > 0:
            qid = str(getattr(qr, "query_id"))
            did = str(getattr(qr, "doc_id"))
            if did in doc_lookup:
                qrels_by_q[qid].append(did)

    # Build reranking examples
    examples = []
    skipped = 0
    for q in ds.queries_iter():
        qid = str(getattr(q, "query_id"))
        qtext = getattr(q, "text", "") or ""
        pos_ids = qrels_by_q.get(qid, [])
        if not qtext or not pos_ids:
            skipped += 1
            continue

        # Sample negatives not in positives
        negs = []
        tries = 0
        target = max_negatives
        while len(negs) < target and tries < target * 20 and all_doc_ids:
            did = _random.choice(all_doc_ids)
            if (did not in pos_ids) and (did not in negs):
                negs.append(did)
            tries += 1

        # Build a *small* per-query corpus: all positives + sampled negatives
        limited_ids = pos_ids + negs
        corpus = {did: doc_lookup[did] for did in limited_ids}
        examples.append({
            "query": qtext,
            "corpus": corpus,
            "relevant_docs": pos_ids,
        })

    try:
        q_count = ds.queries_count()
    except Exception:
        q_count = len(examples)
    print(
        f"IRDS (python) loader for beir/{base_name}/test: "
        f"docs={len(doc_lookup)}, queries={q_count}, built_examples={len(examples)}, skipped={skipped}"
    )
    return examples



# -----------------------------
# MS MARCO + TREC-DL loaders (via ir_datasets)
# -----------------------------

REL_THRESH_TREC = 2          # TREC DL considers rel>=2 as relevant for RR/AP variations
REL_THRESH_MSM = 1           # MS MARCO qrels are binary (1)
TOPK_CANDIDATES = 1000       # Tier-1 experiments require top-1000 candidates


def _combine_title_text(doc):
    title = getattr(doc, "title", "") or ""
    text = getattr(doc, "text", "") or ""
    return (title + " " + text).strip() if title else text.strip()


def _build_examples_from_irds(ds, topk=TOPK_CANDIDATES, rel_threshold=1):
    if not HAS_IR_DATASETS:
        raise RuntimeError(
            "ir_datasets is required for MS MARCO / TREC-DL loading.\n"
            "Install it with: pip install ir_datasets"
        )

    # Build qrels map with thresholding
    qrels_by_q = defaultdict(set)
    if ds.has_qrels():
        for qr in ds.qrels_iter():
            try:
                rel = int(getattr(qr, "relevance", 0))
            except Exception:
                rel = 0
            if rel >= rel_threshold:
                qrels_by_q[str(getattr(qr, "query_id"))].add(str(getattr(qr, "doc_id")))

    # Group top-k candidate doc ids by query
    cand_by_q = defaultdict(list)
    if ds.has_scoreddocs():
        for sd in ds.scoreddocs_iter():
            qid = str(getattr(sd, "query_id"))
            if len(cand_by_q[qid]) < topk:
                cand_by_q[qid].append(str(getattr(sd, "doc_id")))

    # Fast doc lookup
    if not ds.has_docs():
        raise RuntimeError("Dataset has no docs() — cannot build passages for reranking.")
    docstore = ds.docs_store()

    examples = []
    skipped = 0
    for q in ds.queries_iter():
        qid = str(getattr(q, "query_id"))
        qtext = getattr(q, "text", "") or ""
        pos_ids = list(qrels_by_q.get(qid, []))
        cand_ids = cand_by_q.get(qid, [])

        # Ensure positives are included in the candidate pool
        if pos_ids:
            for did in pos_ids:
                if did not in cand_ids:
                    cand_ids.append(did)
        if not qtext or not cand_ids:
            skipped += 1
            continue

        # Fetch documents via docstore
        # Use get_many for efficiency; fall back to per-id get on error
        try:
            docs_map = docstore.get_many(cand_ids)
        except Exception:
            docs_map = {did: docstore.get(did) for did in cand_ids if did}

        # Build minimal per-query corpus
        corpus = {}
        for did, doc in docs_map.items():
            if not doc:
                continue
            combined = _combine_title_text(doc)
            if combined:
                corpus[did] = combined

        if not corpus:
            skipped += 1
            continue

        # Keep only positives that exist in corpus
        pos_in_corpus = [did for did in pos_ids if did in corpus]
        if not pos_in_corpus:
            # In rare cases, no judged relevant doc is in the top-k pool; skip to avoid degenerate examples
            skipped += 1
            continue

        examples.append({
            "query": qtext,
            "corpus": corpus,
            "relevant_docs": pos_in_corpus,
        })

    print(
        f"IRDS loader: {ds} -> built_examples={len(examples)} skipped={skipped} "
        f"(topk={topk}, rel_threshold={rel_threshold})"
    )
    return examples


def load_msmarco_dev_small_via_irds(topk=TOPK_CANDIDATES):
    ds = ir_datasets.load("msmarco-passage/dev/small")
    return _build_examples_from_irds(ds, topk=topk, rel_threshold=REL_THRESH_MSM)


def load_trec_dl_passage(year: int, topk=TOPK_CANDIDATES):
    if year not in (2019, 2020):
        raise ValueError("year must be 2019 or 2020 for TREC DL passage.")
    ds = ir_datasets.load(f"msmarco-passage/trec-dl-{year}/judged")
    return _build_examples_from_irds(ds, topk=topk, rel_threshold=REL_THRESH_TREC)


def load_benchmark_split(dataset_name: str):
    """Unified loader used by the benchmark loop.
    - For MTEB-style reranking datasets (e.g., mteb/scidocs-reranking): use split="test" directly.
    - For BeIR/* datasets: build a test split via the ir_datasets Python API (beir/*/test).
    - For MS MARCO and TREC-DL: use ir_datasets to build reranking pools from scoreddocs + qrels.
    """
    if dataset_name.startswith("BeIR/"):
        base = dataset_name.split("/")[-1]
        return load_beir_via_irds(base)
    if dataset_name == "MSMARCO/dev-small":
        return load_msmarco_dev_small_via_irds()
    if dataset_name == "TREC-DL-2019-passage":
        return load_trec_dl_passage(2019)
    if dataset_name == "TREC-DL-2020-passage":
        return load_trec_dl_passage(2020)
    # default: datasets Hub split
    return load_dataset(dataset_name, split="test")

# Main - Modified to test on multiple datasets
def main():
    parser = argparse.ArgumentParser(description="MAW reranker experiment suite")
    parser.add_argument("--mode", choices=["suite", "build-pools", "dev-sweep"], default="suite")
    parser.add_argument("--datasets", nargs="*", default=list(PRIMARY_BENCHMARKS))
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--variants", choices=["default", "ablations", "all"], default="all")
    parser.add_argument("--force-pools", action="store_true")
    parser.add_argument("--with-dev-sweep", action="store_true")

    args = parser.parse_args()

    datasets = list(args.datasets)
    if args.include_secondary:
        datasets.extend(list(SECONDARY_DATASETS))
    # Deduplicate while preserving order
    datasets = list(dict.fromkeys(datasets))

    if args.mode == "build-pools":
        build_pools_for_datasets(datasets, force=args.force_pools)
        return

    if args.mode == "dev-sweep":
        tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        results = {}
        for dataset in datasets:
            print(f"Running dev sweep for {dataset}...")
            results[dataset] = run_dataset_suite(
                dataset,
                DEV_SWEEP_VARIANTS,
                args.seeds,
                tokenizer,
                output_root=output_root / "dev_sweeps",
                include_scores=False,
            )
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        summary_path = output_root / f"dev_sweep_{timestamp}.json"
        with summary_path.open("w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Dev sweep summary saved to {summary_path}")
        return

    # Suite mode
    variant_groups = []
    if args.variants in ("default", "all"):
        variant_groups.append(DEFAULT_VARIANTS)
    if args.variants in ("ablations", "all"):
        variant_groups.append(ABLATION_VARIANTS)
    variants = unique_variants(*variant_groups)

    dev_sweeps = {"MSMARCO/dev-small": DEV_SWEEP_VARIANTS} if args.with_dev_sweep else None

    start_time = time.time()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    suite_results = run_experiment_suite(
        datasets=datasets,
        variants=variants,
        seeds=args.seeds,
        output_root=output_root,
        include_scores=False,
        dev_sweeps=dev_sweeps,
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"suite_{timestamp}.json"

    summary_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backbone": BACKBONE,
        "seeds": args.seeds,
        "variants": [asdict(v) for v in variants],
        "datasets": suite_results,
        "runtime_seconds": time.time() - start_time,
        "git_commit": get_git_commit(),
        "system_info": get_system_info(),
    }

    with summary_path.open("w") as fh:
        json.dump(summary_payload, fh, indent=2)

    print(f"Suite completed. Summary written to {summary_path}")
    for dataset_name, dataset_report in suite_results.items():
        print(f"\nDataset: {dataset_name}")
        if "error" in dataset_report and "variants" not in dataset_report:
            print(f"  ERROR: {dataset_report['error']}")
            continue
        for variant_name, details in dataset_report.get("variants", {}).items():
            summary = details.get("summary", {})
            if not summary:
                print(f"  {variant_name}: no metrics available")
                continue
            metrics_str = ", ".join(
                f"{metric}={values['mean']:.4f}±{values['std']:.4f}" for metric, values in summary.items()
            )
            print(f"  {variant_name}: {metrics_str}")


if __name__ == "__main__":
    main()
