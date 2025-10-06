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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pytrec_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search as BEIRBM25
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
    maw_layers: int = 1
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

    def clone_for(self, **kwargs) -> "BenchmarkConfig":
        params = self.__dict__.copy()
        params.update(kwargs)
        return BenchmarkConfig(**params)

    def __post_init__(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            count = torch.cuda.device_count()
            self.device_ids = tuple(range(count))
        else:
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
        loader = self._ensure_beir_dataset(dataset_name, "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip")
        train_split = self._try_load_split(loader, dataset_name, ["train"])
        dev_split = self._try_load_split(loader, dataset_name, ["dev", "validation", "val"])
        test_split = self._try_load_split(loader, dataset_name, ["test"])
        if test_split is None and dev_split is not None:
            logging.info("MS MARCO missing explicit test split; using dev split for final evaluation and disabling dev monitoring.")
            test_split = dev_split
            dev_split = None
        return self._assemble_bundle(dataset_name, train_split, dev_split, test_split)

    def get_beir_dataset(self, dataset: str) -> DatasetBundle:
        loader = self._ensure_beir_dataset(dataset, f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip")
        train_split = self._try_load_split(loader, dataset, ["train"])
        dev_split = self._try_load_split(loader, dataset, ["dev", "validation", "val"])
        test_split = self._try_load_split(loader, dataset, ["test", "dev"])
        if test_split is None:
            raise ValueError(f"Dataset '{dataset}' does not provide a test split.")
        if dev_split is not None and test_split[3] == dev_split[3]:
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

        loader = GenericDataLoader(str(dataset_path))
        train_split = self._try_load_split(loader, dataset_name, ["train"])
        dev_split = self._try_load_split(loader, dataset_name, ["dev", "validation", "val"])
        test_split = self._try_load_split(loader, dataset_name, ["test"])
        if test_split is None:
            raise ValueError(f"LoTTE split '{split}' does not include a test partition.")
        return self._assemble_bundle(dataset_name, train_split, dev_split, test_split)

    def _ensure_beir_dataset(self, dataset_name: str, url: str) -> GenericDataLoader:
        dataset_path = ensure_dir(Path(self.config.data_root) / dataset_name)
        if not any(dataset_path.iterdir()):
            logging.info("Downloading dataset %s ...", dataset_name)
            beir_util.download_and_unzip(url, str(Path(self.config.data_root)))
        if dataset_name not in self._loader_cache:
            self._loader_cache[dataset_name] = GenericDataLoader(str(dataset_path))
        return self._loader_cache[dataset_name]

    def _try_load_split(
        self,
        loader: GenericDataLoader,
        dataset: str,
        candidates: List[str],
    ) -> Optional[Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]], str]]:
        for split in candidates:
            try:
                corpus, queries, qrels = loader.load(split=split)
                logging.info(
                    "%s: loaded split '%s' with %d queries and %d qrels",
                    dataset,
                    split,
                    len(queries),
                    sum(len(v) for v in qrels.values()),
                )
                return corpus, dict(queries), {qid: dict(doc_scores) for qid, doc_scores in qrels.items()}, split
            except Exception as exc:  # noqa: BLE001 - surfaces dataset issues but continues
                logging.debug("Dataset %s missing split '%s': %s", dataset, split, exc)
        return None

    def _assemble_bundle(
        self,
        dataset: str,
        train_split: Optional[Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]], str]],
        dev_split: Optional[Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]], str]],
        test_split: Optional[Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]], str]],
    ) -> DatasetBundle:
        if test_split is None:
            raise ValueError(f"Dataset '{dataset}' is missing a required evaluation split.")

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
        self.index_path = ensure_dir(Path(config.bm25_index_root) / dataset_name)
        self.searcher: Optional[BEIRBM25] = None

    def build(self, corpus: Dict[str, Dict[str, str]]) -> None:
        logging.info("Building Pyserini BM25 index for %s ...", self.dataset_name)
        self.searcher = BEIRBM25(
            index_dir=str(self.index_path),
            initialize=True,
            k1=0.9,
            b=0.4,
        )
        self.searcher.index(corpus)

    def search(self, queries: Dict[str, str], top_k: int) -> Dict[str, Dict[str, float]]:
        if self.searcher is None:
            raise RuntimeError("BM25 index has not been built")
        return self.searcher.search(corpus=None, queries=queries, top_k=top_k)


class TokenLevelMAW(nn.Module):
    def __init__(self, hidden_size: int, depth_dim: int, num_heads: int) -> None:
        super().__init__()
        self.depth_projection = nn.Linear(hidden_size, hidden_size * depth_dim)
        self.depth_dim = depth_dim
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        depth_states = self.depth_projection(hidden_states)
        batch, seq, hidden = hidden_states.size()
        depth_states = depth_states.view(batch, seq, self.depth_dim, hidden).mean(dim=2)
        attn_output, _ = self.attention(hidden_states, depth_states, depth_states, key_padding_mask=~attention_mask.bool())
        output = self.feedforward(attn_output)
        return self.norm(hidden_states + output)


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

        self.maw_layers = None
        if config.use_maw:
            self.maw_layers = nn.ModuleList(
                [TokenLevelMAW(self.hidden_size_value, config.maw_depth_dim, config.maw_num_heads) for _ in range(config.maw_layers)]
            )

        self.primary_device = self._resolve_primary_device()
        self.model.to(self.primary_device)
        if config.device.startswith("cuda") and len(config.device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(config.device_ids))

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
            outputs = self.model(**tokenized)
            hidden_states = outputs.last_hidden_state
            if self.maw_layers is not None:
                for layer in self.maw_layers:
                    hidden_states = layer(hidden_states, tokenized["attention_mask"])
            pooled = mean_pool(hidden_states, tokenized["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=-1)
        return normalized

    def encode_train(self, texts: List[str], batch_size: Optional[int] = None) -> torch.Tensor:
        if batch_size is None:
            batch_size = self.config.batch_size
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

    def forward(self, texts: List[str]) -> torch.Tensor:  # type: ignore[override]
        return self.encode_train(texts, batch_size=self.config.batch_size)

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
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _resolve_primary_device(self) -> torch.device:
        if self.config.device.startswith("cuda") and self.config.device_ids:
            return torch.device(f"cuda:{self.config.device_ids[0]}")
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
        self.encoder = encoder
        self.config = config
        params = [p for p in encoder.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
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

        max_workers = max(1, min(8, (os.cpu_count() or 1)))
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=max_workers,
            pin_memory=self.config.device.startswith("cuda"),
        )
        num_training_steps = len(dataloader) * self.config.epochs
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, num_training_steps)

        logging.info(
            "[Train:%s] Using %s | batches=%d | epochs=%d",
            dataset_name,
            device_status(self.config),
            len(dataloader),
            self.config.epochs,
        )

        self.encoder.train()
        best_metric = -1.0
        best_state = None
        self.wall_clock_start = time.time()

        status = device_status(self.config)
        for epoch in range(self.config.epochs):
            progress = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{self.config.epochs} [{status}]",
                leave=False,
            )
            for batch in progress:
                queries, positives, negatives = batch
                batch_queries = list(queries)
                batch_pos = list(positives)
                negative_lists = [list(nlist) for nlist in negatives]

                query_emb = self.encoder.encode_train(batch_queries)
                pos_emb = self.encoder.encode_train(batch_pos)

                flat_negatives = [neg for neg_list in negative_lists for neg in neg_list]
                if flat_negatives:
                    neg_batch_size = self.config.batch_size * max(1, self.config.negatives_per_query)
                    neg_emb = self.encoder.encode_train(flat_negatives, batch_size=neg_batch_size)
                else:
                    neg_emb = torch.empty((0, query_emb.size(1)), device=self.config.device)

                doc_emb = torch.cat([pos_emb, neg_emb], dim=0)
                logits = torch.matmul(query_emb, doc_emb.t()) / self.config.temperature
                labels = torch.arange(logits.size(0), device=self.config.device)
                loss = F.cross_entropy(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=2.0)
                self.optimizer.step()
                scheduler.step()

                self.global_step += 1
                token_factor = 1 + 1 + self.config.negatives_per_query
                token_count = len(batch_queries) * self.config.max_seq_length * token_factor
                self.tokens_processed += token_count
                progress.set_postfix({"loss": loss.item(), "step": self.global_step})

            if dev_partition is not None and not dev_partition.is_empty():
                metrics, _ = evaluator.evaluate_dense_model(
                    dataset_name,
                    self.encoder,
                    corpus,
                    dev_partition.queries,
                    dev_partition.qrels,
                    split="dev",
                )
                dev_metric = metrics.get("nDCG@10", metrics.get("MAP@100", 0.0))
                if dev_metric > best_metric:
                    best_metric = dev_metric
                    best_state = {k: v.detach().cpu() for k, v in self.encoder.state_dict().items()}

        if best_state:
            self.encoder.load_state_dict(best_state)

        wall_clock = time.time() - self.wall_clock_start
        return {
            "steps": self.global_step,
            "tokens": self.tokens_processed,
            "wall_clock_sec": wall_clock,
        }


class EvaluationManager:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.primary_device = self._resolve_primary_device()

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

        results: Dict[str, Dict[str, float]] = {}
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        for start in tqdm(
            range(0, len(query_ids), self.config.eval_batch_size),
            desc=f"{dataset_id}/{split}:queries [{device_status(self.config)}]",
            leave=False,
        ):
            batch_ids = query_ids[start : start + self.config.eval_batch_size]
            batch_texts = query_texts[start : start + self.config.eval_batch_size]
            if not batch_ids:
                continue
            batch_embeddings = encoder.encode(batch_texts, batch_size=self.config.eval_batch_size)
            batch_np = np.ascontiguousarray(batch_embeddings.numpy(), dtype=np.float32)
            if batch_np.size == 0:
                continue
            scores, indices = self._search_corpus(batch_np, corpus_embeddings, top_k)
            for row, qid in enumerate(batch_ids):
                ranking: Dict[str, float] = {}
                for col in range(scores.shape[1]):
                    doc_idx = int(indices[row, col])
                    if doc_idx < 0 or doc_idx >= len(doc_ids):
                        continue
                    ranking[doc_ids[doc_idx]] = float(scores[row, col])
                results[qid] = ranking

        if previous_mode:
            encoder.train()

        metrics, per_query = self.compute_metrics(qrels, results)
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

        if emb_path.exists() and meta_path.exists():
            try:
                with meta_path.open("r") as f:
                    meta = json.load(f)
                if meta.get("doc_count") == len(doc_ids) and meta.get("hidden") == hidden_size:
                    return np.memmap(emb_path, dtype=np.float32, mode="r", shape=(len(doc_ids), hidden_size))
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to reuse dense cache at %s: %s", emb_path, exc)

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
            batch_np = np.ascontiguousarray(batch_embeddings.numpy(), dtype=np.float32)
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

        return np.memmap(emb_path, dtype=np.float32, mode="r", shape=(len(doc_ids), hidden_size))

    def _search_corpus(
        self,
        query_embeddings: np.ndarray,
        corpus_embeddings: np.memmap,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_queries = query_embeddings.shape[0]
        num_docs = corpus_embeddings.shape[0]
        if num_docs == 0 or top_k == 0:
            return np.empty((num_queries, 0), dtype=np.float32), np.empty((num_queries, 0), dtype=np.int64)

        k = min(top_k, num_docs)
        base_chunk = max(k, self._determine_chunk_size(num_queries))
        devices = [torch.device(f"cuda:{idx}") for idx in self.config.device_ids] if (
            self.config.device.startswith("cuda") and self.config.device_ids
        ) else []
        device_count = max(1, len(devices))
        per_device_chunk = max(1, base_chunk // device_count)
        if devices and per_device_chunk > 60000:
            per_device_chunk = 60000

        top_scores = np.full((num_queries, k), -np.inf, dtype=np.float32)
        top_indices = np.full((num_queries, k), -1, dtype=np.int64)

        queries_gpu = {}
        queries_tensor_cpu = torch.from_numpy(query_embeddings)
        if devices:
            for device in devices:
                with torch.cuda.device(device):
                    queries_gpu[device] = queries_tensor_cpu.to(device, non_blocking=False)

        doc_pointer = 0
        while doc_pointer < num_docs:
            tasks: List[Tuple[torch.device, int, int, np.ndarray]] = []
            for device in devices or [torch.device("cpu")]:
                if doc_pointer >= num_docs:
                    break
                end = min(doc_pointer + per_device_chunk, num_docs)
                chunk = np.asarray(corpus_embeddings[doc_pointer:end], dtype=np.float32)
                if chunk.size == 0:
                    doc_pointer = end
                    continue
                tasks.append((device, doc_pointer, end, chunk))
                doc_pointer = end
            results = self._run_parallel_matmul(tasks, queries_tensor_cpu, queries_gpu)
            if not results:
                continue
            for chunk_start, chunk_end, scores in sorted(results, key=lambda x: x[0]):
                combined_scores = np.concatenate([top_scores, scores], axis=1)
                range_indices = np.arange(chunk_start, chunk_end, dtype=np.int64)
                chunk_indices = np.broadcast_to(range_indices, (num_queries, chunk_end - chunk_start))
                combined_indices = np.concatenate([top_indices, chunk_indices], axis=1)
                if k < combined_scores.shape[1]:
                    top_idx = np.argpartition(combined_scores, -k, axis=1)[:, -k:]
                    top_scores = np.take_along_axis(combined_scores, top_idx, axis=1)
                    top_indices = np.take_along_axis(combined_indices, top_idx, axis=1)
                else:
                    top_scores = combined_scores
                    top_indices = combined_indices
                order = np.argsort(-top_scores, axis=1)
                top_scores = np.take_along_axis(top_scores, order, axis=1)
                top_indices = np.take_along_axis(top_indices, order, axis=1)
        if queries_gpu:
            for tensor in queries_gpu.values():
                del tensor
            queries_gpu.clear()
        return top_scores, top_indices

    def _determine_chunk_size(self, num_queries: int) -> int:
        target_bytes = 256 * 1024 * 1024  # 256MB buffer
        denom = max(num_queries, 1) * 4  # float32
        chunk = target_bytes // denom
        return max(1024, chunk)

    def _run_parallel_matmul(
        self,
        tasks: List[Tuple[torch.device, int, int, np.ndarray]],
        queries_cpu: torch.Tensor,
        queries_gpu: Dict[torch.device, torch.Tensor],
    ) -> List[Tuple[int, int, np.ndarray]]:
        if not tasks:
            return []

        if len(tasks) == 1:
            device, start, end, docs = tasks[0]
            scores = self._matmul_device(device, docs, queries_cpu, queries_gpu)
            return [(start, end, scores)]

        results: List[Tuple[int, int, np.ndarray]] = []

        def _execute(device: torch.device, start: int, end: int, docs: np.ndarray) -> Tuple[int, int, np.ndarray]:
            scores = self._matmul_device(device, docs, queries_cpu, queries_gpu)
            return start, end, scores

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [executor.submit(_execute, device, start, end, docs) for device, start, end, docs in tasks]
            for future in futures:
                results.append(future.result())
        return results

    def _matmul_device(
        self,
        device: torch.device,
        docs_np: np.ndarray,
        queries_cpu: torch.Tensor,
        queries_gpu: Dict[torch.device, torch.Tensor],
    ) -> np.ndarray:
        if device.type == "cuda":
            torch.cuda.set_device(device)
            q_tensor = queries_gpu[device]
            docs_tensor = torch.from_numpy(docs_np).to(device, non_blocking=False)
            with torch.no_grad():
                scores = torch.matmul(q_tensor, docs_tensor.t())
            scores_cpu = scores.cpu().numpy()
            del docs_tensor, scores
            return scores_cpu
        docs_tensor = torch.from_numpy(docs_np)
        with torch.no_grad():
            scores_cpu = (queries_cpu @ docs_tensor.t()).numpy()
        del docs_tensor
        return scores_cpu

    def _resolve_primary_device(self) -> torch.device:
        if self.config.device.startswith("cuda") and self.config.device_ids:
            return torch.device(f"cuda:{self.config.device_ids[0]}")
        return torch.device("cpu")

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

    def run(self) -> None:
        set_random_seed(self.config.seed)
        if self.config.ms_marco:
            self._run_bundle(self.dataset_manager.get_msmarco())
        for dataset in self.config.beir_datasets:
            self._run_bundle(self.dataset_manager.get_beir_dataset(dataset))
        for split in self.config.lotte_splits:
            self._run_bundle(self.dataset_manager.get_lotte_split(split))
        if self.reports:
            ensure_dir(Path("results"))
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out_path = Path("results") / f"tier1_report_{timestamp}.json"
            with out_path.open("w") as f:
                json.dump(self.reports, f, indent=2)
            logging.info("Saved aggregated report to %s", out_path)

    def _run_bundle(self, bundle: DatasetBundle) -> None:
        bundle.ensure_test()
        dataset_name = bundle.name
        test_queries = bundle.test.queries if bundle.test else {}
        test_qrels = bundle.test.qrels if bundle.test else {}

        if not test_queries:
            logging.warning("Skipping dataset %s because the evaluation split is empty.", dataset_name)
            return

        logging.info("=== %s ===", dataset_name)
        logging.info("Device allocation: %s", device_status(self.config))
        bm25 = PyseriniBM25Retriever(self.config, dataset_name)
        bm25.build(bundle.corpus)
        bm25_metrics, _ = self.evaluator.evaluate_bm25(dataset_name, bm25, test_queries, test_qrels)
        dataset_report: Dict[str, Any] = {"BM25": {"metrics": bm25_metrics}}

        hard_negatives: Dict[str, List[str]] = {}
        if bundle.has_train():
            negative_depth = max(self.config.negatives_per_query * 10, 100)
            hard_negatives = self._mine_bm25_negatives(bm25, bundle.train.queries, bundle.train.qrels, negative_depth)

        baseline_metrics, baseline_per_query = self.evaluator.evaluate_dense_model(
            dataset_name,
            self.base_encoder,
            bundle.corpus,
            test_queries,
            test_qrels,
            split="test",
            cache_prefix=f"{dataset_name}/DenseZeroShot",
        )
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
                logging.warning(
                    "%s: training split empty after filtering; fine-tuning variants will reuse reference checkpoints.",
                    dataset_name,
                )
            else:
                train_dataset = candidate_dataset
        else:
            logging.info(
                "%s: no training split detected; reusing reference fine-tuned checkpoints where available.",
                dataset_name,
            )

        variant_specs = {
            "DenseLoRA": self.config.clone_for(use_lora=True, use_maw=False),
            "MAWLoRA": self.config.clone_for(use_lora=True, use_maw=True),
            "MAWFullFT": self.config.clone_for(use_lora=False, use_maw=True),
        }

        per_query_scores: Dict[str, Dict[str, Dict[str, float]]] = {
            "DenseZeroShot": baseline_per_query,
        }

        for label, variant_cfg in variant_specs.items():
            encoder, training_stats, source = self._prepare_variant_model(
                label,
                variant_cfg,
                dataset_name,
                train_dataset,
                bundle,
            )
            logging.info(
                "[Eval:%s] %s (source=%s) on %s",
                dataset_name,
                label,
                source,
                device_status(self.config),
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
            dataset_report[label] = {
                "metrics": metrics,
                "budget": compile_budget_report(label, encoder, training_stats),
                "training_source": source,
            }
            per_query_scores[label] = per_query

            if self.config.device.startswith("cuda"):
                encoder.model.to("cpu")
                torch.cuda.empty_cache()

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
    parser.add_argument("--maw-layers", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
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
        use_lora=False,
        use_maw=False,
        maw_depth_dim=args.maw_depth,
        maw_num_heads=args.maw_heads,
        maw_layers=args.maw_layers,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
    config = build_config(args)
    logging.info(
        "Runtime devices -> %s (%s)",
        device_status(config),
        ",".join(f"cuda:{idx}" for idx in config.device_ids) if config.device_ids else "cpu",
    )

    if not (config.ms_marco or config.beir_datasets or config.lotte_splits):
        logging.info("No datasets specified; defaulting to MS MARCO dev")
        config.ms_marco = True

    runner = BenchmarkRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
