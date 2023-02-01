# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from collections import defaultdict
from typing import List, Dict
import numpy as np
import torch
import torch.distributed as dist

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch

from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

import normalize_text

import glob

import json

import transformers

class DenseEncoderModel:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        query_tokenizer=None,
        doc_tokenizer=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        is_dpr=False,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer
        self.tokenizer = doc_tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text
        self.is_dpr = is_dpr

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        idx = range(len(queries))

        queries = [queries[i] for i in idx]
        if self.normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                qencode = self.query_tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                qencode = {key: value.cuda() for key, value in qencode.items()}
                if not self.is_dpr:
                    emb = self.query_encoder(**qencode, normalize=self.norm_query)
                else:
                    emb = self.query_encoder(**qencode)
                    emb = emb.pooler_output
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()

        allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        idx = range(len(corpus))
        corpus = [corpus[i] for i in idx]
        corpus = [c["title"] + " " + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]
        if self.normalize_text:
            corpus = [normalize_text.normalize(c) for c in corpus]
        if self.lower_case:
            corpus = [c.lower() for c in corpus]

        allemb = []
        nbatch = (len(corpus) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(corpus))

                cencode = self.doc_tokenizer.batch_encode_plus(
                    corpus[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                cencode = {key: value.cuda() for key, value in cencode.items()}
                if not self.is_dpr:
                    emb = self.doc_encoder(**cencode, normalize=self.norm_doc)
                else:
                    emb = self.doc_encoder(**cencode)
                    emb = emb.pooler_output
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        allemb = allemb.cpu().numpy()
        return allemb

def evaluate_model(
    query_encoder,
    doc_encoder,
    query_tokenizer,
    doc_tokenizer,
    dataset,
    batch_size=128,
    add_special_tokens=True,
    norm_query=False,
    norm_doc=False,
    split="test",
    score_function="dot",
    beir_dir="../Dataset/original_datasets/",
    save_results_path=None,
    lower_case=False,
    normalize_text=False,
    is_dpr=False,
):

    metrics = defaultdict(list)  # store final results

    if hasattr(query_encoder, "module"):
        query_encoder = query_encoder.module
    query_encoder.eval()

    if doc_encoder is not None:
        if hasattr(doc_encoder, "module"):
            doc_encoder = doc_encoder.module
        doc_encoder.eval()
    else:
        doc_encoder = query_encoder

    dmodel = DenseRetrievalExactSearch(
        DenseEncoderModel(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            query_tokenizer=query_tokenizer,
            doc_tokenizer=doc_tokenizer,
            add_special_tokens=add_special_tokens,
            norm_query=norm_query,
            norm_doc=norm_doc,
            lower_case=lower_case,
            normalize_text=normalize_text,
            is_dpr=is_dpr,
        ),
        batch_size=batch_size,
    )
    retriever = EvaluateRetrieval(dmodel, score_function=score_function)
    data_path = os.path.join(beir_dir, dataset)

    if not os.path.isdir(data_path):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = beir.util.download_and_unzip(url, beir_dir)

    if not dataset == "cqadupstack":
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        results = retriever.retrieve(corpus, queries)
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
            if isinstance(metric, str):
                metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
            for key, value in metric.items():
                metrics[key].append(value)
    elif dataset == "cqadupstack":  # compute macroaverage over datasets
        paths = glob.glob(data_path + "/*")
        for path in paths:
            corpus, queries, qrels = GenericDataLoader(data_folder=path).load(split=split)
            results = retriever.retrieve(corpus, queries)

            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                if isinstance(metric, str):
                    metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                for key, value in metric.items():
                    metrics[key].append(value)
        for key, value in metrics.items():
            assert (
                len(value) == 12
            ), f"cqadupstack includes 12 datasets, only {len(value)} values were compute for the {key} metric"

    metrics = {key: 100 * np.mean(value) for key, value in metrics.items()}

    # Save the results

    if save_results_path is not None:
        save_path = f"{save_results_path}{dataset}/results.json"
        
        with open(save_path, "w") as json_file:
            json.dump(metrics, json_file)

    print(f"Dataset: {dataset}")
    print(metrics)
    print("NDCG@10", metrics["NDCG@10"])
    print("Recall@100", metrics["Recall@100"])
    return metrics
