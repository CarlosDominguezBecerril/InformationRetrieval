from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

import argparse

from collections import defaultdict

import numpy as np

import json

import glob

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def evaluate_dataset(dataset_name, args):

    data_path = os.path.join(args.beir_datasets_path, dataset_name)
    split = "test" if dataset_name != "msmarco" else "dev"
    retriever = args.retriever

    metrics = defaultdict(list)  # store final results

    if not dataset == "cqadupstack":
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        results = retriever.retrieve(corpus, queries)
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
            if isinstance(metric, str):
                metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
            for key, value in metric.items():
                metrics[key].append(value)

    elif dataset_name == "cqadupstack": # compute macroaverage over datasets
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

    else:
        print(f"Dataset with name {dataset_name} doesn't exist")

    metrics = {key: 100 * np.mean(value) for key, value in metrics.items()}

    with open(f"{args.beir_save_path}/results/{dataset_name.replace('/', '_')}_results.json", "w") as json_file:
        json.dump(metrics, json_file)
    
    print(f"Dataset: {dataset}")
    print(metrics)
    print("NDCG@10", metrics["NDCG@10"])
    print("Recall@100", metrics["Recall@100"])

if __name__ == "__main__":

    os.system("pwd")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--beir_datasets_name", nargs="+")
    parser.add_argument("--beir_datasets_path", type=str)
    parser.add_argument("--beir_save_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--similarity", type=str)
    parser.add_argument("--is_dpr", type=str)

    args, _ = parser.parse_known_args()

    is_dpr = False
    if args.is_dpr.lower() == "true":
        is_dpr = True
    
    args.is_dpr = is_dpr

    if args.similarity == "dot_score":
        args.similarity = "dot"

    sep =  " [SEP] " if args.is_dpr else " "
    model = DRES(models.SentenceBERT(args.model_path, sep=sep, is_dpr=args.is_dpr), batch_size=512, corpus_chunk_size=512*9999)
    retriever = EvaluateRetrieval(model, score_function=args.similarity)
    args.retriever = retriever

    for dataset in args.beir_datasets_name:
        evaluate_dataset(dataset, args)



    