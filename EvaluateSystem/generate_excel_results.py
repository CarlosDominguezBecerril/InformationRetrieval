import json

import os

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment
import glob

from collections import defaultdict


def generate_excel(metrics, model_names, datasets):

    titles = ["model name", "average"] + datasets
    wb = Workbook()

    # create sheets:
    for metric in metrics:
        ws1 = wb.create_sheet(title=metric)

        ws1["A1"] = metric
        for i, title in enumerate(titles):
            ws1[f"{chr(ord('a') + i)}2"] = title
    
    del wb[wb.sheetnames[0]]

    row = 3
    for model_name in model_names:
        for metric in metrics:
            ws = wb[metric]
            ws[f"{chr(ord('a') + 0)}{row}"] = model_name
            ws[f"{chr(ord('a') + 0)}{row}"].alignment = Alignment(wrap_text=True)

        path = f"output/{model_name}/results/"
        for dataset in datasets:
            results_path = f"{path}{dataset}_results.json"

            if not os.path.exists(results_path):
                continue

            with open(results_path) as json_file:
                results = json.load(json_file)
            
            # Get the index of the dataset
            index = titles.index(dataset)
            for metric in metrics:
                # Get the sheet
                ws = wb[metric]
                ws[f"{chr(ord('a') + index)}{row}"] = results[metric]

        row += 1

    wb.save(filename="./results.xlsx")


if __name__ == "__main__":
    metrics = ["NDCG@10", "Recall@100", "MRR@10"]

    datasets = [
            "msmarco", "trec-covid", "nfcorpus", "nq", "hotpotqa", "fiqa", "arguana", 
            "webis-touche2020", "cqadupstack", "quora", "dbpedia-entity", "scidocs", 
            "fever", "climate-fever", "scifact"
    ]

    model_names = [
        "msmarco_LLM_facebook_opt-125m_pretrained_on_distilbert-base-uncased",
        "msmarco_LLM_facebook_opt-350m_pretrained_on_distilbert-base-uncased",
        "msmarco_LLM_facebook_opt-6.7b_pretrained_on_distilbert-base-uncased",
    ]

    generate_excel(metrics, model_names, datasets)