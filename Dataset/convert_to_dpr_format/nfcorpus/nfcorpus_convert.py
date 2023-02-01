import pandas as pd
import json
import os

def convert_nfcorpus(root_path, output_dir, split="train"):

    file_path = f'{root_path}qrels/{split}.tsv'

    if not os.path.exists(file_path):
        print(f"The following file doesn't exists: {file_path}")
        return False
        
    # Read TSV file into DataFrame
    df = pd.read_table(file_path)

    corpus = {}
    with open(f"{root_path}corpus.jsonl", "r") as f:
        for line in f:
            info = json.loads(line)
            corpus[info["_id"]] = info

    queries = {}
    with open(f"{root_path}queries.jsonl", "r") as f:
        for line in f:
            info = json.loads(line)
            queries[info["_id"]] = info

    dataset = []
    passage_id_to_int = {}
    for i, row in df.iterrows():
        if row["corpus-id"] not in passage_id_to_int:
            passage_id_to_int[row["corpus-id"]] = len(passage_id_to_int)

        dataset.append({
            "dataset": "nfcorpus",
            "question": queries[row["query-id"]]["text"],
            "answers": "",
            "positive_ctxs": [{
                "title": corpus[row["corpus-id"]]["title"],
                "text": corpus[row["corpus-id"]]["text"],
                "score": int(row["score"]),
                "title_score": 0,
                "passage_id": passage_id_to_int[row["corpus-id"]]
            }],
            "negative_ctxs": [],
            "hard_negative_ctxs": []
        })

    with open(f"{output_dir}nfcorpus_dpr_format_{split}.json", "w", encoding="utf-8") as json_file:
        json.dump(dataset, json_file)

    return True