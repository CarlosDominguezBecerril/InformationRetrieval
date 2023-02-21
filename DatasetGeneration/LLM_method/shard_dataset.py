from utils import json_utilities
import glob
import json
import os
import random

def shard_dataset(args, ratio=1, split="train"):

    save_path = os.path.join(os.path.dirname(args["save_path"]), "shards", split)

    if len(glob.glob(f"{save_path}/*")) > 0:
        print("The dataset is already sharded, ignoring this step")
        return

    # Read first the corpus with all the documents
    dataset_path = os.path.join(args["datasets_folder"], args["dataset_name"])
    corpus = json_utilities.read_jsonl_file(f"{dataset_path}/corpus.jsonl")
    # Sample the corpus without repetition
    corpus_sampled = random.sample(corpus, int(len(corpus) * ratio)) * args["questions_per_document"]

    dataset_length = len(corpus_sampled)

    shards = dataset_length // args["shard_size"]
    shards = max(shards, 1)

    shard_length = dataset_length // shards

    shards_start_end = [[shard_length * i, shard_length * i + shard_length] for i in range(shards)]

    shards_start_end[-1][1] = dataset_length

    for i, (start, end) in enumerate(shards_start_end):
        print(f"Shard from: {start} to {end}")
        json_utilities.save_json_file(f"{save_path}/shard_{i}.json", corpus_sampled[start:end])
    

def check_missing_shards(args, split="train"):
    # Function to check which shards have not been executed yet
    shards_path = os.path.join(os.path.dirname(args["save_path"]), "shards", split)
    shards_list = [os.path.basename(path) for path in list(glob.glob(f"{shards_path}/*"))]

    completed_shards_path = os.path.join(args["save_path"], split, "unsupervised_dataset_sharded")
    completed_shards_list = set([os.path.basename(path) for path in list(glob.glob(f"{completed_shards_path}/*"))])

    missing = []
    for shard in shards_list:
        if shard not in completed_shards_list:
            missing.append(int(shard.split("_")[1].split(".")[0]))
    
    return missing
