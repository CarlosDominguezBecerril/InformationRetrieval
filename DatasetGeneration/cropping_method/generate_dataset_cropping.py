
import os
import random
import numpy as np
import json
from tqdm import tqdm
from utils import json_utilities

def randomcrop(x, ratio_min=0.1, ratio_max=0.5):

    ratio = random.uniform(ratio_min, ratio_max)
    length = int(len(x) * ratio)
    start = random.randint(0, len(x) - length)
    end = start + length
    crop = list(x[start:end])
    return crop

def deleteword(x, p=0.1):
    mask = np.random.rand(len(x))
    x = [e for e, m in zip(x, mask) if m > p]
    return x

def convert_contriever(string):
    s = string.split(" ")
    return " ".join(deleteword(randomcrop(s))).lower()

def generate_cropping(documents):

    cropped_document_list = []
    for document in documents:
        cropped_document_list.append(
            {
                "query": convert_contriever(document["text"]),
                "document": convert_contriever(document["text"]),
                "original_document": document["text"],
            }
        )
    return cropped_document_list

def generate_dataset(args, ratio=1, split="train"):
    # Read first the corpus with all the documents
    dataset_path = os.path.join(args["datasets_folder"], args["dataset_name"])
    corpus = json_utilities.read_jsonl_file(f"{dataset_path}/corpus.jsonl")
    corpus_sampled = random.sample(corpus, int(len(corpus) * ratio)) * args["questions_per_document"]

    # Create the dataset
    cropped_dataset = generate_cropping(corpus_sampled)
    json_utilities.save_json_file(f"{args['save_path']}/{split}.json", cropped_dataset)

    # Check if we want to create a dev dataset
    if args["create_dev"]:
        args["create_dev"] = False
        generate_dataset(args, args["dev_ratio"], split="dev")
