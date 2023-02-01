
import os
import random
import numpy as np
import json
from tqdm import tqdm

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


def convert_to_contriever_format(dataset_name, output_dir=None, split="train", output_root="../output/contriever_format/", input_root="../output/dpr_format/", force=False):
    
    if output_dir is None:
        output_dir = f"{output_root}{dataset_name}/"
    
    # Create folder
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Converting to contriever format ...")

    # Check if dataset exists
    input_path = f"{input_root}{dataset_name}/{dataset_name}_dpr_format_{split}.json"
    if not os.path.exists(input_path):
        print(f"{dataset_name} dataset doesn't exists")
        print(f"Path checked: {input_path}")
        raise Exception("Dataset doesn't exists")

    output_path = f"{output_dir}{dataset_name}_contriever_format_{split}.json"
    if os.path.exists(output_path) and not force:
        print(f"{dataset_name} dataset exists, continuing ...")
        print(f"Path checked: {output_path}")
        print("Ignoring this step, the dataset is already generated\n")
        return 

    new_dataset = []
    with open(input_path, "r") as json_file:
        dataset = json.load(json_file)

        for element in tqdm(dataset):
            element["question"] = convert_contriever(element["positive_ctxs"][0]["text"])

            for i, ctx in enumerate(element["positive_ctxs"]):
                element["positive_ctxs"][i]["text"] = convert_contriever(element["positive_ctxs"][i]["text"])
            
            for i, ctx in enumerate(element["negative_ctxs"]):
                element["negative_ctxs"][i]["text"]  = convert_contriever(element["negative_ctxs"][i]["text"])

            for i, ctx in enumerate(element["hard_negative_ctxs"]):
                element["hard_negative_ctxs"][i]["text"]  = convert_contriever(element["hard_negative_ctxs"][i]["text"])
    
            new_dataset.append(element)

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(dataset, json_file)

if __name__ == "__main__":
    # test the code
    convert_to_contriever_format("scifact")