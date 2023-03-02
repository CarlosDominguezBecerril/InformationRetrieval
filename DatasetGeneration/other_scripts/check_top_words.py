import glob
import os
import json
from collections import Counter

def check_top_words(args):
    save_name = f"LLM_{args['model_name']}".replace("/", "_")

    save_path = os.path.join(args["save_folder"], args["dataset_name"], save_name, "dpr")

    for split in args["splits"]:
        top_words = Counter()

        with open(f"{save_path}/{split}_dpr.json", "r") as json_file:
            dataset = json.load(json_file)

        for example in dataset:
            query = example["question"].lower().strip()
            first_word = query.split(" ")[0]
            top_words[first_word.capitalize()] += 1
        
        top_10_words = top_words.most_common(10)
        print(f"[{args['dataset_name']}][{split}] results")
        for i, (word, value) in enumerate(top_10_words):
            print(word, value, value / len(dataset) * 100)

if __name__ == "__main__":

    model_name = "facebook/opt-6.7b"
    dataset_name = "msmarco"
    save_folder = "unsupervised_datasets"
    splits = ["train"]

    args = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "save_folder": save_folder,
        "splits": splits,
    }

    check_top_words(args)