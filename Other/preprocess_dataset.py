from datasets import load_dataset
import json
from tqdm import tqdm

dataset_all = load_dataset("ms_marco", "v2.1")

unique_passage_ids = {}

for split in ["test"]:
    dataset = {}

    dataset_all_split = dataset_all[split]

    for i in tqdm(range(len(dataset_all_split)), mininterval=10):

        example = dataset_all_split[i]

        info = {
            "answers": example["answers"],
            "query": example["query"],
            "query_id": example["query_id"],
            "query_type": example["query_type"],
            "passages": {
                "is_selected": example["passages"]["is_selected"],
            }
        }
        
        passages_list = []

        for passage in example["passages"]["passage_text"]:
            if passage not in unique_passage_ids:
                unique_passage_ids[passage] = len(unique_passage_ids)

            passages_list.append({"passage": passage, "passage_id": unique_passage_ids[passage]})

        info["passages"]["passages_text"] = passages_list

        dataset[i] = info

    with open(f"./preprocessed/ms_marco_preprocessed_{split}.json", "w") as f:
        json.dump(dataset, f)