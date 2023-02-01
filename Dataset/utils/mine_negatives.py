from rank_bm25 import BM25Okapi
import os
import json

def mine_negatives(
    dataset_name, # dataset_name is the same as save_name in this case
    input_folder_path, 
    input_type,
    split="train", 
    force=True,
    output_folder_root=None):

    print("Generating negatives ...")
    input_file = f"{input_folder_path}{dataset_name}_{input_type}_{split}.json"
    if not os.path.exists(input_file):
        print(f"The file {input_file} doesn't exists")
        raise Exception("The file doesn't exists")

    if output_folder_root is None:
        output_file = input_file
    else:
        output_file = f"{output_folder_root}{dataset_name}_{input_type}_{split}.json"
    
    if os.path.exists(output_file) and not force:
        print(f"The file {output_file} already exists")
        return True

    with open(input_file, "r") as json_file:
        dataset = json.load(json_file)

    # First copy all the corpus
    corpus = []
    for example in dataset:
        corpus.append(example["positive_ctxs"][0]["text"])
    
    # tokenized corpus
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    # Create the bm25 object
    bm25 = BM25Okapi(tokenized_corpus)

    # Mine negatives
    for i, example in enumerate(dataset):
        query = example["question"]
        tokenized_query = query.split(" ")
        
        doc_scores = bm25.get_scores(tokenized_query)

        minimum_score, minimum_index = float("+inf"), -1
        for j, score in enumerate(doc_scores):
            if score < minimum_score:
                minimum_score = score
                minimum_index = j
        
        dataset[i]["hard_negative_ctxs"].append(
            {
               "title": dataset[minimum_index]["positive_ctxs"][0]["title"],
               "text": dataset[minimum_index]["positive_ctxs"][0]["text"],
               "score": dataset[minimum_index]["positive_ctxs"][0]["score"],
               "title_score": dataset[minimum_index]["positive_ctxs"][0]["title_score"],
               "passage_id": dataset[minimum_index]["positive_ctxs"][0]["passage_id"]
            }
        )

    with open(output_file, "w") as json_file:
        json.dump(dataset, json_file)