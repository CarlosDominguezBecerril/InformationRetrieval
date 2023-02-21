import json_utilities
import os
import csv
import glob
from tqdm import tqdm

def apply_template(dataset, question, positive_ctxs, answers="", negative_ctxs=[], hard_negative_ctxs=[]):
    # Check format here: https://github.com/facebookresearch/DPR
    return {
        "dataset": dataset,
        "question": question,
        "answers": answers,
        "positive_ctxs": positive_ctxs,
        "negative_ctxs": negative_ctxs,
        "hard_negative_ctxs": hard_negative_ctxs,
    }

def join_datasets(args, split):
    # Load the unsupervised generated queries
    if not os.path.exists(os.path.join(args["save_path"], split)):
        print(f"[{args['dataset_name']}][{split}] this split doesn't exist")
        return []
    shards_path = os.path.join(os.path.dirname(args["save_path"]), "shards", split)
    shards_list = [os.path.basename(path) for path in list(glob.glob(f"{shards_path}/*"))]

    completed_shards_path = os.path.join(args["save_path"], split, "unsupervised_dataset_sharded")
    completed_shards_list = set([os.path.basename(path) for path in list(glob.glob(f"{completed_shards_path}/*"))])

    if len(shards_list) != len(completed_shards_list):
        print(f"Your dataset doesn't seem complete. Original dataset length: {len(shards_list)} shards. Your dataset length: {len(completed_shards_list)}")
        exit()

    dataset = []
    for path in list(glob.glob(f"{completed_shards_path}/*")):
        dataset.extend(json_utilities.read_json_file(path))
    
    return dataset


def convert_unsupervised_to_dpr_format(args, split):

    # first join the datasets
    if args["method"] == "LLM":
        dataset_joined = join_datasets(args, split)
        if dataset_joined == []: return
    else:
        if not os.path.exists(f"{args['save_path']}/{split}.json"):
            print(f"[{args['dataset_name']}][{split}] this split doesn't exist")
            return
        dataset_joined = json_utilities.read_json_file(f"{args['save_path']}/{split}.json")

    # Load the corpus (we will reuse the original ids)
    dataset_path = os.path.join(args["datasets_folder"], args["dataset_name"])
    corpus_list = json_utilities.read_jsonl_file(f"{dataset_path}/corpus.jsonl")
    corpus = {}
    for c in corpus_list:
        corpus[c["text"]] = {
            "_id": c["_id"],
            "title": c["title"],
        }

    dataset = []
    empty = exclude_answers = not_found = 0
    document_key = "original_document" if args["method"] == "cropping" else "document"

    for example in tqdm(dataset_joined):
        if len(example["query"]) == 0:
            empty += 1
            continue
        elif example["query"].lower().strip() in args["exclude_answers"] and args["method"] == "LLM":
            exclude_answers += 1
            continue
        elif example[document_key] not in corpus:
            not_found += 1
            continue
        
        dataset.append(apply_template(
            dataset=f"{args['dataset_name']}_unsupervised_{args['method']}{'_' + args['model_name']if args['method'] == 'LLM' else ''}", 
            question=example["query"].strip(), 
            positive_ctxs=[{
                "title": corpus[example[document_key]]["title"],
                "text": example["document"],
                "score": 0,
                "title_score": 0,
                "passage_id": str(corpus[example[document_key]]["_id"]),
            }], 
        ))

    info = {
        "corpus_length": len(corpus),
        "unsupervised_dataset_length_(without_postprocessing)": len(dataset_joined),
        "unsupervised_dataset_length_(with_postprocessing)": len(dataset),
        "lost_questions_(all)": len(dataset_joined) - len(dataset),
        "lost_questions_(empty)": empty,
        "lost_questions_(document_not_found):": not_found
    }
    if args["method"] == "LLM":
        info["lost_questions_(same_prompt_answer)"] = exclude_answers,

    print(f"[{args['dataset_name']}][{split}] Corpus length: {len(corpus)}")
    print(f"[{args['dataset_name']}][{split}] Unsupervised dataset length (without postprocessing): {len(dataset_joined)}")
    print(f"[{args['dataset_name']}][{split}] Unsupervised dataset length (with postprocessing): {len(dataset)}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (ALL): {len(dataset_joined) - len(dataset)}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (empty): {empty}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (document not found): {not_found}")

    if args["method"] == "LLM":
        info["lost_questions_(same_prompt_answer)"] = exclude_answers
        print(f"[{args['dataset_name']}][{split}] Lost questions (same prompt answer): {exclude_answers}")

    if len(corpus) != len(dataset_joined):
        print("[ONLY FOR TRAIN SPLIT] If the number of queries is different from the dataset length there might be an error")

    json_utilities.save_json_file(f"{args['save_path']}/{split}_dpr_info.json", info)
    json_utilities.save_json_file(f"{args['save_path']}/{split}_dpr.json", dataset)


def convert_supervised_to_dpr_format(args, split):

    dataset_path = os.path.join(args["datasets_folder"], args["dataset_name"])

    # Load the corpus, queries and the qrels
    corpus_list = json_utilities.read_jsonl_file(f"{dataset_path}/corpus.jsonl")
    corpus = {}
    for c in corpus_list:
        corpus[str(c["_id"])] = {
            "text": c["text"],
            "title": c["title"],
        }

    queries_list = json_utilities.read_jsonl_file(f"{dataset_path}/queries.jsonl")
    queries = {}
    for q in queries_list:
        queries[str(q["_id"])] = q["text"]

    qrels_path = os.path.join(args["datasets_folder"], args["dataset_name"], "qrels")
    qrels_split_path = f"{qrels_path}/{split}.tsv"

    if not os.path.exists(qrels_split_path):
        print(f"The {split} for {args['dataset_name']} doesn't exist. Ignoring ...")
        return

    qrels = {}

    reader = csv.reader(open(qrels_split_path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
        
    for id, row in enumerate(reader):
        query_id, corpus_id, score = str(row[0]), str(row[1]), int(row[2])
            
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score

    # Create the dataset in DPR format
    dataset = []
    for query_id, query_values in tqdm(qrels.items()):
        for corpus_id, score in query_values.items():
            dataset.append(apply_template(
                dataset=f"{args['dataset_name']}_supervised", 
                question=queries[query_id], 
                positive_ctxs=[{
                    "title": corpus[corpus_id]["title"],
                    "text": corpus[corpus_id]["text"],
                    "score": int(score),
                    "title_score": 0,
                    "passage_id": corpus_id,
                }], 
            ))

    info = {
        "original_length": len(dataset),
        "number_of_queries": len(qrels),
    }

    print(f"[{args['dataset_name']}][{split}] Number of queries: {len(qrels)}")
    print(f"[{args['dataset_name']}][{split}] Dataset length: {len(dataset)}")
    
    if len(qrels) != len(dataset):
        print("[ONLY FOR TRAIN SPLIT] If the number of queries is different from the dataset length there might be an error")
    
    json_utilities.save_json_file(f"{args['save_path']}/{split}_dpr_info.json", info)
    json_utilities.save_json_file(f"{args['save_path']}/{split}_dpr.json", dataset)

def convert_to_dpr_format(args):
    for split in ["train", "dev", "test"]:

        save_dataset_path = f"{args['save_path']}/{split}_dpr.json"
        if os.path.exists(save_dataset_path):
            print(f"[{args['dataset_name']}][{split}][{args['method']}]{'[' + args['model_name'] + ']' if args['method'] == 'LLM' else ''} dataset already exists. Path: {save_dataset_path}")
            continue

        if args["method"] == "supervised":
            convert_supervised_to_dpr_format(args, split)
        else:
            convert_unsupervised_to_dpr_format(args, split)


    