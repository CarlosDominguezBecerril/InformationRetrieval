import json_utilities
import os
import csv
import glob
from tqdm import tqdm
import shutil

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

def preprocess(args, split):
    # first join the datasets
    if args["method"] == "LLM":
        dataset_joined = join_datasets(args, split)
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

    return dataset_joined, corpus

def convert_unsupervised_to_beir_format(args, split):

    dataset_joined, corpus = preprocess(args, split)

    s = os.path.join(args["save_path"], "beir")
    os.makedirs(s, exist_ok=True)

    queries = []

    # If exists read the all queries files in order to add new ones.
    if os.path.exists(f"{s}/queries.json"):
        queries = json_utilities.read_json_file(f"{s}/queries.json")
    
    empty = exclude_answers = not_found = repeated_documents = 0
    document_key = "original_document" if args["method"] == "cropping" else "document"

    split_content_path = os.path.join(args["save_path"], "beir", "qrels")
    os.makedirs(split_content_path, exist_ok=True)
    valid = 0
    seen_documents = set()
    print(f"{split_content_path}/{split}.tsv")
    with open(f"{split_content_path}/{split}.tsv", "w") as split_content:
        split_content.write("query-id\tcorpus-id\tscore\n")
        for example in tqdm(dataset_joined):
            if example[document_key] in seen_documents:
                repeated_documents += 1
                continue
            elif len(example["query"]) == 0:
                empty += 1
                continue
            elif example["query"].lower().strip() in args["exclude_answers"] and args["method"] == "LLM":
                exclude_answers += 1
                continue
            elif example[document_key] not in corpus:
                not_found += 1
                continue

            queries.append({
                "_id": str(len(queries)),
                "text": example["query"].strip()
            })

            seen_documents.add(example[document_key])

            valid += 1

            split_content.write(f"{queries[-1]['_id']}\t{(corpus[example[document_key]]['_id'])}\t0\n")

    info = {
        "corpus_length": len(corpus),
        "unsupervised_dataset_length_(without_postprocessing)": len(dataset_joined),
        "unsupervised_dataset_length_(with_postprocessing)": valid,
        "lost_questions_(all)": len(dataset_joined) - valid,
        "lost_questions_(%)": 100 - (valid / len(corpus) * 100),
        "lost_questions_(repeated_documents)": repeated_documents,
        "lost_questions_(empty)": empty,
        "lost_questions_(document_not_found)": not_found,
    }

    if args["method"] == "LLM":
        info["lost_questions_(same_prompt_answer)"] = exclude_answers,

    print(f"[{args['dataset_name']}][{split}] Corpus length: {len(corpus)}")
    print(f"[{args['dataset_name']}][{split}] Unsupervised dataset length (without postprocessing): {len(dataset_joined)}")
    print(f"[{args['dataset_name']}][{split}] Unsupervised dataset length (with postprocessing): {valid}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (ALL): {len(dataset_joined) - valid}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (%): {100 - (valid / len(corpus) * 100)}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (Repeated documents): {repeated_documents}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (empty): {empty}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (document not found): {not_found}")

    if args["method"] == "LLM":
        info["lost_questions_(same_prompt_answer)"] = exclude_answers
        print(f"[{args['dataset_name']}][{split}] Lost questions (same prompt answer): {exclude_answers}")

    json_utilities.save_json_file(f"{split_content_path}/{split}_beir_info.json", info)
    json_utilities.save_json_file(f"{s}/queries.json", queries) 
        

def convert_unsupervised_to_dpr_format(args, split):

    dataset_joined, corpus = preprocess(args, split)

    if dataset_joined == []: return

    dataset = []
    empty = exclude_answers = not_found = repeated_documents = 0
    seen_documents = set()
    document_key = "original_document" if args["method"] == "cropping" else "document"
    for example in tqdm(dataset_joined):
        if example[document_key] in seen_documents:
            repeated_documents += 1
            continue
        elif len(example["query"]) == 0:
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

        seen_documents.add(example[document_key])

    info = {
        "corpus_length": len(corpus),
        "unsupervised_dataset_length_(without_postprocessing)": len(dataset_joined),
        "unsupervised_dataset_length_(with_postprocessing)": len(dataset),
        "lost_questions_(all)": len(dataset_joined) - len(dataset),
        "lost_questions_(%)": 100 - (len(dataset) / len(corpus) * 100),
        "lost_questions_(repeated_documents)": repeated_documents,
        "lost_questions_(empty)": empty,
        "lost_questions_(document_not_found)": not_found,
    }
    if args["method"] == "LLM":
        info["lost_questions_(same_prompt_answer)"] = exclude_answers,

    print(f"[{args['dataset_name']}][{split}] Corpus length: {len(corpus)}")
    print(f"[{args['dataset_name']}][{split}] Unsupervised dataset length (without postprocessing): {len(dataset_joined)}")
    print(f"[{args['dataset_name']}][{split}] Unsupervised dataset length (with postprocessing): {len(dataset)}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (ALL): {len(dataset_joined) - len(dataset)}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (%): {100 - (len(dataset) / len(corpus) * 100)}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (Repeated documents): {repeated_documents}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (empty): {empty}")
    print(f"[{args['dataset_name']}][{split}] Lost questions (document not found): {not_found}")

    if args["method"] == "LLM":
        info["lost_questions_(same_prompt_answer)"] = exclude_answers
        print(f"[{args['dataset_name']}][{split}] Lost questions (same prompt answer): {exclude_answers}")

    s = os.path.join(args["save_path"], "dpr")
    os.makedirs(s, exist_ok=True)
    json_utilities.save_json_file(f"{s}/{split}_dpr_info.json", info)
    json_utilities.save_json_file(f"{s}/{split}_dpr.json", dataset)


def convert_supervised_to_dpr_format(args, split):

    dataset_path = os.path.join(args["save_path"], args["dataset_name"])

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
    
    s = os.path.join(args["save_path"], "dpr")
    os.makedirs(s, exist_ok=True)

    json_utilities.save_json_file(f"{s}/{split}_dpr_info.json", info)
    json_utilities.save_json_file(f"{s}/{split}_dpr.json", dataset)

def convert(args, use_dpr_format=True):

    for split in args["splits"]:
        if args["method"] == "supervised":
            if use_dpr_format:
                convert_supervised_to_dpr_format(args, split)
            else:
                print("The original dataset is already in BEIR format")
        else:
            if use_dpr_format:
                convert_unsupervised_to_dpr_format(args, split)
            else:
                convert_unsupervised_to_beir_format(args, split)

    # Convert to jsonl only if using beir format and copy the corpus
    if not use_dpr_format:
        dst = os.path.join(args['save_path'], 'beir')
        json_utilities.convert_json_to_jsonl(f"{dst}/queries.json")
        shutil.copyfile(f"{os.path.join(args['datasets_folder'], args['dataset_name'])}/corpus.jsonl", f"{dst}/corpus.jsonl")


    