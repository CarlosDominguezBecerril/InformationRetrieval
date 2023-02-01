import json
import os

def join_dataset(dataset_name, save_name, model_name, number_of_shards, output_dir=None, output_root="../output/query_generation_format/",  input_root="../output/query_generation_format/", split="train", force=False):

    print("Joining the dataset ...")
    model_name = model_name.replace("/", "_")

    output_root = f"{output_root}{save_name}/"
    input_root = f"{input_root}{save_name}/{dataset_name}_queries_generated/{model_name}/"

    if output_dir is None:
        output_dir = f"{output_root}final_outputs/"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_dir = f"{output_dir}{model_name}/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_path = f"{output_dir}{save_name}_query_generation_format_{split}.json"
    
    if os.path.exists(output_path) and not force:
        print(f"{save_name} dataset exists, continuing ...")
        print(f"Path checked: {output_path}")
        print("Ignoring this step, the dataset is already joined\n")
        return

    all_datasets = []
    for shard in range(number_of_shards):
        with open(f"{input_root}{dataset_name}_query_generated_shard_{shard}_{split}.json", "r") as json_file:
            dataset = json.load(json_file)
            all_datasets += dataset

    print("Finished joining the dataset")
    with open(output_path, "w") as json_file:
        json.dump(all_datasets, json_file)

