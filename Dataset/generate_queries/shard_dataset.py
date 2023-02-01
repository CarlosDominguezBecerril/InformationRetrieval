import json
import os
import glob

def shard_dataset(dataset_name, save_name=None, mode="auto", shards=128, output_dir=None, split="train", output_root="../output/query_generation_format/", input_root="../output/dpr_format/", force=False):
    
    if save_name is None:
        save_name = dataset_name

    output_root = f"{output_root}{save_name}/"

    if output_dir is None:
        output_dir = f"{output_root}/{dataset_name}_sharded/"

    # Create folders
    if not os.path.exists(output_root):
        os.mkdir(output_root)

    print("Generating sharded dataset ...")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    elif not force:
        print("Ignoring this step, the dataset is already generated\n")
        return []
    else:
        # Clean all the files
        for file_name in list(glob.glob(f"{output_dir}*")):
            if os.path.isfile(file_name):
                os.remove(file_name)

    # Check if dataset exists
    input_path = f"{input_root}{dataset_name}/{dataset_name}_dpr_format_{split}.json"
    if not os.path.exists(input_path):
        print(f"{dataset_name} dataset doesn't exists")
        print(f"Path checked: {input_path}")
        raise Exception("Dataset doesn't exists")

    dataset = None
    with open(input_path, "r") as json_file:
        dataset = json.load(json_file)

    dataset_length = len(dataset)

    print(f"Length of the {dataset_name} dataset: {dataset_length}")

    if mode == "auto":
        shards = dataset_length // 10000
        shards = max(shards, 1)
    
    shard_length = dataset_length // shards
    shards_start_end = [[shard_length * i, shard_length * i + shard_length] for i in range(shards)]
    
    shards_start_end[-1][1] = len(dataset)

    for i, (start, end) in enumerate(shards_start_end):

        print(f"Shard from: {start} to {end}")
 
        with open(f"{output_dir}{dataset_name}_dpr_format_shard_{i}_{split}.json", "w") as json_file:
            json.dump(dataset[start:end], json_file)
    
    return list(range(shards))

if __name__ == "__main__":
    shard_dataset("scifact", "scifact2", mode="manual", shards=6, force=True)