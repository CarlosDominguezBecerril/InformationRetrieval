import glob
import os
import json

def check_errors(args):

    splits = ["train", "dev"]

    save_name = f"LLM_{args['model_name']}".replace("/", "_")

    save_path = os.path.join(args["save_folder"], args["dataset_name"], save_name)

    for split in splits:
        shards = []
        save_split_path = os.path.join(save_path, split)

        if not os.path.exists(save_split_path):
            print(f"The split {split} doesn't exist")
            continue

        dataset_path = os.path.join(save_split_path, "metadata")

        for metadata_path in glob.glob(f"{dataset_path}/*"):
            with open(metadata_path, "r") as json_file:
                metadata = json.load(json_file)

            if metadata["errors"] > 0:
                print(f"{metadata['errors']} errors found in shard {metadata['shard_id']} in split {split}")
                shards.append(metadata['shard_id'])

        shards.sort()
        if len(shards) > 0:
            print(f"We recommend to launch again the following checkpoints for LLM using {args['model_name']} and split {split}", shards)
            if args["delete"]:
                for shard_id in shards:
                    p = os.path.join(save_split_path, "unsupervised_dataset_sharded")
                    shard_path = f"{p}/shard_{shard_id}.json"
                    os.system(f"rm {shard_path}")

                    p = os.path.join(save_split_path, "metadata")
                    metadata_path = f"{p}/metadata_{shard_id}.json"
                    os.system(f"rm {metadata_path}")



if __name__ == "__main__":

    model_name = "facebook/opt-2.7b"
    dataset_name = "msmarco"
    save_folder = "unsupervised_datasets"

    delete = False

    args = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "save_folder": save_folder,
        "delete": delete,
    }

    check_errors(args)