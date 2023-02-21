from utils import json_utilities, download_datasets
import os
from cropping_method import generate_dataset_cropping
from LLM_method import generate_dataset_llm


def create_dataset(args):
    if args["method"] == "cropping":
        generate_dataset_cropping.generate_dataset(args)
    elif args["method"] == "LLM":
        generate_dataset_llm.generate_dataset(args)
    else:
        print(f"The method {args['method']} to create the unsupervised dataset doesn't exist")


if __name__ == "__main__":

    datasets_folder = "datasets"
    save_folder = "unsupervised_datasets"

    # Select dataset and whether you want to use cropping (contriever generation method) or a LLM (Large Language Model)

    method = "LLM" # Possible: cropping or LLM
    dataset_name = "msmarco"
    download_datasets.download_dataset(dataset_name) # You can download all the datasets by calling "download_all_datasets()"
    create_dev = True
    dev_ratio = 0.2 # We generate a dev dataset that is equal to 20% of the whole corpus.
    questions_per_document = 1
    
    # Update these parameters if you are using the LLM method
    batch_size = 64
    gpus = 1
    shard_size = 100000 # We shard the dataset is slices of "shard_size" to parallelize in several GPUs and avoid OOM issues sometimes. Sometimes the shard size is bigger / smaller than what you put here to divide the dataset evenly.

    print_every = 10
    do_sample = True
    p = 0.9
    optimization = False # Apply 8-bit optimization

    model_name = "facebook/opt-350m"

    prompts_path = "LLM_method/prompts.json"
    prompt = "standard"

    # Create the save folder
    save_name = f"{method}{'_' + model_name if method == 'LLM' else ''}".replace("/", "_")

    save_path = os.path.join(save_folder, dataset_name, save_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    # Save all the arguments in a dict and start creating the dataset
    args = {
       "datasets_folder": datasets_folder,
       "method": method,
       "dataset_name": dataset_name,
       "create_dev": create_dev,
       "dev_ratio": dev_ratio,
       "questions_per_document": questions_per_document,
       "batch_size": batch_size,
       "gpus": gpus,
       "shard_size": shard_size,
       "print_every": print_every,
       "do_sample": do_sample,
       "p": p,
       "optimization": optimization,
       "model_name": model_name,
       "prompts_path": prompts_path,
       "prompt": prompt,
       "save_path": save_path,
    }

    create_dataset(args)

