import os
from shard_dataset import shard_dataset, check_missing_shards
from utils import divide_workload

def create_folders(args, split="train"):

    # Create the folder for the shards
    shards_path = os.path.join(os.path.dirname(args["save_path"]), "shards", split)
    if not os.path.exists(shards_path):
        os.makedirs(shards_path, exist_ok=True)

    # Create the folder for the split
    save_path = os.path.join(args["save_path"], split)

    # Create the folders
    for folder_name in ["unsupervised_dataset_sharded", "metadata", "output_and_error_files"]:
        folder_path = os.path.join(save_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

def create_slurm_file(args, process_id, jobs_list, split):

    save_path = os.path.join(args["save_path"], split, "output_and_error_files")

    slurm_file = f"""#!/bin/tcsh
#SBATCH --job-name=preprocess_dataset_{args["dataset_name"]}_{process_id}
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10-00:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --output={save_path}/output_{process_id}.txt
#SBATCH --error={save_path}/error_{process_id}.txt

source ~/.tcshrc

# LLamamos a nuestro script de python
srun python3 ./LLM_method/generate_queries.py \\
        --shard_list {jobs_list} \\
        --split {split} \\
        --datasets_folder {args["datasets_folder"]} \\
        --dataset_name {args["dataset_name"]} \\
        --model_name {args["model_name"]} \\
        --batch_size {args["batch_size"]} \\
        --batch_size_decrease_in_error {args["batch_size_decrease_in_error"]} \\
        --print_every {args["print_every"]} \\
        --do_sample {args["do_sample"]} \\
        --p {args["p"]} \\
        --optimization {args["optimization"]} \\
        --prompt_path {args["prompts_path"]} \\
        --prompt_name {args["prompt"]} \\
        --save_path {args["save_path"]}""".replace(",", "").replace("[","").replace("]", "")
    
    slurm_path = f"{save_path}/tmp_{process_id}.slurm"
    with open(slurm_path, "w") as f:
        f.write(slurm_file)

    os.system(f"sbatch {slurm_path}")

def generate_dataset(args, ratio=1, split="train"):
    
    # first we create the folders to store everything
    create_folders(args, split)
    
    # Second step is to shard the dataset so that we can parallelize the generation in several GPUs
    shard_dataset(args, ratio, split)
    
    # Third step is to check whether do we need to recover from a previous execution
    shards = check_missing_shards(args, split)

    # Fourth step is to divide the shards in the gpus
    work_division = divide_workload.divide_work(args["gpus"], shards)

    print(f"[{split}] GPU division of the shards:")
    for i, s_list in enumerate(work_division):
        print(f"GPU{i}: {s_list}")

    # Fifth step is to execute the jobs according to the work division
    for process_id, job_list_per_gpu in enumerate(work_division):
        create_slurm_file(args, process_id, job_list_per_gpu, split)
        
    # Last step is to check whether we need to do something with the dev dataset
    if args["create_dev"]:
        args["create_dev"] = False
        generate_dataset(args, ratio=args["dev_ratio"], split="dev")