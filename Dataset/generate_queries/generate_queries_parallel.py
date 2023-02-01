
import os

def divide_work(gpus, shard_list):
    if gpus >= len(shard_list):
        return [[l] for l in shard_list ]

    n = len(shard_list) // gpus

    output = []
    i = 0
    while i < len(shard_list) and len(output) < gpus:
        output.append([])

        while i < len(shard_list) and len(output[-1]) != n:
            output[-1].append(shard_list[i])
            i += 1

    if len(output) != len(shard_list):
        i, j = 0, len(output) * n
        while j < len(shard_list):
            output[i].append(shard_list[j])
            j += 1
            i += 1
    
    return output

def create_slurm_file(process_id, version, shard_list, split, dataset_name, save_name, model_name, batch_size, print_every, do_sample, p, optimize, output_root, input_root, output_file_path, error_file_path, prompt_name="standard", prompt_path="./generate_queries/prompts.json"):

    folder = output_file_path
    output_file_path = output_file_path + f"output_{process_id}_V{version}.txt"
    error_file_path = error_file_path + f"error_{process_id}_V{version}.txt"
    slurm_file = f"""#!/bin/bash
#SBATCH --job-name=preprocess_dataset_{dataset_name}_{process_id}
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10-00:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output={output_file_path}
#SBATCH --error={error_file_path}

# Activamos el entorno virtual que necesitemos
source /tartalo01/users/cdominguez019/tfm_carlos/bin/activate

# LLamamos a nuestro script de python
srun python3 ./generate_queries/generate_queries.py \\
        --shard_list {shard_list} \\
        --split {split} \\
        --dataset_name {dataset_name} \\
        --save_name {save_name} \\
        --model_name {model_name} \\
        --batch_size {batch_size} \\
        --print_every {print_every} \\
        --do_sample {do_sample} \\
        --p {p} \\
        --optimize {optimize} \\
        --prompt_name {prompt_name} \\
        --prompt_path {prompt_path} \\
        --output_root {output_root} \\
        --input_root {input_root}""".replace(",", "").replace("[","").replace("]", "")

    with open(f"{folder}tmp_{process_id}.slurm", "w") as f:
        f.write(slurm_file)

def run_parallel(gpus, version, shard_list, split, dataset_name, save_name, model_name, batch_size, print_every, p, do_sample, optimize, output_root, input_root, prompt_name="standard", prompt_path="./generate_queries/prompts.json"):
    
    print("Generating the queries")
    if len(shard_list) == 0:
        print("Ignoring this step, the dataset is already generated\n")
        return

    shards = divide_work(gpus, shard_list)
    print("GPU division of the shards:")
    for i, s_list in enumerate(shards):
        print(f"GPU{i}: {s_list}")

    for process_id, s_list in enumerate(shards):
        folder = f"{output_root}{save_name}/output_and_error_files/"

        # Create folder
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        folder = f"{folder}{model_name.replace('/', '_')}/"

        # Create folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        output_file_path = error_file_path = folder

        create_slurm_file(process_id, version, s_list, split, dataset_name, save_name, model_name, batch_size, print_every, do_sample, p, optimize, output_root, input_root, output_file_path, error_file_path, prompt_name, prompt_path)

        os.system(f"sbatch {folder}tmp_{process_id}.slurm")


if __name__ == "__main__":
    print(divide_work(8, list(range(2))))
    print(divide_work(8, list(range(8))))
    print(divide_work(8, list(range(20))))
    print(divide_work(2, list(range(20))))
    print(divide_work(3, list(range(20))))
    print(divide_work(4, list(range(20))))
    print(divide_work(5, list(range(20))))
    print(divide_work(3, list(range(1))))
    print(divide_work(5, list(range(3))))

