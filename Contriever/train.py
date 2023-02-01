import os

def create_folder(folder_path):
    # Create folders
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

def create_folder_system_and_obtain_paths(
    dataset_name, 
    save_name, 
    model_name, 
    input_type, 
    output_root, 
    input_root,
    ):
    if input_type not in ["contriever", "dpr", "query_generation"]:
        raise Exception(f"The input type '{input_type}'is not valid")
    
    create_folder(output_root)

    model_name = model_name.replace("/", "_")

    save_dir = f"{output_root}{dataset_name}/"
    create_folder(save_dir)

    save_dir = f"{save_dir}{input_type}/"
    create_folder(save_dir)

    if input_type not in ["contriever", "dpr"]:
        save_dir = f"{save_dir}{save_name}/"
        create_folder(save_dir)

        save_dir = f"{save_dir}{model_name}/"
        create_folder(save_dir)

    output_and_error_folder_path = f"{save_dir}output_and_error_files/"
    create_folder(output_and_error_folder_path)

    # Doc dir
    doc_dir = f"{input_root}{input_type}_format/{dataset_name if input_type != 'query_generation' else save_name}/"
    if input_type not in ["contriever", "dpr"]:
        doc_dir = f"{doc_dir}final_outputs/{model_name}/"

    # Name files
    train_name = f"{dataset_name if input_type != 'query_generation' else save_name}_{input_type}_format_train.json"
    train_path = f"{doc_dir}{train_name}"

    return save_dir, train_path, output_and_error_folder_path

def create_slurm(
    gpus,
    save_dir, 
    train_name, 
    dev_list,
    eval_freq=50000,
    save_freq=50000,
    steps=500000, 
    batch_size=2048,
    moco_queue=131072,
    contrastive_mode="moco",
    retriever_model_id="bert-base-uncased", 
    slurm_output="./", 
    beir_datasets_name=[],
    beir_datasets_path="../Dataset/original_datasets/",
    beir_experiment_name="",
    beir_output_root="../Evaluation/BEIR/output/"
    ):

    output_file_path = f"{slurm_output}output.txt"
    error_file_path = f"{slurm_output}error.txt"
    slurm_file_headers = f"""#!/bin/bash
#SBATCH --job-name=train_contriever
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=10-00:00:00
#SBATCH --mem=32GB
#SBATCH --output={output_file_path}
#SBATCH --error={error_file_path}
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
"""

    slurm_file_1 = f"""{slurm_file_headers}
#SBATCH --gres=gpu:{gpus}
#SBATCH --ntasks-per-node={gpus}
# Activamos el entorno virtual que necesitemos
source /tartalo01/users/cdominguez019/tfm_carlos/bin/activate
port=$(shuf -i 15000-16000 -n 1)
srun python3 /gscratch3/users/cdominguez019/PhD/InformationRetrieval/Contriever/train_contriever.py \\
        --output_dir {save_dir} \\
        --train_data {train_name} \\
        --eval_datasets {dev_list} \\
        --total_steps {steps} \\
        --per_gpu_batch_size {batch_size} \\
        --moco_queue {moco_queue} \\
        --retriever_model_id {retriever_model_id} \\
        --query_model {retriever_model_id} \\
        --contrastive_mode {contrastive_mode} \\
        --eval_freq {eval_freq} \\
        --save_freq {save_freq} \\
        --main_port $port""".replace(",", "").replace("[", "").replace("]", "")

    slurm_file_2 = f"""{slurm_file_headers}
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
source /tartalo01/users/cdominguez019/tfm_carlos/bin/activate
srun python3 /gscratch3/users/cdominguez019/PhD/InformationRetrieval/Contriever/utils/best_epoch.py \\
        --save_dir_path {save_dir} \\
        --metric NDCG@10 \\
        --less_better False
srun python3 /gscratch3/users/cdominguez019/PhD/InformationRetrieval/Evaluation/BEIR/eval.py \\
        --save_dir_path {save_dir}checkpoint/best_checkpoint/ \\
        --beir_datasets_name {beir_datasets_name} \\
        --beir_datasets_path {beir_datasets_path} \\
        --beir_experiment_name {beir_experiment_name} \\
        --beir_output_root {beir_output_root} \\
        --is_dpr False""".replace(",", "").replace("[", "").replace("]", "")

    slurm_file_3 = f"""#!/bin/bash
echo "running first job: Contriever"
jid1=$(sbatch {slurm_output}tmp_1.slurm | cut -d ' ' -f4)
echo "running second job: Best checkpoint and evaluation"
jid2=$(sbatch --dependency=afterok:$jid1 {slurm_output}tmp_2.slurm)
    """

    with open(f"{slurm_output}tmp_1.slurm", "w") as f:
        f.write(slurm_file_1)
    
    with open(f"{slurm_output}tmp_2.slurm", "w") as f:
        f.write(slurm_file_2)

    with open(f"{slurm_output}tmp_3.sh", "w") as f:
        f.write(slurm_file_3)

def train_contriever(
    gpus, 
    dataset_name, 
    save_name, 
    model_name, 
    input_type,
    dev_list=[],
    eval_freq=50000,
    save_freq=50000,
    steps=500000, 
    batch_size=2048,
    moco_queue=131072,
    contrastive_mode="moco",
    retriever_model_id="bert-base-uncased",  
    output_root="./output/",
    input_root="../Dataset/output/",  
    beir_datasets_name=[],
    beir_datasets_path="../Dataset/original_datasets/",
    beir_experiment_name="",
    beir_output_root="../Evaluation/BEIR/output/"):

    save_dir, train_name, slurm_output = create_folder_system_and_obtain_paths(dataset_name, save_name, model_name, input_type, output_root, input_root)
    
    create_slurm(gpus, save_dir, train_name, dev_list, eval_freq=eval_freq, save_freq=save_freq, steps=steps, moco_queue=moco_queue, contrastive_mode=contrastive_mode, retriever_model_id=retriever_model_id, batch_size=batch_size, slurm_output=slurm_output, beir_datasets_name=beir_datasets_name, beir_datasets_path=beir_datasets_path, beir_experiment_name=beir_experiment_name, beir_output_root=beir_output_root)
    
    os.system(f"sh {slurm_output}tmp_3.sh")

if __name__ == "__main__":
    
    # Training
    dataset_name = "msmarco"
    save_name = f"{dataset_name}_do_sample=True_p=0.9" # only if input_type is "query_generation"
    input_type = ["dpr", "contriever", "query_generation"][2]
    model_name = "facebook/opt-30b" # only if input_type is "query_generation"
    # 4 GPUS | 31250 | 256
    steps = 500000                # 1GPU INFO # 500000    # 31250 # 62500 # 125000
    batch_size_per_gpu = 256      # 1GPU INFO # 64        # 1024  # 512   # 256
    moco_queue = 16384      
    gpus = 2
    contrastive_mode = "moco" # moco or inbatch
    eval_freq = save_freq = steps // 10 # we create 10 checkpoints 

    # Evaluate on dev / test
    # BE CAREFUL, NOT ALL THE DATASET HAVE A DEV/TEST
    dev_eval = ["msmarco"] # list of the beir datasets to evaluate. This can be used for model selection

    # BEIR
    beir_datasets = ["all"] # list of str / if beir_datasets is ["all"] all beir is checked
    beir_dataset_path = "../Dataset/original_datasets/"
    beir_experiment_name = f"{save_name if input_type == 'query_generation' else dataset_name}_{input_type}_format"
    if input_type == "query_generation":
        beir_experiment_name += f"_{model_name.replace('/', '_')}"

    beir_output_root="../Evaluation/BEIR/output/"
    
    # for input_type in ["dpr", "contriever", "query_generation"]:
    train_contriever(gpus, dataset_name, save_name, model_name, input_type, dev_list=dev_eval, eval_freq=eval_freq, save_freq=save_freq, steps=steps, batch_size=batch_size_per_gpu, moco_queue=moco_queue,
    contrastive_mode=contrastive_mode, beir_datasets_name=beir_datasets, beir_datasets_path=beir_dataset_path, beir_experiment_name=beir_experiment_name, beir_output_root=beir_output_root)
    