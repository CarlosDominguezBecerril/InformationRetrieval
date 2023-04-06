import os

def create_folders(args):
    for folder_name in ["output_and_error_files", "model_output"]:
        folder_path = os.path.join(args["save_path"], folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)


def create_slurm(args):
    
    slurm_file = f"""#!/bin/tcsh
#SBATCH --job-name=train_dpr_{args["method"]}_{args["dataset_name"]}
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10-00:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output={args["save_path"]}/output_and_error_files/output.txt
#SBATCH --error={args["save_path"]}/output_and_error_files/error.txt

# Activamos el entorno virtual que necesitemos
source ~/.tcshrc

srun python3 /gscratch3/users/cdominguez019/PhD/InformationRetrieval/TrainSystem/training_script.py \\
        --data_path_train {args["dataset_path_train"]} \\
        --data_path_dev {args["dataset_path_dev"]} \\
        --model_name {args["pretrained_model_name"]} \\
        --model_path_1 {args["pretrained_model_path_1"]} \\
        --model_path_2 {args["pretrained_model_path_2"]} \\
        --batch_size {args["batch_size"]} \\
        --model_save_path {args["save_path"]}/model_output \\
        --epochs {args["epochs"]} \\
        --similarity {args["similarity"]} \\
        --use_dev {args["use_dev"]} \\
        --is_dpr {args["is_dpr"]} \\
        --reload_model {args["reload_model"]}
        
# LLamamos a nuestro script de python
srun python3 /gscratch3/users/cdominguez019/PhD/InformationRetrieval/EvaluateSystem/evaluate_beir.py \\
        --beir_datasets_name {args['beir_datasets']} \\
        --beir_datasets_path {args['beir_datasets_path']} \\
        --beir_save_path {args['beir_save_path']} \\
        --similarity {args['similarity']} \\
        --model_path {args['save_path']}/model_output/ \\
        --is_dpr {args["is_dpr"]}""".replace(",", "").replace("[", "").replace("]", "")

    slurm_output = f"{args['save_path']}/output_and_error_files/tmp.slurm"
    with open(slurm_output, "w") as f:
        f.write(slurm_file)

    os.system(f"sbatch {slurm_output}")


if __name__ == "__main__":
    
    # Shared_weights
    is_dpr = False
    domain_adaptation = False

    save_folder = f"output{'/dpr' if is_dpr else ''}{'/domain_adaptation' if domain_adaptation else ''}"

    # For selecting the appropiate dataset
    method = "supervised" # Possible values: [supervised, LLM, cropping]
    dataset_name = "scifact"
    model_name = "facebook/opt-2.7b"

    # pre-trained model use for training
    reload_model = False
    # if you want to fine-tune a sentence transformer you need to only set-up pretrainer_model_path1

    pretrained_model_name = "distilbert-base-uncased"
    pretrained_model_path_1 = "distilbert-base-uncased"
    pretrained_model_path_2 = "distilbert-base-uncased"

    similarity = "cos_sim" # "dot_score" or "cos_sim"

    epochs = 1
    batch_size = 256
    use_dev = True
    dev_supervised = True

    # Evaluation
    beir_datasets = [dataset_name] # list of str. if beir_datasets is ["all"] all beir is checked
    beir_datasets_path = "../DatasetGeneration/datasets/"


    # Create the save folder
    save_name = f"{similarity}{'_dpr' if is_dpr else ''}_{dataset_name}_{method}{'_' + model_name if method == 'LLM' else ''}_pretrained_on_{pretrained_model_name}".replace("/", "_")

    save_path = os.path.join(save_folder, dataset_name, save_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if method == "supervised":
        dataset_path_train = f"../DatasetGeneration/datasets/{dataset_name}/"
    else:
        dataset_path_train = f"../DatasetGeneration/unsupervised_datasets/{dataset_name}/{method}{'_' + model_name.replace('/', '_') if method == 'LLM' else ''}/beir/"
    
    if dev_supervised:
        dataset_path_dev = f"../DatasetGeneration/datasets/{dataset_name}/"
    else:
        dataset_path_dev = f"../DatasetGeneration/unsupervised_datasets/{dataset_name}/{method}{'_' + model_name.replace('/', '_') if method == 'LLM' else ''}/beir/"


    beir_save_path = f"../EvaluateSystem/output{'/dpr' if is_dpr else ''}{'/domain_adaptation' if domain_adaptation else ''}/{save_name}"

    args = {
        "method": method,
        "dataset_name": dataset_name,
        "dataset_path_train": dataset_path_train,
        "dataset_path_dev": dataset_path_dev,
        "save_path": save_path,
        "pretrained_model_name": pretrained_model_name,
        "pretrained_model_path_1": pretrained_model_path_1,
        "pretrained_model_path_2": pretrained_model_path_2,
        "batch_size": batch_size,
        "epochs": epochs,
        "use_dev": use_dev,
        "beir_datasets": beir_datasets,
        "beir_datasets_path": beir_datasets_path,
        "beir_save_path": beir_save_path,
        "similarity": similarity,
        "is_dpr": is_dpr,
        "reload_model": reload_model,
    }

    create_folders(args)
    create_slurm(args)

    


