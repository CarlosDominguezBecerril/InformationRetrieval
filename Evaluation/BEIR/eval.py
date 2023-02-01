import os
import argparse

def create_folder(folder_path):
    # Create folders
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

def create_folder_system(experiment_name, datasets, is_dpr, output_root="./output/"):
    create_folder(output_root)
    
    experiment_save_path = f"{output_root}{'dpr' if is_dpr else 'contriever'}/"
    create_folder(experiment_save_path)
    
    experiment_save_path = f"{experiment_save_path}{experiment_name}/"

    create_folder(experiment_save_path)
    create_folder(f"{experiment_save_path}output_and_error_files/")

    for dataset in datasets:
        create_folder(f"{experiment_save_path}{dataset}/")

    return experiment_save_path

def create_slurm(model_path, dataset_name, dataset_path, experiment_save_path, is_dpr):
    
    output_file_path = f"{experiment_save_path}output_and_error_files/output.txt"
    error_file_path = f"{experiment_save_path}output_and_error_files/error.txt"
    slurm_path = f"{experiment_save_path}output_and_error_files/tmp.slurm"

    slurm_file = f"""#!/bin/bash
#SBATCH --job-name=test_beir
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
srun python3 /gscratch3/users/cdominguez019/PhD/InformationRetrieval/Evaluation/BEIR/eval_beir.py \\
        --retriever_path {model_path} \\
        --beir_datasets_name {dataset_name} \\
        --datasets_path {dataset_path} \\
        --experiment_save_path {experiment_save_path} \\
        --is_dpr {is_dpr}""".replace(",", "").replace("[", "").replace("]", "")

    with open(slurm_path, "w") as f:
        f.write(slurm_file)
    
    return slurm_path

def evaluate_all_datasets(dataset_path, experiment_name, model_path, is_dpr, output_root="./output/"):

    # Usually scifact is the last one, but is the first in for sanity check as it can be evaluated in 1 minute
    beir_datasets = [
        "scifact", "msmarco", "trec-covid", "nfcorpus", "nq", "hotpotqa", "fiqa", "arguana", 
        "webis-touche2020", "cqadupstack", "quora", "dbpedia-entity", "scidocs", 
        "fever", "climate-fever"
    ]

    evaluate_dataset(beir_datasets, dataset_path, experiment_name, model_path, is_dpr, output_root)

def evaluate_dataset(datasets_name, dataset_path, experiment_name, model_path, is_dpr, output_root="./output/"):
    if isinstance(datasets_name, str):
        datasets_name = [datasets_name]

    experiment_name = experiment_name.replace("/", "_")

    experiment_save_path = create_folder_system(experiment_name, datasets_name, is_dpr, output_root)

    slurm_path = create_slurm(model_path, datasets_name, dataset_path, experiment_save_path, is_dpr)
    
    os.system(f"sbatch {slurm_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # BEIR
    parser.add_argument("--save_dir_path", type=str)
    parser.add_argument("--beir_datasets_name", nargs="+")
    parser.add_argument("--beir_datasets_path", type=str)
    parser.add_argument("--beir_experiment_name", type=str)
    parser.add_argument("--beir_output_root", type=str)
    parser.add_argument("--is_dpr", type=str)

    args, _ = parser.parse_known_args()

    is_dpr = False
    if args.is_dpr.lower() == "true":
        is_dpr = True

    print("Model Path:", args.save_dir_path)
    print("BEIR Evaluation")
    print("BEIR datasets name:", args.beir_datasets_name)
    print("BEIR datasets path:", args.beir_datasets_path)
    print("BEIR experiment name:", args.beir_experiment_name)
    print("BEIR output root:", args.beir_output_root)
    print("BEIR is dpr:", is_dpr)

    if args.beir_datasets_name == ["all"]:
        evaluate_all_datasets(args.beir_datasets_path, args.beir_experiment_name, args.save_dir_path, is_dpr, args.beir_output_root)
    else:
        evaluate_dataset(args.beir_datasets_name, args.beir_datasets_path, args.beir_experiment_name, args.save_dir_path, is_dpr, args.beir_output_root)

    """
    # For testing
    # Output paths
    output_root="./output/"
    experiment_name = "nfcorpus_dpr_format_pretrained_on_dpr_supervised_ms_marco"
    model_path = "/gscratch3/users/cdominguez019/PhD/InformationRetrieval/DPR/output/nfcorpus/dpr_pretrained_on_dpr_supervised_ms_marco/best_checkpoint/"
    dataset_name = "nfcorpus" # str or list of str
    dataset_path = "../../Dataset/original_datasets/"
    is_dpr = True
    evaluate_dataset(dataset_name, dataset_path, experiment_name, model_path, is_dpr, output_root)
    """
    