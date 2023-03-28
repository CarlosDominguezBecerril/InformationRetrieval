import os
import argparse

def create_folders(args):
    for folder_name in ["output_and_error_files", "results"]:
        folder_path = os.path.join(args.beir_save_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

def create_slurm(args):
    slurm_file = f"""#!/bin/tcsh
#SBATCH --job-name=test_beir
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10-00:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output={args.beir_save_path}/output_and_error_files/output.txt
#SBATCH --error={args.beir_save_path}/output_and_error_files/error.txt

# Activamos el entorno virtual que necesitemos
source ~/.tcshrc

# LLamamos a nuestro script de python
srun python3 /gscratch3/users/cdominguez019/PhD/InformationRetrieval/EvaluateSystem/evaluation_script.py \\
        --beir_datasets_name {args.beir_datasets_name} \\
        --beir_datasets_path {args.beir_datasets_path} \\
        --beir_save_path {args.beir_save_path} \\
        --similarity {args.similarity} \\
        --model_path {args.model_path} \\
        --is_dpr {args.is_dpr}""".replace(",", "").replace("[", "").replace("]", "")

    slurm_path = f"{args.beir_save_path}/output_and_error_files/tmp.slurm"
    with open(slurm_path, "w") as f:
        f.write(slurm_file)

    os.system(f"sbatch {slurm_path}")

def evaluate_datasets(args):
    if isinstance(args.beir_datasets_name, str):
        args.beir_datasets_name = [args.beir_datasets_name]
    
    if args.beir_datasets_name == ["all"]:
        args.beir_datasets_name = [
            "scifact", "msmarco", "trec-covid", "nfcorpus", "nq", "hotpotqa", "fiqa", "arguana", 
            "webis-touche2020", "cqadupstack", "quora", "dbpedia-entity", "scidocs", 
            "fever", "climate-fever"
    ]

    create_folders(args)
    create_slurm(args)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--beir_datasets_name", nargs="+")
    parser.add_argument("--beir_datasets_path", type=str)
    parser.add_argument("--beir_save_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--similarity", type=str)
    parser.add_argument("--is_dpr", type=str)
    args, _ = parser.parse_known_args()

    print("BEIR Evaluation")
    print("BEIR datasets name:", args.beir_datasets_name)
    print("BEIR datasets path:", args.beir_datasets_path)
    print("BEIR save path:", args.beir_save_path)
    print("BEIR model path:", args.model_path)
    print("Model DPR:", args.is_dpr)

    evaluate_datasets(args)