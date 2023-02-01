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
    use_negatives=False,
    dev=False, 
    test=False, 
    use_pretrained_model=False,
    pretrained_model_name=""
    ):
    if input_type not in ["contriever", "dpr", "query_generation"]:
        raise Exception(f"The input type '{input_type}'is not valid")
    
    create_folder(output_root)

    model_name = model_name.replace("/", "_")

    save_dir = f"{output_root}{dataset_name}/"
    create_folder(save_dir)

    extra_name = f"{input_type}"
    if use_negatives:
        extra_name += '_with_negatives'
    
    if use_pretrained_model:
        extra_name += f"_pretrained_on_{pretrained_model_name}"

    save_dir = f"{save_dir}{extra_name}/"

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
    dev_name = f"{dataset_name if input_type != 'query_generation' else save_name}_{input_type}_format_dev.json" if dev else None
    test_name = f"{dataset_name if input_type != 'query_generation' else save_name}_{input_type}_format_test.json" if test else None

    return save_dir, doc_dir, train_name, dev_name, test_name, output_and_error_folder_path

def create_slurm(
    gpus,
    save_dir, 
    doc_dir, 
    train_name,
    dev_name, 
    test_name,
    use_negatives=False,
    epochs=40, 
    batch_size=512,
    grad_acc_steps=8,
    query_model="bert-base-uncased", 
    passage_model="bert-base-uncased", 
    slurm_output="./", 
    use_pretrained_model=False, 
    pretrained_model_path="",
    beir_datasets_name=[],
    beir_datasets_path="../Dataset/original_datasets/",
    beir_experiment_name="",
    beir_output_root="../Evaluation/BEIR/output/"
    ):

    output_file_path = f"{slurm_output}output.txt"
    error_file_path = f"{slurm_output}error.txt"
    slurm_file = f"""#!/bin/bash
#SBATCH --job-name=train_dpr
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10-00:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:{gpus}
#SBATCH --output={output_file_path}
#SBATCH --error={error_file_path}

# Activamos el entorno virtual que necesitemos
source /tartalo01/users/cdominguez019/tfm_carlos/bin/activate

# LLamamos a nuestro script de python
srun python3 /gscratch3/users/cdominguez019/PhD/InformationRetrieval/DPR/train_dpr.py \\
        --save_dir_path {save_dir} \\
        --doc_dir_path {doc_dir} \\
        --train_name_path {train_name} \\
        --dev_name_path {dev_name} \\
        --test_name_path {test_name} \\
        --epochs {epochs} \\
        --batch_size {batch_size} \\
        --grad_acc_steps {grad_acc_steps} \\
        --query_model {query_model} \\
        --passage_model {passage_model} \\
        --use_pretrained_model {use_pretrained_model} \\
        --pretrained_model_path {pretrained_model_path} \\
        --use_negatives {use_negatives}
srun python3 /gscratch3/users/cdominguez019/PhD/InformationRetrieval/DPR/utils/best_epoch.py \\
        --save_dir_path {save_dir} \\
        --metric loss \\
        --less_better True
srun python3 /gscratch3/users/cdominguez019/PhD/InformationRetrieval/DPR/utils/convert_to_hf.py \\
        --load_dir {save_dir}best_checkpoint/
srun python3 /gscratch3/users/cdominguez019/PhD/InformationRetrieval/Evaluation/BEIR/eval.py \\
        --save_dir_path {save_dir}best_checkpoint/ \\
        --beir_datasets_name {beir_datasets_name} \\
        --beir_datasets_path {beir_datasets_path} \\
        --beir_experiment_name {beir_experiment_name} \\
        --beir_output_root {beir_output_root} \\
        --is_dpr True""".replace(",", "").replace("[", "").replace("]", "")

    with open(f"{slurm_output}tmp.slurm", "w") as f:
        f.write(slurm_file)

def train_dpr(
    gpus, 
    dataset_name, 
    save_name, 
    model_name, 
    input_type,
    use_negatives=False,
    dev=False,
    test=False,
    epochs=40, 
    batch_size=512,
    grad_acc_steps=8,
    query_model="bert-base-uncased", 
    passage_model="bert-base-uncased", 
    output_root="./output/", 
    input_root="../Dataset/output/",  
    use_pretrained_model=False, 
    pretrained_model_path="", 
    pretrained_model_name="",
    beir_datasets_name=[],
    beir_datasets_path="../Dataset/original_datasets/",
    beir_experiment_name="",
    beir_output_root="../Evaluation/BEIR/output/"):

    save_dir, doc_dir, train_name, dev_name, test_name, slurm_output = create_folder_system_and_obtain_paths(dataset_name, save_name, model_name, input_type, output_root, input_root, use_negatives, dev, test, use_pretrained_model, pretrained_model_name)
    
    create_slurm(gpus, save_dir, doc_dir, train_name, dev_name, test_name, use_negatives=use_negatives, epochs=epochs, batch_size=batch_size, grad_acc_steps=grad_acc_steps, slurm_output=slurm_output, 
    use_pretrained_model=use_pretrained_model, pretrained_model_path=pretrained_model_path, beir_datasets_name=beir_datasets_name, 
    beir_datasets_path=beir_datasets_path, beir_experiment_name=beir_experiment_name, beir_output_root=beir_output_root)
    
    os.system(f"sbatch {slurm_output}tmp.slurm")

if __name__ == "__main__":
    
    # Training
    dataset_name = "scifact"
    prompt_name = "instruct-gpt"
    save_name = f"{dataset_name}_do_sample=True_p=0.9_prompt={prompt_name}" # only if input_type is "query_generation"
    input_type = ["dpr", "contriever", "query_generation"][2]
    model_name = "instruct-gpt"# only if input_type is "query_generation"
    epochs = 30
    batch_size = 128
    grad_acc_steps = 8
    gpus = 1

    # Negatives:
    use_negatives = False

    # Evaluate on dev / test
    # DPR
    dev, test = True, False

    # Pretrained model
    use_pretrained_model = True
    pretrained_model_name = "dpr_supervised_ms_marco"
    pretrained_model_path = "./output/msmarco/dpr/best_checkpoint/"

    # BEIR
    beir_datasets = ["scifact", "nfcorpus"] # list of str / if beir_datasets is ["all"] all beir is checked
    beir_dataset_path = "../Dataset/original_datasets/"
    beir_experiment_name = f"{save_name if input_type == 'query_generation' else dataset_name}_{input_type}_format"
    if use_negatives:
        beir_experiment_name += "with_negatives"
    if input_type == "query_generation":
        beir_experiment_name += f"_{model_name.replace('/', '_')}"
    if use_pretrained_model:
        beir_experiment_name += f"_pretrained_on_{pretrained_model_name}"

    beir_output_root="../Evaluation/BEIR/output/"
    
    # for input_type in ["dpr", "contriever", "query_generation"]:
    train_dpr(gpus, dataset_name, save_name, model_name, input_type, use_negatives=use_negatives, dev=dev, test=test, epochs=epochs, batch_size=batch_size, grad_acc_steps=grad_acc_steps,
    use_pretrained_model=use_pretrained_model, pretrained_model_name=pretrained_model_name, pretrained_model_path=pretrained_model_path,
    beir_datasets_name=beir_datasets, beir_datasets_path=beir_dataset_path, beir_experiment_name=beir_experiment_name, beir_output_root=beir_output_root)
    