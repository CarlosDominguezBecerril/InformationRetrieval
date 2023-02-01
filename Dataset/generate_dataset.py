import os

import json

from convert_to_dpr_format.convert_to_dpr_format import convert_to_dpr_format
from convert_to_contriever_format.convert_to_contriever_format import convert_to_contriever_format
from generate_queries.shard_dataset import shard_dataset
from generate_queries.generate_queries import run, recover
from generate_queries.generate_queries_parallel import run_parallel
from utils.join_dataset import join_dataset
from utils.split_dataset import split_dataset
from utils.mine_negatives import mine_negatives

def create_folder(folder_path):
    # Create folders
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

if __name__ == "__main__":

    datasets_main_path = "/gscratch3/users/cdominguez019/PhD/InformationRetrieval/Dataset/original_datasets/"

    # Sharding
    mode = "manual"
    shards = 20 # when mode is auto ignore

    # Obtaining dataset info
    dataset_name = "scifact"
    prompt_name = "instruct-gpt"
    save_name = f"{dataset_name}_do_sample=True_p=0.9_prompt={prompt_name}" # It needs to be unique
    gpus = 1
    model_name = "instruct-gpt"
    prompt_path = "./generate_queries/prompts.json"
    batch_size = 10
    print_every = 10
    p = 0.9 
    do_sample = True 
    optimize = True
    split = "train"

    # train - dev split
    ratio = 0.1

    # Mine negatives
    generate_negatives = False

    output_folder = "./output/"
    dpr_folder = f"{output_folder}/dpr_format/"
    contriever_folder = f"{output_folder}/contriever_format/"
    query_generation_folder = f"{output_folder}/query_generation_format/"

    # Create folders
    create_folder(output_folder)
    create_folder(dpr_folder)
    create_folder(contriever_folder)
    create_folder(query_generation_folder)


    # First create the dpr format
    convert_to_dpr_format(dataset_name, root_path=f"{datasets_main_path}{dataset_name}/", output_root="./output/dpr_format/", split=split)

    # Second the contriever format
    convert_to_contriever_format(dataset_name, output_root="./output/contriever_format/", input_root="./output/dpr_format/", split=split)

    # Finally the query generation format
    shard_list = shard_dataset(dataset_name, save_name, mode=mode, shards=shards, output_root="./output/query_generation_format/", input_root="./output/dpr_format/")
    
    shard_list, version, number_of_shards = recover(shard_list, dataset_name, save_name, model_name, output_root="./output/query_generation_format/")
    
    print(shard_list, version, number_of_shards)
    run_parallel(gpus, version, shard_list, split, dataset_name, save_name, model_name, batch_size, print_every, p, do_sample, optimize, output_root="./output/query_generation_format/", input_root="./output/query_generation_format/", prompt_name=prompt_name, prompt_path=prompt_path)
    
    if len(shard_list) == 0:
        join_dataset(dataset_name, save_name, model_name, number_of_shards, output_root="./output/query_generation_format/",  input_root="./output/query_generation_format/")

        if ratio > 0:
            split_dataset(dataset_name, ratio, f"./output/dpr_format/{dataset_name}/", f"./output/dpr_format/{dataset_name}/", "dpr_format")
            split_dataset(dataset_name, ratio, f"./output/contriever_format/{dataset_name}/", f"./output/contriever_format/{dataset_name}/", "contriever_format", use_indexes_file=True, indexes_file_path=f"./output/dpr_format/{dataset_name}/dev_indexes.json")
            split_dataset(save_name, ratio, f"./output/query_generation_format/{save_name}/final_outputs/{model_name.replace('/', '_')}/", f"./output/query_generation_format/{save_name}/final_outputs/{model_name.replace('/', '_')}/", "query_generation_format", use_indexes_file=True, indexes_file_path=f"./output/dpr_format/{dataset_name}/dev_indexes.json")
        
        if generate_negatives:
            # mine_negatives(dataset_name, f"./output/dpr_format/{dataset_name}/", "dpr_format")
            # mine_negatives(dataset_name, f"./output/contriever_format/{dataset_name}/", "contriever_format")
            mine_negatives(save_name, f"./output/query_generation_format/{save_name}/final_outputs/{model_name.replace('/', '_')}/", "query_generation_format")
    