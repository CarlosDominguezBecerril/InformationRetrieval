import os

os.environ["TRANSFORMERS_CACHE"] = "./transformers_cache"

from torch.utils.data import Dataset, DataLoader
import torch
import json
from transformers import GPT2Tokenizer, OPTForCausalLM, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from datetime import datetime

from dataset import DPRDataset

from generate_open_ai import generate_open_ai

import argparse

import glob

def preprocess_output(string, start_point):
    string = string[start_point:]
    first_new_line = string.find("\n") 
    return string[:first_new_line].strip()

def generate_questions(shard_id, input_path, output_path, model_name, model, tokenizer, prompt_info, p=0.9, do_sample=True, batch_size=128, print_every=10, device="cpu"):
    torch.cuda.empty_cache()

    ds = DPRDataset(input_path)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=8, shuffle=False)

    output_dataset = []
    print(f"\nStarting shard {shard_id}")

    current_time = datetime.now()
    prev_time = current_time
    print(f"current time {current_time}")

    print(f"Output file: {output_path}")
    examples = 1

    errors = 0
    for batch in dl:

        if examples % print_every == 0:
            mid_time = datetime.now()
            diff = mid_time - current_time
            
            minutes = diff.total_seconds() / 60
            print(f"Shard id: {shard_id}")
            print(f"Total difference in hours: {minutes / 60}", flush=True)
            print(f"Progress: {examples}/{len(ds) // batch_size}", flush=True)

            diff = mid_time - prev_time
            minutes = diff.total_seconds() / 60

            print(f"Expected_time: {((len(ds) // batch_size) * minutes / print_every) / 60} hours\n", flush=True)

            prev_time = mid_time

        outputs = batch
        try:
            current_information = [[outputs["passage"][i]] for i in range(len(outputs["passage"]))]

            for step in range(prompt_info["steps"]):
                prompt = prompt_info[str(step)]["prompt"]
                next_step = prompt_info[str(step)]["next_step"]

                outputs["prompt"] = []
                for i in range(len(outputs["passage"])):
                    # Create the prompt
                    outputs["prompt"].append(prompt.format(*current_information[i]) + next_step)

                inputs = tokenizer(outputs["prompt"], return_tensors="pt", padding=True).to(device)
                squeeze = inputs.input_ids.squeeze(1)
            
                generate_ids = model.generate(squeeze, max_length=None, max_new_tokens=32, do_sample=do_sample, top_p=p)
                decoded = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                # Add the new information to the current information array
                for i in range(len(generate_ids)):
                    current_information[i].append(preprocess_output(decoded[i], len(outputs["prompt"][i])))

            # print("outputs:", outputs)
            # print("Current information", current_information)
            
            # Save the output
            for i in range(len(generate_ids)):
                # print("query preprocessed:", current_information[i][-1])
                output_dataset.append({
                    "dataset": model_name,
                    "question": current_information[i][-1],
                    "answers": ["empty"],
                    "positive_ctxs": [
                        {
                            "title": "",
                            "text": outputs["passage"][i],
                            "score": 0,
                            "title_score": 0,
                            "passage_id": str(outputs["passage_id"][i].item())
                        }
                    ],
                    "negative_ctxs": [],
                    "hard_negative_ctxs": []
                })
        except Exception as e:
            print(e)
            print(examples)
            errors += 1

        examples += 1

    with open(output_path, "w") as f:
        json.dump(output_dataset, f)

    final_time = datetime.now()
    print(final_time)

    diff = final_time - current_time

    minutes = diff.total_seconds() / 60
    print(f'Total difference in minutes: {minutes}\n')
    print(f"End shard {shard_id}")

    return minutes, errors

def recover(shard_list, dataset_name, save_name, model_name, split="train", output_root="../output/query_generation_format/"):

    if os.path.exists(f"{output_root}{save_name}/final_outputs/{model_name.replace('/', '_')}/{save_name}_query_generation_format_train.json"):
        print("The final dataset already exists")
        return [], 1, len(glob.glob(f"{output_root}{save_name}/{dataset_name}_sharded/*"))
    elif len(shard_list) != 0:
        return shard_list, 1, len(shard_list)
    else:

        print("Recovering number of shards remaining ...")
        
        model_name = model_name.replace("/", "_")

        shard_list_path = f"{output_root}{save_name}/{dataset_name}_sharded/"
        generated_queries_path = f"{output_root}{save_name}/{dataset_name}_queries_generated/"
        error_files_path = f"{output_root}{save_name}/output_and_error_files/{model_name}/"

        number_of_shards = len(list(glob.glob(f"{shard_list_path}*")))
        remaining = []
        for i in range(number_of_shards):
            if not os.path.exists(f"{generated_queries_path}{model_name}/{dataset_name}_query_generated_shard_{i}_{split}.json"):
                remaining.append(i)
        
        version = 1
        files = list(glob.glob(f"{error_files_path}*"))

        if len(files) == 0:
            return remaining, 1, number_of_shards

        for name in files:
            file_name = name.split("/")[-1]
            if "error" in file_name:
                version = max(version, int(file_name.split("_")[2].replace(".txt", "")[1:]))

        if len(remaining) == 0:
            print("All the dataset has been generating")
            print("Ignoring this step, the dataset is already generated\n")
        else:
            print("List of shards remaining:", remaining)
            print("Version execution:", version + 1, "\n")

        return remaining, version + 1, number_of_shards

def run(shards, dataset_name, save_name, model_name, output_dir=None, output_root="../output/query_generation_format/",  input_root="../output/query_generation_format/", split="train", optimize_8_bit=False, prompt_name="standard", prompt_path="./prompts.json", p=0.9, do_sample=True, batch_size=128, print_every=10, device="cuda:0"):
    
    # Load prompt information

    with open(prompt_path, "r") as json_file:
        prompt = json.load(json_file)

    if prompt_name not in prompt:
        raise Exception("That prompt doesn't exists!")

    prompt_info = prompt[prompt_name]

    tokenizer, model = None, None
    if not prompt_info["is_external"]:
        # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # model = OPTForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=optimize_8_bit)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=optimize_8_bit)

        tokenizer.padding_side = "left"
        model.eval()

    output_root = f"{output_root}{save_name}/"
    input_root = f"{input_root}{save_name}/{dataset_name}_sharded/"

    if output_dir is None:
        output_dir = f"{output_root}{dataset_name}_queries_generated/"
        
    # Create folder
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_dir = f"{output_root}{dataset_name}_queries_generated/{model_name.replace('/', '_')}/"
        
    # Create folder        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for shard_id in shards:
        input_path = f"{input_root}{dataset_name}_dpr_format_shard_{str(shard_id)}_{split}.json"
        output_path = f"{output_dir}{dataset_name}_query_generated_shard_{str(shard_id)}_{split}.json"
        if prompt_info["is_external"]:
            if prompt_name == "instruct-gpt":
                time, errors = generate_open_ai(shard_id, input_path, output_path, prompt_info, batch_size)
            else:
                raise Exception("The model doesn't exists")
        else:
            time, errors = generate_questions(shard_id, input_path, output_path, model_name.replace("/", "_"), model, tokenizer, prompt_info, p=p, do_sample=do_sample, batch_size=batch_size, print_every=print_every, device=device)

        metadata_file_path = f"{output_dir}{dataset_name}_query_generated_shard_{str(shard_id)}_{split}_info.json"
        metadata = {
            "dataset": save_name,
            "shard": shard_id,
            "input_path": input_path,
            "output_path": output_path,
            "this_file_path": metadata_file_path,
            "split": split,
            "model_name": model_name,
            "time_minutes": time,
            "errors": errors,
            "optimize_8_bit": optimize_8_bit,
            "do_sample": do_sample,
            "p": p,
            "batch_size": batch_size,
            "prompt_name": prompt_name
        }

        with open(metadata_file_path, "w") as json_file:
            json.dump(metadata, json_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--shard_list", nargs="+")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--optimize", type=str)
    parser.add_argument("--prompt_name", type=str)
    parser.add_argument("--prompt_path", type=str, default="./generate_queries/prompts.json")
    parser.add_argument("--output_root", type=str, default="./output/query_generation_format/")
    parser.add_argument("--input_root", type=str, default="./output/query_generation_format/")

    args, _ = parser.parse_known_args()
    
    print("Shard list:", list(map(int, args.shard_list)))
    print("Split:", args.split)
    print("Dataset name:", args.dataset_name)
    print("Save name:", args.save_name)
    print("Model name:", args.model_name)
    print("Batch size:", args.batch_size)
    print("Print every:", args.print_every)
    print("Do sample:", args.do_sample)
    print("p:", args.p)
    print("Optimize 8-bit:", args.optimize)
    print("Prompt name:", args.prompt_name)
    print("Prompt path:", args.prompt_path)
    print("Output root:", args.output_root)
    print("Input root:", args.input_root)

    optimize = False
    if args.optimize.lower() == "true":
        optimize = True

    run(list(map(int, args.shard_list)), args.dataset_name, args.save_name, args.model_name, output_root=args.output_root, input_root=args.input_root, 
    split=args.split, optimize_8_bit=optimize, prompt_name=args.prompt_name, prompt_path=args.prompt_path, p=args.p, do_sample=args.do_sample, batch_size=args.batch_size, print_every=args.print_every, device="cuda:0")
