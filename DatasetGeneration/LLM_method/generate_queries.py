import os

os.environ["TRANSFORMERS_CACHE"] = "./transformers_cache"

from torch.utils.data import Dataset, DataLoader
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from datetime import datetime

from dataset import DocumentStore

import argparse

import json
 
def preprocess_output(string, start_point):
    string = string[start_point:]
    first_new_line = string.find("\n") 
    return string[:first_new_line].strip()

def generate_questions(args, shard_id, model, tokenizer, prompt_info):
    torch.cuda.empty_cache()

    ds = DocumentStore(f"{os.path.join(os.path.dirname(args.save_path), 'shards', args.split)}/shard_{shard_id}.json")
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=8, shuffle=False)

    output_path = os.path.join(args.save_path, args.split, "unsupervised_dataset_sharded")
    output_dataset = []
    print(f"\nStarting shard {shard_id}")

    current_time = datetime.now()
    prev_time = current_time
    print(f"current time {current_time}")

    print(f"Output file: {output_path}")
    examples = 0
    errors = 0

    for batch in dl:

        if examples != 0 and examples % args.print_every == 0:
            mid_time = datetime.now()
            diff = mid_time - current_time
            
            minutes = diff.total_seconds() / 60
            print(f"Shard id: {shard_id}")
            print(f"Total difference in hours: {minutes / 60}", flush=True)
            print(f"Progress: {examples}/{len(ds) // args.batch_size}", flush=True)

            diff = mid_time - prev_time
            minutes = diff.total_seconds() / 60

            print(f"Expected_time: {((len(ds) // args.batch_size) * minutes / args.print_every) / 60} hours\n", flush=True)

            prev_time = mid_time

        outputs = batch
        try:
            current_information = [[outputs["document"][i]] for i in range(len(outputs["document"]))]
            for step in range(prompt_info["steps"]):
                prompt = prompt_info[str(step)]["prompt"]
                next_step = prompt_info[str(step)]["next_step"]

                outputs["prompt"] = []
                for i in range(len(outputs["document"])):
                    # Create the prompt
                    outputs["prompt"].append(prompt.format(*current_information[i]) + next_step)

                inputs = tokenizer(outputs["prompt"], return_tensors="pt", padding=True).to("cuda" if torch.cuda.is_available() else "cpu")
                squeeze = inputs.input_ids.squeeze(1)
            
                generate_ids = model.generate(squeeze, max_length=None, max_new_tokens=32, do_sample=args.do_sample, top_p=args.p)
                decoded = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                # Add the new information to the current information array
                for i in range(len(generate_ids)):
                    current_information[i].append(preprocess_output(decoded[i], len(outputs["prompt"][i])))
            # Save the output
            for i in range(len(generate_ids)):
                output_dataset.append({
                    "query": current_information[i][-1],
                    "document": outputs["document"][i],
                })
        except Exception as e:
            print(e)
            print(examples)
            errors += 1

        examples += 1

    with open(f"{output_path}/shard_{shard_id}.json", "w") as f:
        json.dump(output_dataset, f)

    final_time = datetime.now()
    print(final_time)

    diff = final_time - current_time

    minutes = diff.total_seconds() / 60
    print(f'Total difference in minutes: {minutes}\n')
    print(f"End shard {shard_id}")

    return minutes, errors

def load_model_and_prepare_data(args):

    with open(args.prompt_path, "r") as json_file:
        prompt_list = json.load(json_file)
    
    if args.prompt_name not in prompt_list:
        raise Exception("That prompt doesn't exists!")
    
    prompt_info = prompt_list[args.prompt_name]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", load_in_8bit=args.optimization)

    tokenizer.padding_side = "left"
    model.eval()

    for shard_id in list(map(int, args.shard_list)):

        time, errors = generate_questions(args, shard_id, model, tokenizer, prompt_info)

        metadata_save_path = os.path.join(args.save_path, args.split, "metadata")
        metadata = {
            "dataset_name": args.dataset_name,
            "shard_id": shard_id,
            "save_path": args.save_path,
            "this_file_path": metadata_save_path,
            "split": args.split,
            "model_name": args.model_name,
            "time_start": str(datetime.now()),
            "time_end": str(datetime.now()),
            "time_in_minutes": time,
            "errors": errors,
            "optimization": args.optimization,
            "do_sample": args.do_sample,
            "p": args.p,
            "batch_size": args.batch_size,
            "prompt_name": args.prompt_name,
        }

        with open(f"{metadata_save_path}/metadata_{shard_id}.json", "w") as f:
            json.dump(metadata, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--shard_list", nargs="+")
    parser.add_argument("--split", type=str)
    parser.add_argument("--dataset_folder", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--print_every", type=int)
    parser.add_argument("--do_sample", type=str)
    parser.add_argument("--p", type=float)
    parser.add_argument("--optimization", type=str)
    parser.add_argument("--prompt_path", type=str)
    parser.add_argument("--prompt_name", type=str)
    parser.add_argument("--save_path", type=str)

    args, _ = parser.parse_known_args()

    print("Shard list:", list(map(int, args.shard_list)))
    print("Split:", args.split)
    print("Dataset folder:", args.dataset_folder)
    print("Dataset name:", args.dataset_name)
    print("Model name:", args.model_name)
    print("Batch size:", args.batch_size)
    print("Print every:", args.print_every)
    print("p:", args.p)
    print("Prompt name:", args.prompt_name)
    print("Prompt path:", args.prompt_path)
    print("Save path:", args.save_path)
    
    optimize = False
    if args.optimization.lower() == "true":
        args.optimization = True

    do_sample = False
    if args.do_sample.lower() == "true":
        args.do_sample = True
    print("Optimize 8-bit:", args.optimization)
    print("Do sample:", args.do_sample)
    
    load_model_and_prepare_data(args)