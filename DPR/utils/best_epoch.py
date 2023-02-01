import json
import os
import argparse
import glob
import shutil

def select_best_epoch(save_dir, metric, less_better=False):
    
    save_dir = f"{save_dir}"

    best_metric, step = float("+inf") if less_better else float("-inf"), ""
    for folder in list(glob.glob(f"{save_dir}*")):
        if "epoch" in folder:
            with open(f"{folder}/results.json", "r") as json_file:
                results = json.load(json_file)
                if less_better and results[0][metric] <= best_metric:
                    if results[0][metric] < best_metric:
                        best_metric = results[0][metric]
                        step = folder
                    elif results[0][metric] == best_metric:
                        # If the metric is the same select the earlier epoch
                        if int(folder.split("_")[-1]) < int(step.split("_")[-1]):
                            step = folder
                elif not less_better and results[0][metric] >= best_metric:
                    if results[0][metric] > best_metric:
                        best_metric = results[0][metric]
                        step = folder
                    elif results[0][metric] == best_metric:
                        # If the metric is the same select the earlier epoch
                        if int(folder.split("_")[-1]) < int(step.split("_")[-1]):
                            step = folder

    print(f"Best checkpoint metric: {best_metric}. Path: {step}.")
    evaluation_checkpoint = f"{save_dir}best_checkpoint/"
    shutil.copytree(step, evaluation_checkpoint)

    # copy tokenizer
    f_names = ["special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt"]
    for name in f_names:
        shutil.copy2(f"{save_dir}query_encoder/{name}", f"{save_dir}best_checkpoint/query_encoder/{name}")
        shutil.copy2(f"{save_dir}passage_encoder/{name}", f"{save_dir}best_checkpoint/passage_encoder/{name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--save_dir_path", type=str)
    parser.add_argument("--metric", type=str)
    parser.add_argument("--less_better", type=str, default="False")
    args, _ = parser.parse_known_args()

    print("Selecting best epoch")
    print("Save dir path:", args.save_dir_path)
    print("Metric:", args.metric)

    less_better = False
    if args.less_better.lower() == "true":
        less_better = True

    print("Less better", less_better)
    
    select_best_epoch(args.save_dir_path, args.metric, less_better)
    
    # select_best_epoch("../output/scifact/query_generation_pretrained_on_dpr_supervised_ms_marco/scifact_do_sample=True_p=0.9/facebook_opt-30b/", "loss", True)
    