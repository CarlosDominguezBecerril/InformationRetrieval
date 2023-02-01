import json
import os
import argparse
import glob
import shutil

def select_best_epoch(save_dir, metric, less_better):
    
    save_dir = f"{save_dir}checkpoint/"

    best_metric, step = -1, ""
    for folder in list(glob.glob(f"{save_dir}*")):
        if "step" in folder:
            with open(f"{folder}/results.json", "r") as json_file:
                results = json.load(json_file)

                if less_better and results[metric] <= best_metric:
                    if results[metric] < best_metric:
                        best_metric = results[metric]
                        step = folder
                    elif results[metric] == best_metric:
                        # If the metric is the same select the earlier epoch
                        if int(folder.split("_")[-1]) < int(step.split("_")[-1]):
                            step = folder
                elif not less_better and  results[metric] >= best_metric:
                    if results[metric] > best_metric:
                        best_metric = results[metric]
                        step = folder
                    elif results[metric] == best_metric:
                        # If the metric is the same select the earlier epoch
                        if int(folder.split("_")[-1]) < int(step.split("_")[-1]):
                            step = folder

    print(f"Best checkpoint metric ({metric}): {best_metric}. Path: {step}.")

    evaluation_checkpoint = f"{save_dir}best_checkpoint/"
    if not os.path.exists(evaluation_checkpoint):
        os.mkdir(evaluation_checkpoint)

    shutil.copyfile(f"{step}/checkpoint.pth", f"{evaluation_checkpoint}checkpoint.pth")


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
    # select_best_epoch("/gscratch3/users/cdominguez019/PhD/InformationRetrieval/Contriever/output/msmarco/query_generation/msmarco_do_sample=True_p=0.9/facebook_opt-30b/", "NDCG@10")
    