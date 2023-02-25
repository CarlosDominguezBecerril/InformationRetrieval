import matplotlib.pyplot as plt
import os
import json
import glob

def plot_time(args):

    splits = ["train", "dev"]



    for split in splits:
        
        times = []
        for model_name in model_names:
            total_time = 0
            save_name = f"LLM_{model_name[0]}".replace("/", "_")
            save_path = os.path.join(args["save_folder"], args["dataset_name"], save_name)
            
            save_split_path = os.path.join(save_path, split)

            if not os.path.exists(save_split_path):
                print(f"The split {split} doesn't exist")
                continue

            dataset_path = os.path.join(save_split_path, "metadata")

            for metadata_path in glob.glob(f"{dataset_path}/*"):
                with open(metadata_path, "r") as json_file:
                    metadata = json.load(json_file)

                    total_time += metadata["time_in_minutes"]

            times.append(total_time / 60 / 24) # Convert to days

        x = args["sizes_in_number"]
        y1 = times
        y2 = [y / 8 for y in y1]

        # plot
        plt.plot(x, y1, linewidth=2.0, marker="o", label="Using 1 GPU")
        plt.plot(x, y2, linewidth=2.0, marker="o", label="Using 8 GPUs")

        plt.yscale("log")
        plt.title("Time to generate the dataset\n(Number of parameters vs. time)")
        plt.xlabel("Number of parameters (in billions)")
        plt.ylabel("Number of days (log scale)")
        plt.legend()

        plt.savefig(f"./other_scripts/plot_time_vs_parameters_{split}.png")
        plt.show()
        plt.clf()

if __name__ == "__main__":

    sizes = ["125m", "350m", "1.3b", "2.7b", "6.7b", "13b"]
    sizes_in_number = [0.125, 0.350, 1.3, 2.7, 6.7, 13]
    model_names = [(f"facebook/opt-{size}", size) for size in sizes]
    dataset_name = "msmarco"
    save_folder = "unsupervised_datasets"

    args = {
        "sizes": sizes,
        "sizes_in_number": sizes_in_number,
        "model_names": model_names,
        "dataset_name": dataset_name,
        "save_folder": save_folder,
    }

    plot_time(args)