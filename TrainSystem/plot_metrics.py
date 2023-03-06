import matplotlib.pyplot as plt
import pandas as pd
import json

def plot_metrics(csv_path, output_path, similarity="cos_sim",):

    plt.rcParams["figure.figsize"] = (15, 5)

    df = pd.read_csv(csv_path)

    x = df["epoch"]
    max_row = 3
    max_col = 6
    fig, axs = plt.subplots(max_row, max_col)
    plt.setp(axs, xticks=x)
    i, j = 0, 0
    for column_name in sorted(df.columns):
        if similarity not in column_name:
            continue
        y = df[column_name]
        axs[i, j].plot(x, y)
        axs[i, j].set_title(column_name)
        
        j += 1

        if j == max_col:
            j = 0
            i += 1

    fig.tight_layout()
    fig.savefig(output_path)
    
if __name__ == "__main__":
    eval_path = "./output/msmarco/msmarco_LLM_facebook_opt-2.7b_pretrained_on_distilbert-base-uncased/model_output/eval/"
    csv_path = f"{eval_path}Information-Retrieval_evaluation_eval_results.csv"
    for similarity in ["dot_score", "cos_sim"]:
        output_path = f"{eval_path}{similarity}_metric_plot.png"
        plot_metrics(csv_path, output_path, similarity)