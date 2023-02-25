from utils import convert_dataset, download_datasets
import os

if __name__ == "__main__":

    datasets_folder = "datasets"
    save_folder_supervised = "supervised_datasets"
    save_folder_unsupervised = "unsupervised_datasets"

    use_dpr_format = False # True -> Uses DPR format. False -> Beir format

    splits = ["train"] # Possible values: "train", "dev", "test"

    method = "cropping" # Possible: [cropping, LLM, supervised]
    dataset_name = "scifact"
    download_datasets.download_dataset(dataset_name) # You can download all the datasets by calling "download_all_datasets()"

    # Only for LLM
    model_name = "facebook/opt-125m"
    # answers to exclude that are part of the prompt

    exclude_answers = ["Is a little caffeine ok during pregnancy?", "What fruit is native to Australia?", "How large is the Canadian military?"]
    exclude_answers = set([ex_an.lower().strip for ex_an in exclude_answers])

    # Create the save folder
    if method == "supervised":
        save_path = os.path.join(save_folder_supervised, dataset_name)
    else:
        save_name = f"{method}{'_' + model_name if method == 'LLM' else ''}".replace("/", "_")
        save_path = os.path.join(save_folder_unsupervised, dataset_name, save_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    args = {
        "datasets_folder": datasets_folder,
        "save_folder_supervised": save_folder_supervised,
        "save_folder_unsupervised": save_folder_unsupervised,
        "save_path": save_path,
        "method": method,
        "dataset_name": dataset_name,
        "model_name": model_name.replace("/", "_"),
        "exclude_answers": exclude_answers,
        "splits": splits
    }

    convert_dataset.convert(args, use_dpr_format)
