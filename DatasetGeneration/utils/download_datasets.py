
import os
import beir.util

def download_dataset(dataset_name, beir_dir="datasets"):

    data_path = os.path.join(beir_dir, dataset_name)
    if not os.path.isdir(data_path):
        print(f"Downloading dataset {dataset_name}")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = beir.util.download_and_unzip(url, beir_dir)

def download_all_datasets(beir_dir="datasets"):

    beir_datasets = [
        "scifact", "msmarco", "trec-covid", "nfcorpus", "nq", "hotpotqa", "fiqa", "arguana", 
        "webis-touche2020", "cqadupstack", "quora", "dbpedia-entity", "scidocs", 
        "fever", "climate-fever"
    ]

    for dataset_name in beir_datasets:
        download_dataset(dataset_name, beir_dir)

if __name__ == "__main__":
    download_all_datasets()
    