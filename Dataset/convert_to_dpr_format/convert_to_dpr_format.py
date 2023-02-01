import os

from scifact.scifact_convert import convert_scifact
from nfcorpus.nfcorpus_convert import convert_nfcorpus

def convert_to_dpr_format(dataset_name, root_path, output_dir=None, split="train", output_root="../output/dpr_format/", force=False):
    
    if output_dir is None:
        output_dir = f"{output_root}{dataset_name}/"

    # Create folder
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Converting to dpr format ...")
    output_path = f"{output_dir}{dataset_name}_dpr_format_{split}.json"
    if os.path.exists(output_path) and not force:
        print(f"{dataset_name} dataset exists, continuing ...")
        print(f"Path checked: {output_path}")
        print("Ignoring this step, the dataset is already generated\n")
        return True

    if dataset_name == "scifact":
        return convert_scifact(root_path, output_dir, split)
    elif dataset_name == "nfcorpus":
        return convert_nfcorpus(root_path, output_dir, split)
    else:
        raise Exception("The dataset doesn't exists")
        
if __name__ == "__main__":
    # test the code
    convert_to_dpr_format("scifact", "/gscratch3/users/cdominguez019/PhD/InformationRetrieval/Dataset/original_datasets/scifact")
