import os
import json
import random


def split_dataset(
    dataset_name, # dataset_name is the same as save_name in this case
    ratio, 
    input_folder_path, 
    output_folder_root,
    input_type,
    from_split="train", 
    to_split="dev", 
    force=False,
    indexes_file_path="",
    use_indexes_file=False):

    print("Dividing the dataset ...")
    input_file = f"{input_folder_path}{dataset_name}_{input_type}_{from_split}.json"
    if not os.path.exists(input_file):
        print(f"The file {input_file} doesn't exists")
        raise Exception("The file doesn't exists")

    output_file = f"{output_folder_root}{dataset_name}_{input_type}_{to_split}.json"
    if os.path.exists(output_file) and not force:
        print(f"The file {output_file} already exists")
        return True
    
    with open(input_file, "r") as json_file:
        dataset = json.load(json_file)

        length = len(dataset)

        indexes_to_dev = []
        if use_indexes_file:
            with open(indexes_file_path, "r") as json_indexes_file:
                indexes_to_dev = json.load(json_indexes_file)
        else:
            indexes_to_dev = random.choices(list(range(length)), k=int(length*ratio))
            with open(f"{output_folder_root}/dev_indexes.json", "w") as json_indexes_file:
                json.dump(indexes_to_dev, json_indexes_file)

    indexes_to_dev = set(indexes_to_dev)

    dataset_from, dataset_to = [], []

    for i in range(length):
        if i in indexes_to_dev:
            dataset_to.append(dataset[i])
        else:
            dataset_from.append(dataset[i])
    
    if len(dataset_from) + len(dataset_to) != length:
        raise Exception("The lenghts of {from_split} and {to_split} don't math the original dataset")

    # Create a copy just in case
    copy_path = f"{input_folder_path}{dataset_name}_{input_type}_{from_split}_and_{to_split}.json"
    with open(copy_path, "w") as json_file:
        json.dump(dataset, json_file)

    with open(input_file, "w") as json_file:
        json.dump(dataset_from, json_file)

    with open(output_file, "w") as json_file:
        json.dump(dataset_to, json_file)
    
    return True


        



