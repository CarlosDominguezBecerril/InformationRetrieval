import json

def read_jsonl_file(file_name):
    with open(file_name, 'r') as json_file:
        json_list = list(json_file)

    results = []
    for json_str in json_list:
        results.append(json.loads(json_str))

    return results

def read_json_file(file_name):
    with open(file_name, "r") as json_file:
        results = json.load(json_file)
    return results

def save_json_file(file_name, output):
    with open(file_name, "w") as json_file:
        json.dump(output, json_file)