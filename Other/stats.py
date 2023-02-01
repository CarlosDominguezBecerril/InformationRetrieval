import json
from collections import Counter

path = "./hay_stack_format/ms_marco_facebook_opt-125m_x5_8bit_False_do_sample_True_p_0.9_train.json"

with open(path, "r") as json_file:
    dataset = json.load(json_file)

    length = len(dataset) // 5
    
    lengths = []
    for i in range(length):
        diff = set()
        for j in range(5):
            diff.add(dataset[j * length + i]["question"])
        lengths.append(len(diff))

    c = Counter(lengths)

    print(c)
    
    