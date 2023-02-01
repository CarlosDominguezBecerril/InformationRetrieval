import json
from collections import defaultdict, Counter

# dataset_path = "./contriever_format/test.txt"
# dataset_path = "./preprocessed/ms_marco_preprocessed_test.json"
dataset_path = "/gscratch3/users/cdominguez019/PhD/InformationRetrieval/Dataset/output/query_generation_format/msmarco_do_sample=True_p=0.9/final_outputs/facebook_opt-30b/msmarco_do_sample=True_p=0.9_query_generation_format_train.json"
# dataset_path = "./dpr_contriever_format/dpr_contriever_train.json"
# dataset_path = "./dpr_format/ms_marco_supervised.json"

total = 0
with open(dataset_path, "r") as f:
    dataset = json.load(f)
    print(len(dataset))
    
    c = Counter()
    not_question = 0
    for i in range(len(dataset)):
        if dataset[i]["question"] == "":
            not_question += 1
            continue
        first_word = dataset[i]["question"].strip().split(" ")[0].lower()

        c[first_word] += 1
    
    print(c.most_common(10))
    print(f"Total: {len(dataset)}")
    print(f"No question: {not_question}")
    print(f"Valid questiosn:{len(dataset) - not_question}")
        

# dpr_contriever_ no questions 1789/8069749
"""
with open(dataset_path2, "r") as f:
    dataset = json.load(f)

    seen2 = defaultdict(int)
    total = 0
    for key, info in dataset.items():
        for passage in info["passages"]["passages_text"]:
            seen2[passage["passage"]] += 1
            total += 1
    print(total)

distribution1 = defaultdict(int)
distribution2 = defaultdict(int)

for key, value in seen.items():
    distribution1[value] += 1
    # if distribution1[value] > 100000:
    #     print(key)

print("-------------")

for key, value in seen2.items():
    distribution2[value] += 1
    # if distribution2[value] > 100000:
    #     print(key)

print(sorted(tuple(distribution1.items())))
print(sorted(tuple(distribution2.items())))

# 125m -> 8069749
# 350m -> 8069749
# 1,3b -> 8069749
"""