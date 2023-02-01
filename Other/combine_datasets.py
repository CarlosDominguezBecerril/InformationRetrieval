import json

NAMES_PATH = {
    "/tartalo01/users/cdominguez019/TFM/dataset/ms_marco/hay_stack_format/ms_marco_facebook_opt-125m_8bit_False_do_sample_True_p_0_9_train.json",
    "/gscratch3/users/cdominguez019/dataset_generation/hay_stack_format/ms_marco_facebook_opt-125m_2_8bit_False_do_sample_True_p_0.9_train.json",
    "/gscratch3/users/cdominguez019/dataset_generation/hay_stack_format/ms_marco_facebook_opt-125m_3_8bit_False_do_sample_True_p_0.9_train.json",
    "/gscratch3/users/cdominguez019/dataset_generation/hay_stack_format/ms_marco_facebook_opt-125m_4_8bit_False_do_sample_True_p_0.9_train.json",
    "/gscratch3/users/cdominguez019/dataset_generation/hay_stack_format/ms_marco_facebook_opt-125m_5_8bit_False_do_sample_True_p_0.9_train.json",
}

save_path = "./hay_stack_format/ms_marco_facebook_opt-125m_x5_8bit_False_do_sample_True_p_0.9_train.json"

output = []

for file_name in NAMES_PATH:
    with open(file_name, "r") as json_file:
        dataset = json.load(json_file)
        output += dataset

with open(save_path, "w") as f:
    json.dump(output, f)