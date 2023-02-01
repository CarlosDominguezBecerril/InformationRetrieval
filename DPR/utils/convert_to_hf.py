from haystack.modeling.model.biadaptive_model import BiAdaptiveModel
from haystack.nodes import DensePassageRetriever
from transformers import AutoConfig, AutoModel, DPRContextEncoder
import torch
import time
import os

import argparse


def convert_to_hugginface(load_dir):
    print("Converting to huggingface model")
    create_hf_models(load_dir)
    move_weights(load_dir)

def create_hf_models(load_dir):
    model = BiAdaptiveModel.load(load_dir=load_dir, device="cpu", lm1_name="query_encoder", lm2_name="passage_encoder")

    transformers_query_encoder, transformers_passage_encoder = model.convert_to_transformers()
    transformers_query_encoder.save_pretrained(f"{load_dir}dpr_query_encoder")
    transformers_passage_encoder.save_pretrained(f"{load_dir}dpr_passage_encoder")

def move_weights(load_dir):
    for encoder_name in ["query", "passage"]:
        dpr_path = f"{load_dir}dpr_{encoder_name}_encoder"

        reloaded_retriever = DensePassageRetriever.load(load_dir=load_dir, document_store=None)
        if encoder_name == "query":
            try:
                retriever_model = reloaded_retriever.model.module.language_model1 
            except: 
                retriever_model = reloaded_retriever.model.language_model1 
        else:
            try:
                retriever_model = reloaded_retriever.model.module.language_model2
            except:
                retriever_model = reloaded_retriever.model.language_model2

        retriever_weights = dict(retriever_model.named_parameters())
        
        if encoder_name == "query":
            dpr_encoder = AutoModel.from_pretrained(dpr_path)
        else:
            dpr_encoder = DPRContextEncoder.from_pretrained(dpr_path)

        dpr_weights = dict(dpr_encoder.named_parameters())

        print(dpr_weights.keys())

        k = 6
        prefix = ""
        if encoder_name == "query": prefix = ""
        # "question_encoder.bert_model."

        print("before")

        for key, value in retriever_weights.items():
            if key[k:].replace(prefix, "") not in dpr_weights:
                print(key[k:].replace(prefix, ""), "not found")
            else:
                print(key[k:].replace(prefix, ""), torch.equal(value.cpu(), dpr_weights[key[k:].replace(prefix, "")].cpu()))
        
        
        # Copying
        for key, value in retriever_weights.items():
            # print(key, key in source.keys())
            if key[k:].replace(prefix, "") not in dpr_weights.keys():
                continue
            dpr_weights[key[k:].replace(prefix, "")].data.copy_(value.data)

        print("after")
        for key, value in retriever_weights.items():
            if key[k:].replace(prefix, "") not in dpr_weights:
                print(key[k:].replace(prefix, ""), "not found")
            else:
                print(key[k:].replace(prefix, ""), torch.equal(value.cpu(), dpr_weights[key[k:].replace(prefix, "")].cpu()))

        dpr_encoder.save_pretrained(dpr_path)

if __name__ == "__main__":
    # convert_to_hugginface("/gscratch3/users/cdominguez019/PhD/InformationRetrieval/DPR/output/msmarco/ms_marco_test/")
    # convert_to_hugginface("/gscratch3/users/cdominguez019/PhD/InformationRetrieval/DPR/output/nfcorpus/query_generation_pretrained_on_dpr_supervised_ms_marco/nfcorpus/facebook_opt-30b/")
    # convert_to_hugginface("/gscratch3/users/cdominguez019/PhD/InformationRetrieval/DPR/output/scifact/query_generation_pretrained_on_dpr_supervised_ms_marco/scifact_do_sample=True_p=0.9/facebook_opt-30b/")
    # convert_to_hugginface("./output/scifact/query_generation_pretrained_on_dpr_supervised_ms_marco/scifact_do_sample=True_p=0.9/facebook_opt-30b/")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--load_dir", type=str)
    args, _ = parser.parse_known_args()

    convert_to_hugginface(args.load_dir)