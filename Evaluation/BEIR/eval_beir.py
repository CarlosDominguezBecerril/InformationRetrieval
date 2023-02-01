import argparse

from transformers import (
    AutoModel,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoderTokenizerFast,
    DPRContextEncoder,
    DPRQuestionEncoder
)


from beir_utils import evaluate_model
from models import load_retriever

def evaluate_beir_dataset(dataset_name, datasets_path, retriever_path, experiment_save_path, is_dpr):
    
    if is_dpr:
        
        query_encoder_path, passage_encoder_path = f"{retriever_path}query_encoder", f"{retriever_path}passage_encoder"

        # Init & Load Encoders
        query_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=query_encoder_path,
            revision=None,
            do_lower_case=True,
            use_fast=True,
            use_auth_token=None,
        )

        passage_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=passage_encoder_path,
            revision=None,
            do_lower_case=True,
            use_fast=True,
            use_auth_token=None,
        )

        query_encoder_path, passage_encoder_path = f"{retriever_path}dpr_query_encoder", f"{retriever_path}dpr_passage_encoder"

        # query_encoder = AutoModel.from_pretrained(query_encoder_path)
        query_encoder = DPRQuestionEncoder.from_pretrained(query_encoder_path)
        passage_encoder = DPRContextEncoder.from_pretrained(passage_encoder_path)

        doc_encoder = passage_encoder.cuda()
        query_encoder = query_encoder.cuda()

        doc_encoder.eval()
        query_encoder.eval()
    else:
        model, tokenizer, retriever_model_id = load_retriever(retriever_path)
        model = model.cuda()
        model.eval()

        query_encoder = model
        doc_encoder = model

        query_tokenizer = passage_tokenizer = tokenizer

    metrics = evaluate_model(
        query_encoder=query_encoder,
        doc_encoder=doc_encoder,
        query_tokenizer=query_tokenizer,
        doc_tokenizer=passage_tokenizer,
        dataset=dataset_name,
        batch_size=128,
        norm_query=False,
        norm_doc=False,
        split="dev" if dataset_name == "msmarco" else "test",
        score_function="dot",
        beir_dir=datasets_path,
        save_results_path=experiment_save_path,
        lower_case=False,
        normalize_text=False,
        is_dpr=is_dpr,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--beir_datasets_name", nargs="+")
    parser.add_argument("--datasets_path", type=str)
    parser.add_argument("--retriever_path", type=str)
    parser.add_argument("--experiment_save_path", type=str)
    parser.add_argument("--is_dpr", type=str)

    args, _ = parser.parse_known_args()

    is_dpr = False
    if args.is_dpr.lower() == "true":
        is_dpr = True

    print("Datasets to evaluate:", args.beir_datasets_name)
    print("Datasets path:", args.datasets_path)
    print("Retriever path:", args.retriever_path)
    print("Experiment save path:", args.experiment_save_path)
    print("Is DPR:", is_dpr)

    for dataset in args.beir_datasets_name:
        print(f"Evaluating {dataset} ...")
        evaluate_beir_dataset(dataset, args.datasets_path, args.retriever_path, args.experiment_save_path, is_dpr)

