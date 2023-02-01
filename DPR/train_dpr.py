
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import FAISSDocumentStore
import torch
import logging

import argparse

import os

def train_dpr(save_dir_path, doc_dir_path, train_name_path, dev_name_path=None, test_name_path=None, use_negatives=False, epochs=40, batch_size=512, grad_acc_steps=8, query_model="bert-base-uncased", passage_model="bert-base-uncased", use_pretrained_model=False, pretrained_model_path=None):

    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.INFO)

    if use_pretrained_model:
        print("Loading pretrained model ...")
        retriever = DensePassageRetriever.load(
            load_dir=pretrained_model_path, 
            document_store=FAISSDocumentStore(sql_url=f"sqlite:///{save_dir_path}output_and_error_files/faiss_document_store.db")
        )
    else:
        retriever = DensePassageRetriever(
            document_store=FAISSDocumentStore(sql_url=f"sqlite:///{save_dir_path}output_and_error_files/faiss_document_store.db"),
            query_embedding_model=query_model,
            passage_embedding_model=passage_model,
            max_seq_len_query=64,
            max_seq_len_passage=256
        ) 

    print("Training model ...")
    retriever.train(
        data_dir=doc_dir_path,
        train_filename=train_name_path,
        dev_filename=dev_name_path,
        test_filename=test_name_path,
        checkpoint_root_dir=save_dir_path,
        n_epochs=epochs,
        batch_size=batch_size,
        grad_acc_steps=grad_acc_steps,
        save_dir=save_dir_path,
        evaluate_every=20000000000, # We wvaluate automatically every epoch
        embed_title=False,
        num_positives=1,
        num_hard_negatives=1 if use_negatives else 0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--save_dir_path", type=str)
    parser.add_argument("--doc_dir_path", type=str)
    parser.add_argument("--train_name_path", type=str)
    parser.add_argument("--dev_name_path", type=str, default=None)
    parser.add_argument("--test_name_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--grad_acc_steps", type=int, default=8)
    parser.add_argument("--query_model", type=str, default="bert-base-uncased")
    parser.add_argument("--passage_model", type=str, default="bert-base-uncased")
    parser.add_argument("--use_pretrained_model", type=str, default="")
    parser.add_argument("--pretrained_model_path", type=str, default="")
    parser.add_argument("--use_negatives", type=str, default="")

    args, _ = parser.parse_known_args()

    print("Save dir path:", args.save_dir_path)
    print("Doc dir path:", args.doc_dir_path)
    print("Train name path:", args.train_name_path)
    print("Dev name path:", args.dev_name_path)
    print("Test name path:", args.test_name_path)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Grad acc steps:", args.grad_acc_steps)
    print("Query model:", args.query_model)
    print("Passage model:", args.passage_model)
    use_pretrained_model = False
    if args.use_pretrained_model.lower() == "true":
        use_pretrained_model = True
    print("Use pretrained model:", use_pretrained_model)
    print("Pretrained model path:", args.pretrained_model_path)
    use_negatives = False
    if args.use_negatives.lower() == "true":
        use_negatives = True

    print("Use negatives:", use_negatives)
    

    dev_name_path = None
    if args.dev_name_path.lower() != "none":
        dev_name_path = args.dev_name_path
    
    test_name_path = None
    if args.test_name_path.lower() != "none":
        test_name_path = args.test_name_path

    train_dpr(args.save_dir_path, args.doc_dir_path, args.train_name_path, dev_name_path, test_name_path, use_negatives, args.epochs, args.batch_size, args.grad_acc_steps, args.query_model, args.passage_model, use_pretrained_model, args.pretrained_model_path)    