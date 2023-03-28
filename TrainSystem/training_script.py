import os

os.environ["TRANSFORMERS_CACHE"] = "./transformers_cache"

from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib
import logging

import argparse

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def train(args):

    #### Provide the data_path where nfcorpus has been downloaded and unzipped
    print(f"Train path: {args.data_path_train}")
    corpus, queries, qrels = GenericDataLoader(args.data_path_train).load(split="train")
    print(f"Corpus length: {len(corpus)}. Queries length: {len(queries)}. qrels length: {len(qrels)}")
    #### Please Note not all datasets contain a dev split, comment out the line if such the case
    dev_corpus, dev_queries, dev_qrels = None, None, None
    if args.use_dev:
        print(f"Dev path: {args.data_path_dev}")
        dev_corpus, dev_queries, dev_qrels = GenericDataLoader(args.data_path_dev).load(split="dev")
        print(f"Corpus length: {len(dev_corpus)}. Queries length: {len(dev_queries)}. qrels length: {len(dev_qrels)}")

    #### Provide any sentence-transformers or HF model
    if args.reload_model:
        print("Reloading the model")
        model = SentenceTransformer(args.model_path_1)
    else:
        if args.is_dpr:
            query_embedding_model = models.Transformer(args.model_path_1, max_seq_length=64)
            query_pooling_model = models.Pooling(query_embedding_model.get_word_embedding_dimension())

            document_embedding_model = models.Transformer(args.model_path_2, max_seq_length=256)
            document_pooling_model = models.Pooling(document_embedding_model.get_word_embedding_dimension())
            
            asym_model = models.Asym({'query': [query_embedding_model, query_pooling_model], 'doc': [document_embedding_model, document_pooling_model]})
            model = SentenceTransformer(modules=[asym_model])
        else:
            word_embedding_model = models.Transformer(args.model_path_1, max_seq_length=256)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    #### Or provide pretrained sentence-transformer model
    # model = SentenceTransformer("msmarco-distilbert-base-v3")
    sep =  " [SEP] " if args.is_dpr else " "
    retriever = TrainRetriever(model=model, batch_size=args.batch_size, sep=sep)

    #### Prepare training samples
    train_samples = retriever.load_train(corpus, queries, qrels, is_dpr=args.is_dpr)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

    if args.similarity == "cos_sim":
        #### Training SBERT with cosine-product
        train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    else:
        #### training SBERT with dot-product
        train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

   
    if args.use_dev:
        #### Prepare dev evaluator
        ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels, dev_batch_size=args.batch_size, main_score_function=args.similarity, is_dpr=args.is_dpr)
    else:
        #### If no dev set is present from above use dummy evaluator
        ir_evaluator = retriever.load_dummy_evaluator()

    #### Provide model save path
    model_save_path = args.model_save_path
    os.makedirs(model_save_path, exist_ok=True)

    #### Configure Train params
    num_epochs = args.epochs
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

    retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                    evaluator=ir_evaluator, 
                    epochs=num_epochs,
                    output_path=model_save_path,
                    warmup_steps=warmup_steps,
                    use_amp=True,
                    checkpoint_path=os.path.join(model_save_path, "checkpoints"),
                    checkpoint_save_steps=len(train_dataloader))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path_train", type=str)
    parser.add_argument("--data_path_dev", type=str)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--model_path_1", type=str, default="distilbert-base-uncased")
    parser.add_argument("--model_path_2", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--similarity", type=str)
    parser.add_argument("--use_dev", type=str)
    parser.add_argument("--is_dpr", type=str)
    parser.add_argument("--reload_model", type=str)

    args, _ = parser.parse_known_args()

    use_dev = False
    if args.use_dev.lower() == "true":
        use_dev = True
    
    args.use_dev = use_dev

    is_dpr = False
    if args.is_dpr.lower() == "true":
        is_dpr = True
    
    args.is_dpr = is_dpr

    reload_model = False
    if args.reload_model.lower() == "true":
        reload_model = True
    
    args.reload_model = reload_model

    print("data_path_train:", args.data_path_train)
    print("data_path_dev:", args.data_path_dev)
    print("model_name:", args.model_name)
    print("model_path_1:", args.model_path_1)
    print("model_path_2 (only for DPR):", args.model_path_2)
    print("batch_size:", args.batch_size)
    print("model_save_path:", args.model_save_path)
    print("epochs:", args.epochs)
    print("similarity:", args.similarity)
    print("use_dev:", args.use_dev)
    print("is_dpr:", args.is_dpr)

    train(args)