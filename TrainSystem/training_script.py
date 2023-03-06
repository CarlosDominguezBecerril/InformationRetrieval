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

    word_embedding_model = models.Transformer(args.model_name, max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    #### Or provide pretrained sentence-transformer model
    # model = SentenceTransformer("msmarco-distilbert-base-v3")

    retriever = TrainRetriever(model=model, batch_size=args.batch_size)

    #### Prepare training samples
    train_samples = retriever.load_train(corpus, queries, qrels)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

    if args.similarity == "cos_sim":
        #### Training SBERT with cosine-product
        train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    else:
        #### training SBERT with dot-product
        train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

   
    if args.use_dev:
        #### Prepare dev evaluator
        ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels, dev_batch_size=args.batch_size, main_score_function=args.similarity)
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--similarity", type=str)
    parser.add_argument("--use_dev", type=str)

    args, _ = parser.parse_known_args()

    use_dev = False
    if args.use_dev.lower() == "true":
        use_dev = True
    
    args.use_dev = use_dev

    train(args)