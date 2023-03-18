# InformationRetrieval

The repository pertains to the field of Information Retrieval and focuses on the implementation of an unsupervised question generation method for the purpose of information retrieval. It encompasses the following processes:

- Generation of questions through an unsupervised approach.
- Training of a dense passage retrieval system (DPR) using the generated questions as a basis. (Using haystack https://github.com/deepset-ai/haystack)
- Training of the contriever system using the generated questions. (https://github.com/facebookresearch/contriever).
- Evaluation of the system using the benchmarked evaluation method BEIR. (https://github.com/beir-cellar/beir)


Further information about the project can be found at: https://drive.google.com/file/d/1JGQH_SvFMhNlcYqrM3beMTCaWJznO7j8/view?usp=sharing

# How to use the code

Note: The code makes use of Slurm to simplify the workflow of the project.

## Question Generation

To get started, download the datasets by executing the following command:

```
python3 utils/download_datasets.py
```

You can generate the dataset by executing the following command inside the 'DatasetGeneration' folder:

```
python generate_unsupervised_dataset.py
```

Important parameters in the script include:
- method: the values are "cropping" or "LLM". Cropping stands for contriever generation method and LLM for large language model.
- dataset name: name of the dataset to generate questions for.
- create dev: whether to create a dev dataset or not.
- dev_rato: how many documents to use from the corpus to create the dev split. (float. 0.2 -> 20% of the documents)
- question_per_document: how many questions to create per document.

This parameters are only useful when using LLM method
- gpus: specify the number of GPUs to use.
- prompt name: "prompt" used to generate the questions. (See prompts.json for the different available options).
- model name: name of the model to use (HuggingFace).
- shard size: size of the shard to create the shard of the dataset (this is used to parallelize the generation process across different GPUs).


If the system stops during the generation of the dataset using the LLM method, you can run the script again to recover from where it was left.

After generating the dataset run the following command (update the necessary parameters).

```
python3 postprocessing.py
```

### (OTHER) Using your own dataset

To use your dataset own dataset use the BEIR format. You will need to create a file called "corpus.jsonl" inside the "datasets/my_dataset" folder.

The format should be one dictionary per line with the following keys:

"_id" -> Unique id of the document

"title" -> Title of the document

"text" -> Document


# Training

**Note**: Before training check the modifications I made to the BEIR and Sentence Transformers source code. The modifications are available at the end of this README

Training is as easy as calling to 
```
python train.py
```

Make sure that the parameters are set correctly according to what you generated during the dataset generation, and update other parameters such as batch size.

You can also evaluate on BEIR by updating the 'beir_datasets' variable.

Multi-gpu is not supported due to sentence transformers.

# Evaluation (BEIR)

Ideally, you should set up the datasets you want to evaluate while training. If you forget to add a dataset, you can evaluate it by running the following command in the BEIR folder:

```
python evaluate_beir.py
```

# Results

Coming soon.

# Pretrained models and datasets

Pretrained models and datasets are available under request.

# Changes in BEIR and Sentence Transformers to make the code usable

Some changes has been made to the BEIR library and sentence transformers.

## BEIR:

### File: beir/beir/retrieval/train.py

load_ir_evaluator

- add two input parameters dev_batch_size and main_score_function
- change return function. return InformationRetrievalEvaluator(queries, corpus, rel_docs, name=name, batch_size=dev_batch_size, main_score_function=main_score_function)

Updated code with the changes:
```
    def load_ir_evaluator(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], 
                 qrels: Dict[str, Dict[str, int]], max_corpus_size: int = None, name: str = "eval", dev_batch_size=32, main_score_function="dot_score") -> SentenceEvaluator:

        if len(queries) <= 0:
            raise ValueError("Dev Set Empty!, Cannot evaluate on Dev set.")
        
        rel_docs = {}
        corpus_ids = set()
        
        # need to convert corpus to cid => doc      
        corpus = {idx: corpus[idx].get("title") + " " + corpus[idx].get("text") for idx in corpus}
        
        # need to convert dev_qrels to qid => Set[cid]        
        for query_id, metadata in qrels.items():
            rel_docs[query_id] = set()
            for corpus_id, score in metadata.items():
                if score >= 1:
                    corpus_ids.add(corpus_id)
                    rel_docs[query_id].add(corpus_id)
        
        if max_corpus_size:
            # check if length of corpus_ids > max_corpus_size
            if len(corpus_ids) > max_corpus_size:
                raise ValueError("Your maximum corpus size should atleast contain {} corpus ids".format(len(corpus_ids)))
            
            # Add mandatory corpus documents
            new_corpus = {idx: corpus[idx] for idx in corpus_ids}
            
            # Remove mandatory corpus documents from original corpus
            for corpus_id in corpus_ids:
                corpus.pop(corpus_id, None)
            
            # Sample randomly remaining corpus documents
            for corpus_id in random.sample(list(corpus), max_corpus_size - len(corpus_ids)):
                new_corpus[corpus_id] = corpus[corpus_id]

            corpus = new_corpus

        logger.info("{} set contains {} documents and {} queries".format(name, len(corpus), len(queries)))
        return InformationRetrievalEvaluator(queries, corpus, rel_docs, name=name, batch_size=dev_batch_size, main_score_function=main_score_function)
```

### File: beir/beir/retrieval/models/sentence_bert.py 

encode_queries
- Change the return to convert the output to cpu. return self.q_model.encode(queries, batch_size=batch_size, **kwargs).cpu()

Updated code with the changes:
```
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs).cpu()
```

encode_corpus
- Change the return to convert the output to cpu. return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs).cpu()

Updated code with the changes:
```
    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dict[str, List]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs).cpu()

```

## Sentence Transformers:

### File: sentence-transformers/sentence_transformers/evaluation/InformationRetrievalEvaluator.py

init of the object
- Added k=100. Precision_recall_at_k: List[int] = [1, 3, 5, 10, 100],

call function
- Select best epoch according to ncg@k

Updated code with the changes:
```
class InformationRetrievalEvaluator(SentenceEvaluator):
    """
    This class evaluates an Information Retrieval (IR) setting.

    Given a set of queries and a large corpus set. It will retrieve for each query the top-k most similar document. It measures
    Mean Reciprocal Rank (MRR), Recall@k, and Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(self,
                 queries: Dict[str, str],  #qid => query
                 corpus: Dict[str, str],  #cid => doc
                 relevant_docs: Dict[str, Set[str]],  #qid => Set[cid]
                 corpus_chunk_size: int = 50000,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [1, 3, 5, 10, 100],
                 precision_recall_at_k: List[int] = [1, 3, 5, 10, 100],
                 map_at_k: List[int] = [100],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = '',
                 write_csv: bool = True,
                 score_functions: List[Callable[[Tensor, Tensor], Tensor] ] = {'cos_sim': cos_sim, 'dot_score': dot_score},       #Score function, higher=more similar
                 main_score_function: str = None
                 ):

        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in accuracy_at_k:
                self.csv_headers.append("{}-Accuracy@{}".format(score_name, k))

            for k in precision_recall_at_k:
                self.csv_headers.append("{}-Precision@{}".format(score_name, k))
                self.csv_headers.append("{}-Recall@{}".format(score_name, k))

            for k in mrr_at_k:
                self.csv_headers.append("{}-MRR@{}".format(score_name, k))

            for k in ndcg_at_k:
                self.csv_headers.append("{}-NDCG@{}".format(score_name, k))

            for k in map_at_k:
                self.csv_headers.append("{}-MAP@{}".format(score_name, k))

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Information Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]['accuracy@k'][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]['precision@k'][k])
                    output_data.append(scores[name]['recall@k'][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]['mrr@k'][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]['ndcg@k'][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]['map@k'][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if self.main_score_function is None:
            return max([scores[name]['ndcg@k'][max(self.ndcg_at_k)] for name in self.score_function_names])
        else:
            return scores[self.main_score_function]['ndcg@k'][max(self.ndcg_at_k)]

```

# Information about the project

- Author: Carlos Domínguez Becerril.
- Supervisors: Eneko Agirre Bengoa and Gorka Azkune Galparsoro
