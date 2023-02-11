# InformationRetrieval



The repository pertains to the field of Information Retrieval and focuses on the implementation of an unsupervised question generation method for the purpose of information retrieval. It encompasses the following processes:

- Generation of questions through an unsupervised approach.
- Training of a dense passage retrieval system (DPR) using the generated questions as a basis. (Using haystack https://github.com/deepset-ai/haystack)
- Training of the contriever system using the generated questions. (https://github.com/facebookresearch/contriever).
- Evaluation of the system using the benchmarked evaluation method BEIR.


Further information about the project can be found at: https://drive.google.com/file/d/1JGQH_SvFMhNlcYqrM3beMTCaWJznO7j8/view?usp=sharing

# How to use the code

Note: The code makes use of Slurm to simplify the workflow of the project.

## Question Generation

You can generate the dataset by executing the following command inside the 'dataset' folder:
```
python generate_dataset.py
```

Important parameters in the script include:

- dataset name: name of the dataset to generate questions for
- gpus: specify the number of GPUs to use
- prompt name: "prompt" used to generate the questions. (See prompts.json for the different available options. If the prompt is "is_external," it uses the OpenAI system.)
- model name: name of the model to use (HuggingFace)
 -shards: number of shards to create when generating the dataset (this is used to parallelize the generation process across different GPUs)
- mode: 'manual' or 'auto,' if manual it uses the 'shards' variable, otherwise it creates shards based on the dataset size.

To get started, download the datasets. The easiest way is to use the BEIR GitHub repository (https://github.com/beir-cellar/beir). Then, include the datasets as follows:

```

\ Dataset
    \ original_datasets
        \ arguana
            \ qrels
                - test.tsv
            - corpus.jsonl
            - queries.jsonl
        \ climate-fever
            ...
        \ msmarco
            ...
```

To generate the dataset, you first need to set up your dataset in the DPR format (https://github.com/facebookresearch/DPR). The easiest way is to create a script in the "convert_to_dpr_format" folder. (We provide two scripts, for NFCorpus and Scifact datasets, as examples.)

Three datasets will be generated: the original dataset in the DPR format (original dataset modified), a dataset generated using the contriever generation method (in the DPR format), and a dataset generated using an LLM in an unsupervised way (in the DPR format).

If the system stops during the generation of the dataset using the LLM, you can run the script again to recover from where it was left.

After generating the three datasets, run the script again to join the one generated using the LLM (remember they are in shards).

### Other

If the "ratio" is greater than 0, the training dataset will be split into two (train and dev).

If "generative_negatives" is set, hard negatives will be added using BM25. (Avoid running the code twice to avoid creating two hard negatives.)

# Training

Once the dataset is created, training the system is easy.

## DPR

Inside the **DPR** folder execute:

```
python train.py
```

Make sure that the parameters are set correctly according to what you generated during the dataset generation, and update other parameters such as batch size.

You can also evaluate on BEIR by updating the 'beir_datasets' variable.

## Contriever

Inside the **Contriever** folder execute:

```
python train.py
```

Make sure that the parameters are set correctly according to what you generated during the dataset generation, and update other parameters such as batch size.

You can also evaluate on BEIR by updating the 'beir_datasets' variable.

# Evaluation (BEIR)

Ideally, you should set up the datasets you want to evaluate while training. If you forget to add a dataset, you can evaluate it by running the following command in the BEIR folder:


```
python eval.py
```

In the eval.py file, comment out everything inside ```if __name__ == "__main__":``` and uncomment the last comment. Then, update the parameters to match your model.

# Results

BEIR benchmark rasults.

|   | DPR supervised (ms marco)| GenQ (Supervised) | BM25 (Unsupervised) | LaPraDoR (Unsupervised)  | Contriever (Unsupervised) | ours (Unsupervised)  |
|---|--------------------------|------|------|----------|-----------|---------------|
| Average | 37.07 | **42.49** | 43.01 | 30.21 | 37.06 | 33.31 |
| MS-MARCO | 27.75 | **40.80** | 22.80 | - | 20.60 | **22.88** |
| TREC-covid | 58.90 | **61.90** | 65.60 | 22.70 | 27.40 | 37.50 |
| NFCorpus | 26.98 | **31.90** | 32.50 | 31.10 | 31.70 | 29.25 |
| NaturalQuestions | 31.93 | **35.80** | 32.90 | 18.10 | 25.40 | 24.63 |
| HotpotQA | 40.45 | **53.40** | 60.30 | 30.30 | 48.10 | 38.52 |
| FiQA | 22.61 | **30.80** | 23.60 | 20.30 | 24.50 | 22.21 |
| ArguAna | 45.25 | **49.30** | 31.50 | 45.90 | 37.90 | 47.32 |
| Tóuche-2020 | **19.17** | 18.20 | 36.70 | 9.40 | 19.30 | 14.39 |
| CQAdupstack | 27.46 | **34.70** | 29.90 | 22.00 | 28.40 | 24.64 |
| Quora | 77.59 | **83.00** | 78.90 | 78.70 | 83.50 | 75.27 |
| DBpedia | 29.77 | **32.80** | 31.30 | 25.00 | 29.20 | 25.57 |
| Scidocs | 12.87 | **14.30** | 15.80 | 13.30 | 14.90 | 11.21 |
| Fever | 56.25 | **66.90** | 75.30 | 36.80 | 68.20 | 50.46 |
| Climate-fever | 16.89 | **17.50** | 21.30 | 13.80 | 15.50 | 13.01 |
| Scifact | 52.92 | **64.40** | 66.50 | 55.50 | 64.90 | 52.35 |


Notes:

- Unfortunately, due to Contriever being trained on a different dataset, the results are not fully comparable. Although our system performs worse, the difference in performance is not significant as Contriever was trained on 1,024,000,000 examples, while our system was trained on only 32,000,000 examples.
- GenQ trains a T5 (base) model using MS MARCO to generate questions. Then, for each dataset, it generates five questions using the recently trained model and finally trains a TAS-B bi-encoder model for each dataset (domain adaptation).

# Pretrained models

Pretrained models and datasets are available under request.

# Information about the project

- Type of project: End of master's project.
- Author: Carlos Domínguez Becerril.
- Supervisors: Eneko Agirre Bengoa and Gorka Azkune Galparsoro
