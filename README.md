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

CURRENTLY CLEANING THE CODE TO MAKE IT EASIER TO USE. IT MIGHT NOT WORK THE FOLLOWING.

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
