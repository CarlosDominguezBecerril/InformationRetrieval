# InformationRetrieval

The repository pertains to the field of Information Retrieval and focuses on the implementation of an unsupervised question generation method for the purpose of information retrieval. It encompasses the following processes:

- Generation of questions through an unsupervised approach.
- Training of a passage retrieval system using the generated questions as a basis. (Using beir https://github.com/beir-cellar/beir)
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

**Note**: Before training the model please update the BEIR and Sentence Transformers with the files inside ```libraries_to_update``` folder. More explanations about it can be found in the README inside.

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

# Pretrained models and datasets

Pretrained models and datasets are available under request.

# Information about the project

- Author: Carlos Domínguez Becerril.
- Supervisors: Eneko Agirre Bengoa, Jon Ander Campos Gorka Azkune Galparsoro
