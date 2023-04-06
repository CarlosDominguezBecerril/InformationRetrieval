# Changes to BEIR and Sentence Transformers

## Sentence Transformers

Updated the ```sentence_transformers/evaluation/InformationRetrievalEvaluator.py``` with the following changes:
- Updated the constructor to add the variable ```main_metric```. This variable is used to do model selection according to ```main_metric```

## Beir

Updated the ```beir/retrieval/train.py``` with the following changes:
- Updated the constructor to add the variable ```is_dpr```. This variable is used to change the input format for Sentence Transformers in functions ```load_train``` and ```load_ir_evaluator```.
- Updated the function ```load_ir_evaluator```. It accepts now the ```dev_batch_size```, ```main_score_function```, and ```main_metric``` as parameters

Updated the ```beir/retrieval/models/sentence_bert.py``` with the following changes:
- Updated the constructor to add the variable ```is_dpr```. This variable is used to change the input format for Sentence Transformers in functions ```encode_corpus_parallel```, ```encode_corpus```, and ```encode_queries```.
- When calling to ```encode_corpus``` and ```encode_queries``` the tensors are move to CPU to avoid OOM isues.

