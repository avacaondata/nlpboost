# Example scripts of how to use nlpboost for each task

In the examples folder you will find example scripts showing how to fine-tune models for different tasks. These tasks are divided in directories, as you see. In all scripts it is also shown how to use `ResultsPlotter` to save a metrics comparison figure of the models trained.

* `classification`
    For classification we have 2 examples. [train_classification.py](https://github.com/avacaondata/nlpboost/blob/main/examples/classification/train_classification.py) shows how to train a BERTIN model for multi-class classification (emotion detection, check tweet_eval: emotion dataset for more info.). On the other hand, [train_multilabel.py](https://github.com/avacaondata/nlpboost/blob/main/examples/classification/train_multilabel.py) shows how to train a model on a multilabel task.

* `extractive_qa`
    For extractive QA we have only 1 example, as this type of task is very similar in all cases: [train_sqac.py](https://github.com/avacaondata/nlpboost/blob/main/examples/extractive_qa/train_sqac.py) shows how to train a MarIA-large (Spanish Roberta-large) model on SQAC dataset, with hyperparameter search.

* `NER`
    For NER, there is an example script, showing how to train multiple models on multiple NER datasets with different format, where we need to apply a `pre_func` to one of the datasets. The script is called [train_spanish_ner.py](https://github.com/avacaondata/nlpboost/blob/main/examples/NER/train_spanish_ner.py).

* `seq2seq`
    For this task, check out [train_maria_encoder_decoder_marimari.py](https://github.com/avacaondata/nlpboost/blob/main/examples/seq2seq/train_maria_encoder_decoder_marimari.py), which shows how to train a seq2seq model when no encoder-decoder architecture is readily available for a certain language, in this case Spanish. On the other hand, check out [train_summarization_mlsum.py](https://github.com/avacaondata/nlpboost/blob/main/examples/seq2seq/train_summarization_mlsum.py) to learn how to configure training for two multilingual encoder-decoder models for MLSUM summarization task.

**Important**: For more detailed tutorials in Jupyter-Notebook format, please check [nlpboost notebooks](https://github.com/avacaondata/nlpboost/tree/main/notebooks). These tutorials have explanations on all the configuration, which is helpful for getting to better know the tool. They are intended to provide a deep understanding on the different configurations that are needed for each of the tasks, so that the user can easily adapt the scripts for their own tasks and needs.