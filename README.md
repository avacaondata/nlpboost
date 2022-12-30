# NLPBOOST: A library for automatic training and comparison of Transformer models 

This library is useful for training multiple transformer-like models for a bunch of datasets in one go, without writing much code or making much effort (the machine does the effort, not you).

# TODO: CHANGE THIS BELOW.

The system architecture is depicted in the following figure:

![Diagrama benchmarker](./imgs/diagram_benchmarker.png "Diagram for benchmarker")

## INSTALATION

To install nlpboost run: 

```
pip install git+https://github.com/alexvaca0/nlpboost.git
```

Be aware that pytorch must be built on a cuda version that is compatible with the machine's installed cuda version. In case pytorch's default cuda version is not compatible visit https://pytorch.org/get-started/locally/ and install a compatible pytorch version.

## MODULES

The library is composed mainly of 3 important objects: the ModelConfig, DatasetConfig, and BenchMarker. The two first are useful for configuring the experiments in a user-friendly way; both of them are dataclasses. BenchMarker, on the other hand, serves for optimizing the models with the configurations passed to it. It uses Optuna in the background to optimize the models' parameters, which are passed in the ModelConfig.

### MODELCONFIG

The ModelConfig class allows to configure each of the models' configurations. The parameters for the ModelConfig class are the following:

- **name: str**

    Name of the model, either in the HF hub or a path to the local directory where it is stored.

- **save_name: str**

    Alias for the model, used for saving it.

- **hp_space**

    The hyperparameter space for hyperparameter search with optuna. Must be a function receiving a trial and returning a dictionary with the corresponding suggest_categorical and float fields.

- **dropout_vals: List**

    Dropout values to try.

- **custom_config_class: transformers.PretrainedConfig**

    Custom configuration for a model. Useful for training ensembles of transformers.

- **custom_model_class: transformers.PreTrainedModel**

    Custom model. None by default. Only used for ensemble models and other strange creatures of Nature.

- **partial_custom_tok_func_call: Any**

    Partial call for a tokenization function, with all necessary parameters passed to it.

- **encoder_name: str**

    Useful for summarization problems, when we want to create an encoder-decoder and want those models to be different.

- **decoder_name: str**

    Useful for summarization problems, when we want to create an encoder-decoder and want those models to be different.

- **tie_encoder_decoder: bool**

    Useful for summarization problems, when we want to have the weights of the encoder and decoder in an EncoderDecoderModel tied.

- **max_length_summary: int**

    Max length of the summaries. Useful for summarization datasets.

- **min_length_summary : int**

    Min length of the summaries. Useful for summarization datasets.

- **no_repeat_ngram_size: int**

    Number of n-grams to don't repeat when doing summarization.

- **early_stopping_summarization: bool**

    Whether to have early stopping when doing summarization tasks.

- **length_penalty: float**

    Length penalty for summarization tasks.

- **num_beams: int**

    Number of beams in beam search for summarization tasks.

- **dropout_field_name: str**

    Name for the dropout field in the pooler layer.

- **n_trials : int**

    Number of trials (trainings) to carry out with this model.

- **random_init_trials: int**

    Argument for optuna sampler, to control number of initial trials to run randomly.

- **trainer_cls_summarization: Any**

    Class for the trainer. Useful when it is desired to override the default trainer cls for summarization.

- **model_cls_summarization: Any**

    Class for the trainer. Useful when it is desired to override the default trainer cls for summarization.

- **custom_proc_func_tokenization: Any**

    Custom function for tokenizing summarization tasks with a model.

- **only_test: bool**

    Whether to only test, not train (for already trained models).

- **test_batch_size: int**

    Batch size for test; only used when doing only testing.

- **overwrite_training_args: Dict**

    Arguments to overwrite the default arguments for the trainer, for example to change the optimizer for this concrete model.

- **save_dir: str**

    The directory to save the trained model.

- **push_to_hub: bool**

    Whether to push the best model to the hub.

- **additional_params_tokenizer: Dict**

    Additional arguments to pass to the tokenizer.

- **resume_from_checkpoint: bool**

    Whether to resume from checkpoint to continue training.

- **config_problem_type: str**

    The type of the problem, for loss fct.

- **custom_trainer_cls: Any**

    Custom trainer class to override the current one.

- **do_nothing: bool**

    Whether to do nothing or not. If true, will not train nor predict.

- **custom_params_config_model: Dict**

    Dictionary with custom parameters for loading AutoConfig.

- **generation_params: Dict**

    Parameters for generative tasks, for the generate call.

There are some examples in the following lines on how to instantiate a class of this type for different kind of models.

- Example 1: instantiate a roberta large with a given hyperparameter space to save it under the name bsc@roberta-large, in a directory "/prueba/". We are going to run 20 trials, the first 8 of them will be random.

```python
from benchmark import ModelConfig

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 5e-5, log=True
        ),
        "num_train_epochs": trial.suggest_categorical(
            "num_train_epochs", [5, 10, 15, 20]
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8]),
        "per_device_eval_batch_size": trial.suggest_categorical(
            "per_device_eval_batch_size", [16]),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [2, 4, 8, 16]),
        "warmup_ratio": trial.suggest_float(
            "warmup_ratio", 0.1, 0.10, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-2, 0.1, log=True
        ),
        "adam_epsilon": trial.suggest_float(
            "adam_epsilon", 1e-10, 1e-6, log=True
        ),
    }

bsc_large_config = ModelConfig(
        name="PlanTL-GOB-ES/roberta-large-bne",
        save_name="bsc@roberta-large",
        hp_space=hp_space,
        save_dir="/prueba/",
        n_trials=20,
        random_init_trials=8,
        dropout_vals=[0.0],
        only_test=False,
    )
```

On the other hand, if the model we are configuring is aimed at doing a seq2seq task, we could configure it like this:

```python
from transformers import Seq2SeqTrainer, MT5ForConditionalGeneration

def tokenize_dataset(examples, tokenizer, dataset_config):
    inputs = ["question: {} context: {}".format(q, c) for q, c in zip(examples["question"], examples["context"])]
    targets = examples[dataset_config.label_col]
    model_inputs = tokenizer(inputs, max_length=1024 if tokenizer.model_max_length != 512 else 512, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=dataset_config.max_length_summary, padding=True, truncation=True)# , return_tensors="np")

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"] # [:, 1:].tolist()
    return model_inputs

mt5_config = ModelConfig(
         name="google/mt5-base",
         save_name="mt5-base",
         hp_space=hp_space,
         num_beams=4,
         trainer_cls_summarization=Seq2SeqTrainer,
         model_cls_summarization=MT5ForConditionalGeneration,
         custom_proc_func_tokenization=tokenize_dataset,
         only_test=False,
         **{
            "min_length_summary": 64,
            "max_length_summary": 360,
            "random_init_trials": 3,
            "n_trials": 1,
            "save_dir": "/prueba_seq2seq/"
         }
)
```

## DATASETCONFIG

Next we have the DatasetConfig class, aimed at configuring all the specifications of a dataset: the fields where data is located, how to process it, what kind of task it is, etc.

The parameters are the following:


- **dataset_name: str**

    The name of the dataset.

- **alias: str**

    Alias for the dataset, for saving it.

- **task: str**

    The task of the dataset. Currenlty, only classification, ner and qa (question answering) are available.

- **fixed_training_args: Dict**
    
    The training arguments (to use in transformers.TrainingArguments) for every model on this dataset, in dictionary format.

- **is_multilabel: bool**

    Whether it is multilabel classification

- **multilabel_label_names: List**

    Names of the labels for multilabel training.

- **hf_load_kwargs: Dict**
    
    Arguments for loading the dataset from the huggingface datasets' hub. Example: {'path': 'wikiann', 'name': 'es'}.
    If None, it is assumed that all necessary files exist locally and are passed in the files field.

- **type_load: str**

    The type of load to perform in load_dataset; for example, if your data is in csv format (d = load_dataset('csv', ...)), this should be csv.

- **files: Dict**

    Files to load the dataset from, in Huggingface's datasets format. Possible keys are train, validation and test.

- **data_field: str**
    
    Field to load data from in the case of jsons loading in datasets.

- **partial_split: bool**
    
    Wheter a partial split is needed, that is, if you only have train and test sets, this should be True so that a new validation set is created.

- **split: bool**
    
    This should be true when you only have one split, that is, a big train set; this creates new validation and test sets.

- **label_col: str**
    
    Name of the label column.

- **val_size: float**
    
    In case no validation split is provided, the proportion of the training data to leave for validation.

- **test_size: float**
    
    In case no test split is provided, the proportion of the total data to leave for testing.

- **pre_func**
    
    Function to perform previous transformations. For example, if your dataset lacks a field (like xquad with title field for example), you can fix it in a function provided here.

- **squad_v2: bool**
    
    Only useful for question answering. Whether it is squad v2 format or not. Default is false.

- **text_field: str**
    
    The name of the field containing the text. Useful only in case of unique-text-field datasets,like most datasets are. In case of 2-sentences datasets like xnli or paws-x this is not useful. Default is text.

- **is_2sents: bool**
    
    Whether it is a 2 sentence dataset. Useful for processing datasets like xnli or paws-x.

- **sentence1_field: str**
    
    In case this is a 2 sents dataset, the name of the first sentence field.

- **sentence2_field: str**
    
    In case this is a 2 sents dataset, the name of the second sentence field.

- **summary_field: str = field**
    
    The name of the field with summaries (we assume the long texts are in the text_field field). Only useful for summarization tasks. Default is summary.

- **callbacks: List**
    
    Callbacks to use inside transformers.

- **metric_optimize: str**
    
    Name of the metric you want to optimize in the hyperparameter search.

- **direction_optimize : str**
    
    Direction of the optimization problem. Whether you want to maximize or minimize metric_optimize.

- **custom_eval_func: Any**
    
    In case we want a special evaluation function, we can provide it here. It must receive EvalPredictions by trainer, like any compute_metrics function in transformers.

- **seed : int**
    
    Seed for optuna sampler.

- **max_length_summary: int**
    
    Max length of the summaries, for tokenization purposes. It will be changed depending on the ModelConfig.

- **num_proc : int**
    
    Number of processes to preprocess data.

- **loaded_dataset: Any**
    
    In case you want to do weird things like concatenating datasets or things like that, you can do that here, by passing a (non-tokenized) dataset in this field.

- **additional_metrics: List**
    
    List of additional metrics loaded from datasets, to compute over the test part.

- **retrain_at_end: bool**
    
    whether to retrain with the best performing model. In most cases this should be True, except when you're only training 1 model with 1 set of hyperparams.

- **config_num_labels: int**
    
    Number of labels to set for the config, if None it will be computed based on number of labels detected.

- **smoke_test: bool**
    
    Whether to select only top 10 rows of the dataset for smoke testing purposes.

- **augment_data: bool**
    
    Whether to augment_data or not.

- **data_augmentation_steps: List**

    List of data augmentation techniques to use from NLPAugPipeline.

- **pretokenized_dataset: datasets.DatasetDict**

    Pre-tokenized dataset, to avoid tokenizing inside benchmarker, which may cause memory issues with huge datasets.


Here we will see different examples of how to create a DatasetConfig for different tasks. There are certain objects that are used in all the examples:

```python
fixed_train_args = {
        "evaluation_strategy": "steps",
        "num_train_epochs": 10,
        "do_train": True,
        "do_eval": True,
        "logging_strategy": "steps",
        "eval_steps": 1,
        "save_steps": 1,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "seed": 69,
        "fp16": False,
        "no_cuda": True,
        "dataloader_num_workers": 2,
        "load_best_model_at_end": True,
        "per_device_eval_batch_size": 16,
        "adam_epsilon": 1e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "max_steps": 1
    }

```

* Example 1: Create a config for Conll2002 dataset, loading it from the Hub:

```python
from transformers import EarlyStoppingCallback
from benchmark import DatasetConfig


conll2002_config = {
    "seed": 44,
    "direction_optimize": "maximize", # whether to maximize or minimize the metric_optimize.
    "metric_optimize": "eval_f1-score", # metric to optimize; must be returned by compute_metrics_func
    "callbacks": [EarlyStoppingCallback(1, 0.00001)], # callbacks
    "fixed_training_args": fixed_train_args, # fixed train args defined before
    "dataset_name": "conll2002", # the name for the dataset
    "alias": "conll2002", # the alias for our dataset
    "task": "ner", # the type of tasl
    "hf_load_kwargs": {"path": "conll2002", "name": "es"}, # this are the arguments we should pass to datasets.load_dataset
    "label_col": "ner_tags", # in this column we have the tags in list of labels format. 
}

conll2002_config = DatasetConfig(**conll2002_config) # Now we have it ready for training with benchmarker!

```

* Example 2: Create a config for MLSUM dataset (for summarization)

```python
from transformers import EarlyStoppingCallback
from benchmark import DatasetConfig

mlsum_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_rouge2",
        "callbacks": [EarlyStoppingCallback(1, 0.00001)],
        "fixed_training_args": fixed_train_args,
        "dataset_name": "mlsum",
        "alias": "mlsum",
        "retrain_at_end": False,
        "task": "summarization",
        "hf_load_kwargs": {"path": "mlsum", "name": "es"},
        "label_col": "summary",
        "num_proc": 16
    }

mlsum_config = DatasetConfig(**mlsum_config)
```

* Example 3: Create a config for a NER task which is in json format.

```python
from transformers import EarlyStoppingCallback
from benchmark import DatasetConfig, joinpaths

data_dir = "/home/loquesea/livingnerdata/"

livingner1_config = {
    "seed": 44,
    "direction_optimize": "maximize",
    "metric_optimize": "eval_f1-score",
    "callbacks": [EarlyStoppingCallback(1, 0.00001)],
    "fixed_training_args": fixed_train_args,
    "dataset_name": "task1-complete@livingner",
    "alias": "task1-complete@livingner",
    "task": "ner",
    "split": False,
    "label_col": "ner_tags", # in this field of each json dict labels are located.
    "text_field": "token_list", # in this field of each json dict the tokens are located
    "files": {"train": joinpaths(data_dir, "task1_train_complete.json"),
            "validation": joinpaths(data_dir, "task1_val_complete.json"),
            "test": joinpaths(data_dir, "task1_val_complete.json")
}

livingner1_config = DatasetConfig(**livingner1_config)
```

You can refer to the examples folder to see more ways of using DatasetConfig, as well as to understand the functionalities of it that are specific to a certain task.


## ADDITIONAL TOOLS

### NLPAugPipeline

This is a pipeline for data augmentation. With this, you can easily integrate [nlpaug](https://github.com/makcedward/nlpaug/) into your datasets from Huggingface, in an easy way. Below there is an example of how to build a pipeline that will be applied over the dataset with different data augmentation methods.
In the below example, 10% of the examples are augmented with contextual word embeddings in inserting mode (that is, a word from the language model is inserted somewhere in the text); 15% are augmented with the same type of augmenter but substituting the words instead of inserting them. Moreover, we also use a backtranslation augmenter over 20% of the examples, translating them to german and then back to english.
If you want more information on how to use and configure each of these augmenters, just check [this notebook](https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb).

```python
from datasets import load_dataset
from benchmark.augmentation import NLPAugPipeline, NLPAugConfig

dataset = load_dataset("ade_corpus_v2", "Ade_corpus_v2_classification")

dataset = dataset["train"].select(range(100))

steps = [
    NLPAugConfig(name="contextual_w_e", proportion=0.1, aug_kwargs={"model_path": "bert-base-cased", "action": "insert", "device":"cuda"}),
    NLPAugConfig(name="contextual_w_e", proportion=0.15, aug_kwargs={"model_path": "bert-base-cased", "action": "substitute", "device": "cuda"}),
    NLPAugConfig(
        name="backtranslation", proportion=0.2, aug_kwargs={"from_model_name": "facebook/wmt19-en-de", "to_model_name": "facebook/wmt19-de-en"}
    ),
]
aug_pipeline = NLPAugPipeline(steps=steps)
augmented_dataset = dataset.map(aug_pipeline.augment, batched=True)
```

It is already integrated with Benchmarker via the DatasetConfig, as shown below. Note that not all objects are declared, so the below example as it is would throw an error. If you want to try it yourself, please define a hp_space function.

```python
from benchmark import DatasetConfig, ModelConfig, BenchMarker
from benchmark.augmentation import NLPAugConfig

augment_steps = [
    NLPAugConfig(name="contextual_w_e", proportion=0.3, aug_kwargs={"model_path": "bert-base-cased", "action": "insert", "device":"cuda"}),
    NLPAugConfig(name="contextual_w_e", proportion=0.3, aug_kwargs={"model_path": "bert-base-cased", "action": "substitute", "device": "cuda"}),
    NLPAugConfig(
        name="backtranslation", proportion=0.3, aug_kwargs={"from_model_name": "Helsinki-NLP/opus-mt-es-en", "to_model_name": "Helsinki-NLP/opus-mt-en-es", "device": "cuda"}
    ),
]

data_config = DatasetConfig(
    **{
        "hf_load_kwargs": {"path": "ade_corpus_v2", "name": "Ade_corpus_v2_classification"},
        "task": "classification",
        # we would put many other parameters here.
        "augment_data": True,
        "data_augmentation_steps": augment_steps
    }
)

# now we can create a model and train it over this dataset with data augmentation.

model_config = ModelConfig(
    name="bert-base-uncased",
    save_name="bert_prueba",
    hp_space = hp_space, # we would have to define this object before.
    n_trials=10,
    random_init_trials=5
)

benchmarker = BenchMarker(
    model_configs = [model_config],
    dataset_configs = [data_config]
)

benchmarker()
```

In this way, we are using the pipeline to internally augment data before training, therefore we will increment the amount of training data, without modifying the validation and test subsets.