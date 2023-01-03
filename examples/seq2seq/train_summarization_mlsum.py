from nlpboost import AutoTrainer, DatasetConfig, ModelConfig, ResultsPlotter
from transformers import EarlyStoppingCallback
from transformers import Seq2SeqTrainer, MT5ForConditionalGeneration, XLMProphetNetForConditionalGeneration

if __name__ == "__main__":

    fixed_train_args = {
        "evaluation_strategy": "epoch",
        "num_train_epochs": 10,
        "do_train": True,
        "do_eval": True,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "seed": 69,
        "bf16": True,
        "dataloader_num_workers": 16,
        "load_best_model_at_end": True,
        "adafactor": True,
    }

    mlsum_config = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_rouge2",
        "callbacks": [EarlyStoppingCallback(1, 0.00001)],
        "fixed_training_args": fixed_train_args
    }

    mlsum_config.update(
        {
            "dataset_name": "mlsum",
            "alias": "mlsum",
            "retrain_at_end": False,
            "task": "summarization",
            "hf_load_kwargs": {"path": "mlsum", "name": "es"},
            "label_col": "summary",
            "num_proc": 16}
    )

    mlsum_config = DatasetConfig(**mlsum_config)

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [3e-5, 5e-5, 7e-5, 2e-4]
            ),
            "num_train_epochs": trial.suggest_categorical(
                "num_train_epochs", [10]
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [8]),
            "per_device_eval_batch_size": trial.suggest_categorical(
                "per_device_eval_batch_size", [8]),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [8]),
            "warmup_ratio": trial.suggest_categorical(
                "warmup_ratio", [0.08]
            ),
        }

    def preprocess_function(examples, tokenizer, dataset_config):
        model_inputs = tokenizer(
            examples[dataset_config.text_field],
            truncation=True,
            max_length=1024
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[dataset_config.summary_field], max_length=dataset_config.max_length_summary, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    mt5_config = ModelConfig(
        name="google/mt5-large",
        save_name="mt5-large",
        hp_space=hp_space,
        num_beams=4,
        trainer_cls_summarization=Seq2SeqTrainer,
        model_cls_summarization=MT5ForConditionalGeneration,
        custom_tokenization_func=preprocess_function,
        n_trials=1,
        random_init_trials=1
    )
    xprophetnet_config = ModelConfig(
        name="microsoft/xprophetnet-large-wiki100-cased",
        save_name="xprophetnet",
        hp_space=hp_space,
        num_beams=4,
        trainer_cls_summarization=Seq2SeqTrainer,
        model_cls_summarization=XLMProphetNetForConditionalGeneration,
        custom_tokenization_func=preprocess_function,
        n_trials=1,
        random_init_trials=1
    )
    autotrainer = AutoTrainer(
        model_configs=[mt5_config, xprophetnet_config],
        dataset_configs=[mlsum_config],
        metrics_dir="mlsum_multilingual_models",
        metrics_cleaner="metrics_mlsum"
    )

    results = autotrainer()
    print(results)

    plotter = ResultsPlotter(
        metrics_dir=autotrainer.metrics_dir,
        model_names=[model_config.save_name for model_config in autotrainer.model_configs],
        dataset_to_task_map={dataset_config.alias: dataset_config.task for dataset_config in autotrainer.dataset_configs},
        metric_field="rouge2"
    )
    ax = plotter.plot_metrics()
    ax.figure.savefig("results.png")