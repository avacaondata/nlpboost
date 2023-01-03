from nlpboost import AutoTrainer, DatasetConfig, ModelConfig
from transformers import EarlyStoppingCallback
import evaluate


if __name__ == "__main__":

    fixed_train_args = {
        "evaluation_strategy": "epoch",
        "num_train_epochs": 10,
        "do_train": True,
        "do_eval": True,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "max_steps": 1,
        "seed": 69,
        "no_cuda": True,
        "bf16": False,
        "dataloader_num_workers": 8,
        "load_best_model_at_end": True,
        "per_device_eval_batch_size": 48,
        "adam_epsilon": 1e-8,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "group_by_length": True,
        "max_grad_norm": 1.0
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
            "task": "summarization",
            "hf_load_kwargs": {"path": "mlsum", "name": "es"},
            "label_col": "summary",
            "num_proc": 8, "additional_metrics": [evaluate.load("meteor")]}
    )

    mlsum_config = DatasetConfig(**mlsum_config)

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical(
                "learning_rate", 3e-5, 7e-5, log=True
            ),
            "num_train_epochs": trial.suggest_categorical(
                "num_train_epochs", [7]
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [16]),
            "per_device_eval_batch_size": trial.suggest_categorical(
                "per_device_eval_batch_size", [32]),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [2]),
            "warmup_steps": trial.suggest_categorical(
                "warmup_steps", [50, 100, 500, 1000]
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", 0.0, 0.1
            ),
        }

    def preprocess_function(examples, tokenizer, dataset_config):
        model_inputs = tokenizer(
            examples[dataset_config.text_field],
            truncation=True,
            max_length=tokenizer.model_max_length
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[dataset_config.summary_field], max_length=dataset_config.max_length_summary, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    marimari_roberta2roberta_config = ModelConfig(
        name="marimari-r2r",
        save_name="marimari-r2r",
        hp_space=hp_space,
        encoder_name="BSC-TeMU/roberta-base-bne",
        decoder_name="BSC-TeMU/roberta-base-bne",
        num_beams=4,
        n_trials=1,
        random_init_trials=1,
        custom_tokenization_func=preprocess_function,
        only_test=False,
    )
    autotrainer = AutoTrainer(
        model_configs=[marimari_roberta2roberta_config],
        dataset_configs=[mlsum_config],
        metrics_dir="mlsum_marimari"
    )

    autotrainer()
