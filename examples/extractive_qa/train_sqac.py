from nlpboost import AutoTrainer, ModelConfig, DatasetConfig, ResultsPlotter
from transformers import EarlyStoppingCallback

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
        "dataloader_num_workers": 8,
        "load_best_model_at_end": True,
        "per_device_eval_batch_size": 16,
        "adam_epsilon": 1e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
    }

    default_args_dataset = {
        "seed": 44,
        "direction_optimize": "minimize",
        "metric_optimize": "eval_loss",
        "callbacks": [EarlyStoppingCallback(1, 0.00001)],
        "fixed_training_args": fixed_train_args
    }

    sqac_config = default_args_dataset.copy()
    sqac_config.update(
        {
            "dataset_name": "sqac",
            "alias": "sqac",
            "task": "qa",
            "text_field": "context",
            "hf_load_kwargs": {"path": "PlanTL-GOB-ES/SQAC"},
            "label_col": "question",
        }
    )
    sqac_config = DatasetConfig(**sqac_config)

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [1.5e-5, 2e-5, 3e-5, 4e-5]
            ),
            "num_train_epochs": trial.suggest_categorical(
                "num_train_epochs", [1]
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [16]),
            "per_device_eval_batch_size": trial.suggest_categorical(
                "per_device_eval_batch_size", [32]),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [1]),
            "warmup_steps": trial.suggest_categorical(
                "warmup_steps", [50, 100, 500, 1000]
            ),
            "weight_decay": trial.suggest_categorical(
                "weight_decay", [0.0]
            )
        }

    bsc_config = ModelConfig(
        name="PlanTL-GOB-ES/roberta-base-bne",
        save_name="bsc@roberta",
        hp_space=hp_space,
    )
    autotrainer = AutoTrainer(
        model_configs=[bsc_config],
        dataset_configs=[sqac_config],
        metrics_dir="spanish_qa_metrics",
        metrics_cleaner="spanish_qa_cleaner_metrics"
    )

    experiment_results = autotrainer()
    print(experiment_results)

    plotter = ResultsPlotter(
        metrics_dir=autotrainer.metrics_dir,
        model_names=[model_config.save_name for model_config in autotrainer.model_configs],
        dataset_to_task_map={dataset_config.alias: dataset_config.task for dataset_config in autotrainer.dataset_configs},
    )
    ax = plotter.plot_metrics()
    ax.figure.savefig("results.png")
