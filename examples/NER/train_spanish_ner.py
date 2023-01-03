from nlpboost import AutoTrainer, DatasetConfig, ModelConfig, dict_to_list, ResultsPlotter
from transformers import EarlyStoppingCallback


if __name__ == "__main__":
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

    default_args_dataset = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "callbacks": [EarlyStoppingCallback(1, 0.00001)],
        "fixed_training_args": fixed_train_args
    }

    conll2002_config = default_args_dataset.copy()
    conll2002_config.update(
        {
            "dataset_name": "conll2002",
            "alias": "conll2002",
            "task": "ner",
            "text_field": "tokens",
            "hf_load_kwargs": {"path": "conll2002", "name": "es"},
            "label_col": "ner_tags",
        }
    )

    conll2002_config = DatasetConfig(**conll2002_config)

    ehealth_config = default_args_dataset.copy()

    ehealth_config.update(
        {
            "dataset_name": "ehealth_kd",
            "alias": "ehealth",
            "task": "ner",
            "text_field": "sentence",
            "hf_load_kwargs": {"path": "ehealth_kd"},
            "label_col": "label_list",
            "pre_func": dict_to_list
        }
    )

    ehealth_config = DatasetConfig(**ehealth_config)

    dataset_configs = [
        conll2002_config,
        ehealth_config
    ]

    # AHORA PREPARAMOS LA CONFIGURACIÓN DE LOS MODELOS, EN ESTE CASO BSC Y BERTIN.

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [1.5e-5, 2e-5, 3e-5, 4e-5]
            ),
            "num_train_epochs": trial.suggest_categorical(
                "num_train_epochs", [1]
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [1]),
            "per_device_eval_batch_size": trial.suggest_categorical(
                "per_device_eval_batch_size", [1]),
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

    bertin_config = ModelConfig(
        name="bertin-project/bertin-roberta-base-spanish",
        save_name="bertin",
        hp_space=hp_space
    )

    model_configs = [bsc_config, bertin_config]

    # Y POR ÚLTIMO VAMOS A INICIALIZAR EL BENCHMARKER CON LA CONFIG DE MODELOS Y DATASETS,
    # Y LO LLAMAMOS PARA LLEVAR A CABO LA BÚSQUEDA DE PARÁMETROS.
    autotrainer = AutoTrainer(
        model_configs=model_configs,
        dataset_configs=dataset_configs,
        metrics_dir="pruebas_ner_bertin_bsc",
    )

    results = autotrainer()
    print(results)

    plotter = ResultsPlotter(
        metrics_dir=autotrainer.metrics_dir,
        model_names=[model_config.save_name for model_config in autotrainer.model_configs],
        dataset_to_task_map={dataset_config.alias: dataset_config.task for dataset_config in autotrainer.dataset_configs},
    )
    ax = plotter.plot_metrics()
    ax.figure.savefig("results.png")
