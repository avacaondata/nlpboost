from nlpboost import DatasetConfig, ModelConfig, AutoTrainer, ResultsPlotter
from nlpboost.default_param_spaces import hp_space_base


def pre_parse_func(example):
    label_cols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "L", "M", "N", "Z"]
    new_example = {"text": example["abstractText"]}
    for col in label_cols:
        new_example[f"label_{col}"] = example[col]
    return new_example


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
        "no_cuda": False,
        "load_best_model_at_end": True,
        "per_device_eval_batch_size": 16,
        "max_steps": 1
    }
    default_args_dataset = {
        "seed": 44,
        "direction_optimize": "maximize",
        "metric_optimize": "eval_f1-score",
        "retrain_at_end": False,
        "fixed_training_args": fixed_train_args
    }
    pubmed_config = default_args_dataset.copy()
    pubmed_config.update(
        {
            "dataset_name": "pubmed",
            "alias": "pubmed",
            "task": "classification",
            "is_multilabel": True,
            "multilabel_label_names": [f"label_{col}" for col in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "L", "M", "N", "Z"]],
            "text_field": "text",
            "label_col": "label_A",
            "hf_load_kwargs": {"path": "owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH"},
            "pre_func": pre_parse_func,
            "remove_fields_pre_func": True,
            "config_num_labels": 14,  # for multilabel we need to pass the number of labels for the config.
            "split": True  # as the dataset only comes with train split, we need to split in train, val, test.
        }
    )
    pubmed_config = DatasetConfig(**pubmed_config)
    model_config = ModelConfig(
        name="bertin-project/bertin-roberta-base-spanish",
        save_name="bertin",
        hp_space=hp_space_base,
        n_trials=1
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[pubmed_config],
        metrics_dir="pubmed_metrics"
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
