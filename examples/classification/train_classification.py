from nlpboost import DatasetConfig, ModelConfig, AutoTrainer, ResultsPlotter
from nlpboost.default_param_spaces import hp_space_base

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
    tweet_eval_config = default_args_dataset.copy()
    tweet_eval_config.update(
        {
            "dataset_name": "tweeteval",
            "alias": "tweeteval",
            "task": "classification",
            "text_field": "text",
            "label_col": "label",
            "hf_load_kwargs": {"path": "tweet_eval", "name": "emotion"}
        }
    )
    tweet_eval_config = DatasetConfig(**tweet_eval_config)
    model_config = ModelConfig(
        name="bertin-project/bertin-roberta-base-spanish",
        save_name="bertin",
        hp_space=hp_space_base,
        n_trials=1,
        only_test=True
    )
    autotrainer = AutoTrainer(
        model_configs=[model_config],
        dataset_configs=[tweet_eval_config],
        metrics_dir="tweeteval_metrics"
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
