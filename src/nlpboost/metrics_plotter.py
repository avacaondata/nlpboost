import json
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import numpy as np
from typing import List, Dict
from .utils import joinpaths
import os
import re


class ResultsPlotter:
    """
    Tool for plotting the results of the models trained.

    Parameters
    ----------
    metrics_dir: str
        Directory name with metrics.
    model_names: List
        List with the names of the models.
    dataset_to_task_map: Dict
        Dictionary that maps dataset names to tasks. Can be built with the list of DatasetConfigs.
    remove_strs: List
        List of strings to remove from filename.
    metric_field: str
        Name of the field with the objective metric.
    """

    def __init__(
        self,
        metrics_dir: str,
        model_names: List,
        dataset_to_task_map: Dict,
        remove_strs: List = [],
        metric_field: str = "f1-score",
    ):
        self.metrics_dir = metrics_dir
        self.model_names = model_names
        self.dataset_to_task_map = dataset_to_task_map
        self.remove_strs = remove_strs
        self.metric_field = metric_field

    def plot_metrics(self):
        """Plot the metrics as a barplot."""
        df_metrics = self.read_metrics()
        df_metrics = df_metrics.groupby(
            ["dataset_name", "model_name"], as_index=False
        ).aggregate("max")
        df_metrics = df_metrics.append(
            df_metrics.groupby(by="model_name", as_index=False).aggregate("mean")
        )
        df_metrics.loc[df_metrics["dataset_name"].isna(), "dataset_name"] = "AVERAGE"
        plot = self._make_plot(df_metrics)
        return plot

    def _make_plot(self, df):
        """
        Build the plot with the dataset in the correct format.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the metrics data.

        Returns
        -------
        ax: matplotlib.axes.Axes
            ax object as returned by matplotlib.
        """
        plt.rcParams["figure.figsize"] = (20, 15)
        plt.rcParams["xtick.labelsize"] = "large"
        plt.rcParams["ytick.labelsize"] = "large"

        ax = sns.barplot(
            y="dataset_name",
            x=self.metric_field,
            data=df.sort_values(["model_name", "dataset_name"]),
            hue="model_name",
        )
        ax.set_xticks(np.linspace(0.0, 1.0, 25))
        plt.grid(True, color="#93a1a1", alpha=0.9, linestyle="--", which="both")
        plt.title(
            "Experiments Results",
            size=22,
            fontdict={
                "fontstyle": "normal",
                "fontfamily": "serif",
                "fontweight": "bold",
            },
        )
        plt.ylabel("Dataset Name", size=18, fontdict={"fontfamily": "serif"})
        plt.xlabel(
            f"{self.metric_field}", size=18, fontdict={"fontfamily": "serif"}
        )
        sns.despine()
        plt.legend(bbox_to_anchor=(0.9, 0.98), loc=3, borderaxespad=0.0)
        return ax

    def read_metrics(
        self,
    ):
        """Read the metrics in the self.metrics_dir directory, creating a dataset with the data."""
        dics = []
        files = [joinpaths(self.metrics_dir, f) for f in os.listdir(self.metrics_dir)]
        for file in tqdm(files, desc="reading metrics files..."):
            try:
                if ".json" in file:
                    with open(file, "r") as f:
                        d = json.load(f)
                else:
                    with open(file, "r") as f:
                        d = f.read()
                        d = ast.literal_eval(d)
                file = (
                    file.replace(self.metrics_dir, "")
                    .replace("/", "")
                    .replace("-dropout_0.0.json", "")
                )
                for remove_str in self.remove_strs:

                    file = re.sub(remove_str, "", file)
                dataset_name = (
                    file.replace(".json", "").replace(".txt", "").split("#")[-1]
                )
                model_name = file.replace(".json", "").replace(".txt", "").split("#")[0]
                if dataset_name not in self.dataset_to_task_map:
                    task = "qa"
                else:
                    task = self.dataset_to_task_map[dataset_name]
                if task == "qa" and d["f1"] > 1.0:
                    f1 = d["f1"] * 0.01
                elif task == "multiple_choice":
                    f1 = d["accuracy"]
                else:
                    f1 = d[self.metric_field]
                newdic = {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    f"{self.metric_field}": f1,
                    "namefile": file,
                    "task": task,
                }
                dics.append(newdic)
            except Exception as e:
                print(e)
                continue
        return pd.DataFrame(dics)
