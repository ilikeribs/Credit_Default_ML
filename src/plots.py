import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
from typing import Union, List, Optional


def boxplots(df: pd.DataFrame, x_column: str, y_columns: List[str]):
    """
    Plot a subplot of boxplots.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        x_column (str): The column name to be used for the x-axis.
        y_columns (List[str]): A list of column names to be used for the y-axis.
    """
    fig, axes = plt.subplots(len(y_columns), 1, figsize=(6, 12))

    for i, y_column in enumerate(y_columns):
        sns.boxplot(ax=axes[i], data=df, x=x_column, y=y_column)
        axes[i].set_title(f"Boxplot for {y_column}")

    plt.tight_layout()
    plt.show()


def feature_balance(
    df: pd.DataFrame, target: str, hue: Optional[str] = None
) -> plt.Axes:
    ax = sns.countplot(data=df, x=target, hue=hue)

    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    for c in ax.containers:
        ax.bar_label(c, fmt=lambda v: f"{(v/len(df))*100:0.1f}%")

    return ax


def large_feature_balance(
    df: pd.DataFrame, y: str, order: Optional[list] = None
) -> plt.Axes:
    """
    Create a Seaborn countplot with percentage labels for each category.

    Parameters:
    - df: DataFrame containing the data.
    - y: Column for the y-axis.
    - order: Optional, list specifying the order of categories.
    - color: Optional, color for the bars.

    Returns:
    - ax: Matplotlib Axes object.
    """

    ax = sns.countplot(data=df, y=y, order=order, color="steelblue")

    ax.set_frame_on(False)
    ax.set_ylabel(None)

    ax.axes.get_xaxis().set_visible(False)

    for c in ax.containers:
        ax.bar_label(c, fmt=lambda v: f"{(v/len(df))*100:0.1f}%")

    return ax


def num_feature_histogram(df: pd.DataFrame, features: list) -> None:
    """
    Plot a 1x3 subplot of Seaborn histplots for a list of features.

    Parameters:
    - df: DataFrame containing the data.
    - features: List of features to be plotted.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.histplot(data=df, x=abs(df[feature]), hue="TARGET", bins=15, ax=ax)
        ax.set_title(f"{feature} Distribution")
        ax.set_xlabel(f"{feature}")
        ax.set_ylabel("Application Count")

        # Customize legend
        legend_labels = ["Paid Off", "Defaulted"]
        ax.legend(title="Loan Status", labels=legend_labels)

    plt.tight_layout()
    plt.show()


def num_feature_combined(df: pd.DataFrame, features: list) -> None:
    """
    Plot a 2x3 subplot of Seaborn histplots and boxplots for a list of features.

    Parameters:
    - df: DataFrame containing the data.
    - features: List of features to be plotted.
    """

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    for i, feature in enumerate(features):
        ax_hist = axes[i, 0]
        sns.histplot(data=df, x=abs(df[feature]), hue="TARGET", bins=30, ax=ax_hist)
        ax_hist.set_title(f"{feature} Distribution")
        ax_hist.set_xlabel(f"{feature}")
        ax_hist.set_ylabel("Application Count")

        ax_box = axes[i, 1]
        sns.boxplot(data=df, x="TARGET", y=df[feature], ax=ax_box)
        ax_box.set_title(f"{feature} Boxplot")
        ax_box.set_xlabel("Loan Status")
        ax_box.set_ylabel(f"{feature}")

        if i == 0:
            ax_hist.legend(title="Loan Status", labels=["Paid Off", "Defaulted"])

    plt.tight_layout()
    plt.show()


def kde_plots(df: pd.DataFrame, features: list) -> None:
    """
    Plot a 1x3 subplot of Seaborn kdeplots for a list of features
    with target feature applied.

    Parameters:
    - df: DataFrame containing the data.
    - features: List of features to be plotted.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.kdeplot(
            data=df[df["TARGET"] == 0][feature],
            ax=ax,
            label="Paid Off",
            color="blue",
            fill=False,
        )
        sns.kdeplot(
            data=df[df["TARGET"] == 1][feature],
            ax=ax,
            label="Defaulted",
            color="orange",
            fill=False,
        )
        ax.set_title(f"{feature} Density Plot")
        ax.set_xlabel(f"{feature}")
        ax.set_ylabel("Density")

        ax.legend(title="Loan Status")

    plt.tight_layout()
    plt.show()


def plot_roc(y_val: pd.Series, y_pred: pd.Series) -> None:
    """
    Plot the ROC curve for a binary classifier model.

    Parameters:
    - y_val (Series): True labels.
    - y_pred (Series): Predicted probabilities for the positive class.

    Returns:
    None
    """
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr, tpr, color="darkorange", label="ROC curve (area = {:.2f})".format(roc_auc)
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


def multi_roc_curves(
    model_names: List[str], fpr_list: List[List[float]], tpr_list: List[List[float]]
) -> None:
    """
    Plot multiple ROC curves on the same plot for binary classifier models.

    Parameters:
    - model_names (List[str]): List of model names for labeling in the legend.
    - fpr_list (List[List[float]]): List of false positive rates for each model.
    - tpr_list (List[List[float]]): List of true positive rates for each model.

    Returns:
    None
    """
    plt.figure(figsize=(8, 8))

    for i in range(len(model_names)):
        roc_auc = auc(fpr_list[i], tpr_list[i])
        plt.plot(fpr_list[i], tpr_list[i], label=f"{model_names[i]}")

    # add diagonal no-skill line
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves")

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.legend(loc="lower right", labels=model_names)

    plt.show()
    
