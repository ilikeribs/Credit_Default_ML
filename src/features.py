from typing import List
import pandas as pd
from lightgbm import LGBMClassifier


def features_above_threshold(
    X_train: pd.DataFrame, y_train: pd.Series, threshold: int = 1
) -> List[str]:
    """
    Get the top features with importance scores above a specified threshold using LightGBM.

    Parameters:
    - X_train (pd.DataFrame): The training data.
    - y_train (pd.Series): The target variable.
    - threshold (int, optional): The importance threshold. Features with importance scores equal to or
      above this threshold will be included in the result. Default is 1.

    Returns:
    - List[str]: List of top features sorted by importance
    """
    X_train[X_train.select_dtypes(["object"]).columns] = X_train.select_dtypes(
        ["object"]
    ).astype("category")

    model = LGBMClassifier(verbose=-1, random_state=42).fit(X_train, y_train)

    feature_importance = model.feature_importances_
    feature_names = model.feature_name_

    feature_importance_tuples = list(zip(feature_names, feature_importance))
    feature_importance_tuples.sort(key=lambda x: x[1], reverse=True)
    top_features = [
        feature[0] for feature in feature_importance_tuples if feature[1] >= threshold
    ]

    return top_features


def add_scores(
    scores_df: pd.DataFrame,
    model_name: str,
    validation_score: float,
    test_score: float,
    num_features: int,
) -> pd.DataFrame:
    """
    Add model scores to the DataFrame.

    Parameters:
    - scores_df (pd.DataFrame): The existing DataFrame containing model scores.
    - model_name (str): The name of the model.
    - validation_score (float): The ROC AUC score on the validation set.
    - test_score (float): The ROC AUC score on the test set.
    - num_features (int): The number of features used in the model.

    Returns:
    pd.DataFrame: Updated DataFrame with the new model scores.

    """
    new_row = pd.DataFrame(
        {
            "Model": [model_name],
            "Validation Score": [validation_score],
            "Test Score": [test_score],
            "Number Of Features": [num_features],
        }
    )

    if scores_df.empty:
        scores_df = new_row
    else:
        scores_df = pd.concat([scores_df, new_row], ignore_index=True)

    return scores_df
