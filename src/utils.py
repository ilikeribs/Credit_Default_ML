import pandas as pd
import numpy as np
from typing import Union, List, Tuple
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
import shap
from sklearn.base import BaseEstimator


def missing_percentage(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the percentage of missing values for each column in a DataFrame.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - missing_percentages: pandas Series
    """
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count() * 100).sort_values(
        ascending=False
    )
    missing_percentages = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    return missing_percentages


def check_categorical_cardinality(
    df: pd.DataFrame, threshold: int = 10
) -> pd.DataFrame:
    """
    Check the cardinality of categorical features in a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - threshold: int, optional, default: 10
        The threshold to consider a feature as high cardinality.

    Returns:
    - high_cardinality_features: pandas DataFrame
        A DataFrame with columns 'Feature' and 'Cardinality' indicating features with high cardinality.
    """
    categorical_features = df.select_dtypes(include=["object", "category"])

    high_cardinality_features = []

    for feature in categorical_features.columns:
        cardinality = len(categorical_features[feature].unique())
        if cardinality >= threshold:
            high_cardinality_features.append(
                {"Feature": feature, "Cardinality": cardinality}
            )

    if high_cardinality_features:
        return pd.DataFrame(high_cardinality_features)
    else:
        return pd.DataFrame(columns=["Feature", "Cardinality"])


def data_type_count(dataframe: pd.DataFrame) -> None:
    """
    Display data types and feature counts in the given DataFrame.

    Parameters:
    - dataframe: Pandas DataFrame

    Returns:
    - None
    """
    data_types: pd.Series = dataframe.dtypes

    data_types = data_types.apply(
        lambda x: x.name if pd.api.types.is_categorical_dtype(x) else x
    )

    grouped_by_dtype: pd.Series = data_types.groupby(data_types).count()

    print("Data Types and Feature Counts in DataFrame:")
    for dtype, count in grouped_by_dtype.items():
        print(f"{dtype}: {count} features")


def reduce_df_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes dataframe to reduce memory usage by assigning correct datatypes

    Parameters:
    - df: pandas DataFrame

    Returns:
    - df: pandas DataFrame
        A DataFrame with optimized datatypes.
    """

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    return df


def hypothesis_chi2(
    data: pd.DataFrame, categorical_features: List[str], target_feature: str
) -> pd.DataFrame:
    """
    Performs Chi-squared tests on a list of categorical features with respect to a target feature.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the categorical and target features.
    - categorical_features (List[str]): A list of column names representing categorical features.
    - target_feature (str): The column name of the target feature.

    Returns:
    pd.DataFrame: A DataFrame containing the results of Chi-squared tests for each categorical feature.
    The columns include 'Feature' (categorical feature name), 'P-Value', and 'Reject Null Hypothesis'.
    'Reject Null Hypothesis' indicates whether the null hypothesis is rejected at a significance level of 0.05.
    """

    result_df = pd.DataFrame(columns=["Feature", "P-Value", "Null Hypothesis"])

    for feature in categorical_features:
        contingency_table = pd.crosstab(data[feature], data[target_feature])

        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        reject_null_hypothesis = "Reject" if p_value < 0.05 else "Accept"

        result_df = pd.concat(
            [
                result_df,
                pd.DataFrame(
                    {
                        "Feature": [feature],
                        "P-Value": [p_value],
                        "Null Hypothesis": [reject_null_hypothesis],
                    }
                ),
            ],
            ignore_index=True,
        )

    return result_df


def hypothesis_chi2_power(
    data: pd.DataFrame, categorical_features: List[str], target_feature: str
) -> pd.DataFrame:
    """
    Perform chi-squared tests on a list of categorical features with respect to a target feature.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the categorical and target features.
    - categorical_features (List[str]): A list of column names representing categorical features.
    - target_feature (str): The column name of the target feature.

    Returns:
    pd.DataFrame: A DataFrame containing the results of chi-squared tests for each categorical feature.
    The columns include 'Feature' (categorical feature name), 'P-Value', 'Reject Null Hypothesis',
    and 'Cramer's V'.

    """
    result_df = pd.DataFrame(
        columns=["Feature", "P-Value", "Null Hypothesis", "Cramer's V"]
    )

    for feature in categorical_features:
        contingency_table = pd.crosstab(data[feature], data[target_feature])

        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))

        reject_null_hypothesis = "Reject" if p_value < 0.05 else "Accept"

        result_df = pd.concat(
            [
                result_df,
                pd.DataFrame(
                    {
                        "Feature": [feature],
                        "P-Value": [p_value],
                        "Null Hypothesis": [reject_null_hypothesis],
                        "Cramer's V": [cramers_v],
                    }
                ),
            ],
            ignore_index=True,
        )

    result_df = result_df.sort_values(by="Cramer's V", ascending=False).reset_index(
        drop=True
    )

    return result_df


def hypothesis_ttest_power(
    data: pd.DataFrame, numerical_features: List[str], target_feature: str
) -> pd.DataFrame:
    """
    Perform t-tests on a list of numerical continuous features with respect to a target feature.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the numerical and target features.
    - numerical_features (List[str]): A list of column names representing numerical continuous features.
    - target_feature (str): The column name of the target feature.

    Returns:
    pd.DataFrame: A DataFrame containing the results of t-tests for each numerical feature.
    The columns include 'Feature' (numerical feature name), 'P-Value', 'Reject Null Hypothesis',
    and "Cohen's d".

    """
    result_df = pd.DataFrame(
        columns=["Feature", "P-Value", "Null Hypothesis", "Cohen's d"]
    )

    for feature in numerical_features:
        data_subset = data[[feature, target_feature]].dropna()

        group1 = data_subset[data_subset[target_feature] == 0][feature]
        group2 = data_subset[data_subset[target_feature] == 1][feature]

        if len(group1) > 0 and len(group2) > 0:
            _, p_value = ttest_ind(group1, group2, equal_var=False)

            pooled_std = np.sqrt(
                ((len(group1) - 1) * group1.var() + (len(group2) - 1) * group2.var())
                / (len(group1) + len(group2) - 2)
            )
            cohens_d = (group1.mean() - group2.mean()) / pooled_std

        else:
            p_value = float("nan")
            cohens_d = float("nan")

        reject_null_hypothesis = "Reject" if p_value < 0.05 else "Accept"

        result_df = pd.concat(
            [
                result_df,
                pd.DataFrame(
                    {
                        "Feature": [feature],
                        "P-Value": [p_value],
                        "Null Hypothesis": [reject_null_hypothesis],
                        "Cohen's d": [cohens_d],
                    }
                ),
            ],
            ignore_index=True,
        )

    result_df = result_df.sort_values(by="Cohen's d", ascending=False).reset_index(
        drop=True
    )

    return result_df


def shap_force_feats(
    model: BaseEstimator, X_validation: pd.DataFrame, y_validation: pd.Series
) -> Tuple[shap.Explanation, shap.Explanation, int, int]:
    """
    Generate SHAP force plots for two instances:
    1. The correct prediction with the highest probability.
    2. The incorrect prediction with the lowest probability.

    Parameters:
    - model (BaseEstimator): Your trained binary classifier.
    - X_validation (DataFrame): Validation dataset features.
    - y_validation (Series): True labels for the validation dataset.

    Returns:
    Tuple[shap.Explanation, shap.Explanation, int, int]:
        - shap_values_correct: SHAP values for the correct prediction with the highest probability.
        - shap_values_incorrect: SHAP values for the incorrect prediction with the lowest probability.
        - prob_correct (int): Index of the correct prediction with the highest probability.
        - prob_incorrect (int): Index of the incorrect prediction with the lowest probability.
    """
    predictions_proba = model.predict_proba(X_validation)
    predictions = model.predict(X_validation)

    correct_indices = np.where(predictions == y_validation)[0]
    incorrect_indices = np.where(predictions != y_validation)[0]

    # Identify the instance with the highest and lowest prediction probability for predictions
    prob_correct = correct_indices[np.argmax(predictions_proba[correct_indices, 1])]
    prob_incorrect = incorrect_indices[
        np.argmin(predictions_proba[incorrect_indices, 1])
    ]

    # Use SHAP to explain the model for the selected instances
    explainer = shap.Explainer(model)
    shap_values_correct = explainer.shap_values(X_validation.iloc[[prob_correct], :])
    shap_values_incorrect = explainer.shap_values(
        X_validation.iloc[[prob_incorrect], :]
    )

    return shap_values_correct, shap_values_incorrect, prob_correct, prob_incorrect


def train_test_val_split(X: pd.DataFrame, y: pd.Series, train_test_size: float, random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split the input data and labels into training, validation, and test sets, stratification
    and shuffling is True by default

    Parameters:
    - X (pd.DataFrame): Input data.
    - y (pd.Series): Target labels.
    - train_test_size (float): The proportion of the dataset to include in the training set.
    - random_seed (int): Seed for random number generation.

    Returns:
    Tuple of Pandas objects: (X_train, X_test, X_val, y_train, y_test, y_val)
    """
    test_val_size = (100 - train_test_size) / 2
    test_val_size /= 100 

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=train_test_size, random_state=random_seed
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=test_val_size, random_state=random_seed
    )

    return X_train, X_test, X_val, y_train, y_test, y_val