from __future__ import annotations

import numpy as np
import pandas as pd


def grab_col_names(dataframe: pd.DataFrame, cat_th: int = 10, car_th: int = 20):
    cat_col = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_col = cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    num_col = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_col = [col for col in num_col if col not in num_but_cat]

    return cat_col, num_col, cat_but_car


def missing_value_table(dataframe: pd.DataFrame, na_name: bool = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"]).astype({"n_miss": int})
    if na_name:
        return missing_df, na_columns
    return missing_df


def label_encode_binary(dataframe: pd.DataFrame, col: str) -> pd.DataFrame:
    dataframe[col] = (dataframe[col].astype(str).str.lower().eq("yes")).astype(int)
    return dataframe


def one_hot_encoder(dataframe: pd.DataFrame, categorical_col: list[str], drop_first: bool = False) -> pd.DataFrame:
    return pd.get_dummies(dataframe, columns=categorical_col, drop_first=drop_first)


def outlier_thresholds(dataframe: pd.DataFrame, col_name: str, q1: float = 0.05, q3: float = 0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit


def check_outlier(dataframe: pd.DataFrame, col_name: str) -> bool:
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return bool(dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None))


def replace_with_threshold(dataframe: pd.DataFrame, variable: str, q1: float = 0.05, q3: float = 0.95) -> None:
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

