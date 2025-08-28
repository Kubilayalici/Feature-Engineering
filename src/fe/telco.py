from __future__ import annotations

import pandas as pd
from .preprocess import label_encode_binary, one_hot_encoder, grab_col_names


def add_telco_features(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()
    # Clean types
    if "TotalCharges" in dff.columns:
        dff["TotalCharges"] = pd.to_numeric(dff["TotalCharges"], errors="coerce")

    # Target to binary 1/0 for churn
    if "Churn" in dff.columns:
        dff["Churn"] = dff["Churn"].apply(lambda x: 1 if str(x) == "Yes" else 0)

    # Tenure bins
    if "tenure" in dff.columns:
        dff.loc[(dff["tenure"] >= 0) & (dff["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
        dff.loc[(dff["tenure"] > 12) & (dff["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
        dff.loc[(dff["tenure"] > 24) & (dff["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
        dff.loc[(dff["tenure"] > 36) & (dff["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
        dff.loc[(dff["tenure"] > 48) & (dff["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
        dff.loc[(dff["tenure"] > 60) & (dff["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

    # Engagement flags
    if "Contract" in dff.columns:
        dff["NEW_Engaged"] = dff["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

    if {"OnlineBackup", "DeviceProtection", "TechSupport"}.issubset(dff.columns):
        dff["New_noProt"] = dff.apply(
            lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0,
            axis=1,
        )

    if {"SeniorCitizen"}.issubset(dff.columns) and "NEW_Engaged" in dff.columns:
        dff["New_Young_Not_Engaged"] = dff.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

    streaming_cols = ["StreamingTV", "StreamingMovies"]
    svc_cols = [
        "PhoneService",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    if set(svc_cols).issubset(dff.columns):
        dff["NEW_TotalServices"] = (dff[svc_cols] == "Yes").sum(axis=1)
    if set(streaming_cols).issubset(dff.columns):
        dff["NEW_FLAG_ANY_STREAMING"] = dff.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

    if "PaymentMethod" in dff.columns:
        dff["NEW_FLAG_AutoPayment"] = dff["PaymentMethod"].apply(
            lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0
        )

    if {"TotalCharges", "tenure"}.issubset(dff.columns):
        dff["NEW_AVG_Charges"] = dff["TotalCharges"] / (dff["tenure"] + 1)
    if {"NEW_AVG_Charges", "MonthlyCharges"}.issubset(dff.columns):
        dff["NEW_Inrease"] = dff["NEW_AVG_Charges"] / dff["MonthlyCharges"]
    if {"MonthlyCharges"}.issubset(dff.columns) and "NEW_TotalServices" in dff.columns:
        dff["NEW_AVG_Service_fee"] = dff["MonthlyCharges"] / (dff["NEW_TotalServices"] + 1)

    return dff


def telco_basic_pipeline(df: pd.DataFrame, drop_cols: list[str] | None = None, drop_first: bool = True):
    dff = add_telco_features(df)
    # Missing TotalCharges -> fill with median if exists
    if "TotalCharges" in dff.columns:
        dff["TotalCharges"].fillna(dff["TotalCharges"].median(), inplace=True)

    # Prepare encoding
    cat_cols, num_cols, _ = grab_col_names(dff)
    # Label-encode binary categoricals
    binary_cols = [col for col in dff.columns if dff[col].dtypes == "O" and dff[col].nunique() == 2]
    for col in binary_cols:
        dff = label_encode_binary(dff, col)
    # One-hot the rest (excluding target-like and derived numeric counts)
    cat_cols = [c for c in cat_cols if c not in binary_cols and c not in ["Churn", "NEW_TotalServices"]]
    dff = one_hot_encoder(dff, cat_cols, drop_first=drop_first)

    if drop_cols:
        dff = dff.drop(columns=[c for c in drop_cols if c in dff.columns])
    return dff

