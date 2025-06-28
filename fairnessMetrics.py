from fairlearn.metrics import demographic_parity_difference, equal_opportunity_difference, selection_rate, MetricFrame
from sklearn.metrics import accuracy_score
import pandas as pd

# 0. Helper to ensure binary labels (0 or 1)
def ensure_binary_labels(y):
    # Convert to int if needed, then threshold by max value to binary 0/1
    y_int = y.astype(int) if hasattr(y, "astype") else pd.Series(y).astype(int)
    max_val = y_int.max()
    return y_int.apply(lambda x: 1 if x == max_val else 0)

# 1.Finding Sensitive Features-
def find_sensitive_features(X):
    known_sensitive_keywords = [
        "age", "sex", "gender", "gender_identity",
        "race", "ethnicity", "ethnic_group",
        "nationality", "citizenship", "country_of_origin",
        "language", "primary_language",
        "disability", "mental_health", "physical_disability",
        "health_status", "chronic_illness", "medical_condition",
        "religion", "faith", "belief_system",
        "marital_status", "civil_status",
        "sexual_orientation", "lgbtq", "orientation",
        "income", "household_income", "socioeconomic_status", "poverty_status",
        "education_level", "literacy", "employment_status",
        "zipcode", "postal_code", "residence_area", "region", "neighborhood",
        "parental_status", "num_children", "family_size"
    ]

    sensitive_features = []

    for col in X.columns:
        col_norm = col.lower().replace(" ", "_")
        for keyword in known_sensitive_keywords:
            keyword_norm = keyword.lower().replace(" ", "_")
            if keyword_norm in col_norm:
                sensitive_features.append(col)
                break

    return sensitive_features


# 2.Metrics-
def calc_statistical_parity(y_true, y_pred, sensitive_features):
    return demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)

def calc_equal_opportunity(y_true, y_pred, sensitive_features):
    return equal_opportunity_difference(y_true, y_pred, sensitive_features=sensitive_features)

def calc_disparate_impact(y_true, y_pred, sensitive_features):
    metric_frame = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    rates = metric_frame.by_group

    if isinstance(rates, pd.Series):
        min_rate = rates.min()
        max_rate = rates.max()
        ratio = min_rate / max_rate if max_rate != 0 else 0
    else:
        ratio = rates
    return ratio

def calc_group_wise_acc(y_true, y_pred, sensitive_features):
    metric_frame = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    return metric_frame.by_group.to_dict()


# 3.Combining Metrics
def eval_fairness_metrics(model, x_test, y_test, sensitive_feature_names=None):
    # If no sensitive feature provided, try auto detect
    if sensitive_feature_names is None:
        sensitive_feature_names = find_sensitive_features(x_test)

    if not sensitive_feature_names:
        raise ValueError("No sensitive features detected. Please provide sensitive features manually.")

    if isinstance(sensitive_feature_names, str):
        sensitive_features = x_test[[sensitive_feature_names]]
    else:
        sensitive_features = x_test[sensitive_feature_names]

    # Get predictions
    y_pred = model.predict(x_test)

    # Ensure binary labels for y_test and y_pred
    y_test_bin = ensure_binary_labels(y_test)
    y_pred_bin = ensure_binary_labels(pd.Series(y_pred))

    metric_result = {
        "Statistical Parity Difference": calc_statistical_parity(y_test_bin, y_pred_bin, sensitive_features),
        "Equal Opportunity Difference": calc_equal_opportunity(y_test_bin, y_pred_bin, sensitive_features),
        "Disparate Impact": calc_disparate_impact(y_test_bin, y_pred_bin, sensitive_features),
        "Group-wise Accuracy": calc_group_wise_acc(y_test_bin, y_pred_bin, sensitive_features)
    }
    return metric_result


# 4.Interpreting Metrics
def interpret_metrics(metrics):
    interpretation = {}

    spd = metrics.get("Statistical Parity Difference")
    interpretation["Statistical Parity Difference"] = (
        "No significant disparity detected across groups."
        if abs(spd) < 0.1 else
        "Warning: Statistical parity difference suggests possible bias."
    )

    eod = metrics.get("Equal Opportunity Difference")
    interpretation["Equal Opportunity Difference"] = (
        "Equal opportunity appears to be maintained among groups."
        if abs(eod) < 0.1 else
        "Warning: Equal opportunity difference indicates potential bias in true positive rates."
    )

    di = metrics.get("Disparate Impact")
    interpretation["Disparate Impact"] = (
        "Disparate impact is within acceptable limits."
        if 0.8 <= di <= 1.25 else
        "Warning: Disparate impact indicates possible unfair treatment of certain groups."
    )

    acc_dict = metrics.get("Group-wise Accuracy", {})
    if isinstance(acc_dict, dict) and len(acc_dict) > 1:
        max_acc = max(acc_dict.values())
        min_acc = min(acc_dict.values())
        gap = max_acc - min_acc
        interpretation["Group-wise Accuracy"] = (
            f"Accuracy gap among groups is {gap:.3f}, which is within acceptable range."
            if gap < 0.1 else
            f"Warning: Accuracy gap among groups is {gap:.3f}, indicating potential bias."
        )
    else:
        interpretation["Group-wise Accuracy"] = "Insufficient group data to evaluate accuracy fairness."

    return interpretation