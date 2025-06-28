import numpy as np
import pandas as pd

def simulate_mia(model, X_train, X_test):
    try:
        if hasattr(model, "predict_proba"):
            train_probs = model.predict_proba(X_train)
            test_probs = model.predict_proba(X_test)
        else:
            # Fallback: simulate hard prediction "confidence"
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            classes = np.unique(train_preds)
            train_probs = np.eye(len(classes))[train_preds]
            test_probs = np.eye(len(classes))[test_preds]

        train_confidences = np.max(train_probs, axis=1)
        test_confidences = np.max(test_probs, axis=1)

        avg_train = round(np.mean(train_confidences), 3)
        avg_test = round(np.mean(test_confidences), 3)
        gap = round(avg_train - avg_test, 3)

        risk_level = "High Risk" if gap > 0.15 else "Low Risk"
        message = (
            f"The model is significantly more confident on training data "
            f"than test data (gap = {gap}). This may indicate a vulnerability to membership inference attacks."
            if gap > 0.15 else
            f"The model shows similar confidence on training and test data (gap = {gap}), indicating low risk."
        )

        return {
            "Risk Level": risk_level,
            "Details": {
                "Average Confidence (Train Set)": avg_train,
                "Average Confidence (Test Set)": avg_test,
                "Confidence Gap": gap
            },
            "Interpretation": message
        }
    except Exception as e:
        return {
            "Risk Level": "Not Evaluated",
            "Interpretation": "Could not run Membership Inference Attack simulation.",
            "Error": str(e)
        }

def check_differential_privacy(model):
    try:
        module_name = type(model).__module__
        is_dp = "diffprivlib" in module_name.lower()

        if is_dp:
            return {
                "Risk Level": "Privacy Preserved",
                "Interpretation": "The model was trained using a differential privacy library (diffprivlib)."
            }
        else:
            return {
                "Risk Level": "Not Detected",
                "Interpretation": "No signs of differential privacy usage were found. Consider using DP methods to protect sensitive data."
            }
    except Exception as e:
        return {
            "Risk Level": "Not Evaluated",
            "Interpretation": "Could not verify differential privacy.",
            "Error": str(e)
        }

def check_data_leakage(X_train, X_test):
    try:
        train_rows = X_train.astype(str).agg("".join, axis=1)
        test_rows = X_test.astype(str).agg("".join, axis=1)

        overlap = len(set(train_rows) & set(test_rows))
        percent = round(100 * overlap / len(X_test), 2)

        risk_level = "High Risk" if percent > 5 else "Low Risk"
        message = (
            f"{overlap} test samples appear to be duplicates from the training data "
            f"({percent}% overlap), which could indicate data leakage."
            if percent > 5 else
            f"No significant overlap found between training and test data ({percent}%), indicating low leakage risk."
        )

        return {
            "Risk Level": risk_level,
            "Details": {
                "Duplicate Rows in Test Data": overlap,
                "Leakage Percentage": f"{percent}%"
            },
            "Interpretation": message
        }
    except Exception as e:
        return {
            "Risk Level": "Not Evaluated",
            "Interpretation": "Could not evaluate data leakage.",
            "Error": str(e)
        }

def evaluate_privacy(model, X_train, X_test, y_train, y_test):
    print("Running Privacy Checks...")

    return {
        "Membership Inference Risk": simulate_mia(model, X_train, X_test),
        "Differential Privacy Check": check_differential_privacy(model),
        "Data Leakage Check": check_data_leakage(X_train, X_test)
    }
