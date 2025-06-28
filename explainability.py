# explainability.py
import shap
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

def explain_with_shap(model, X_train, X_test, max_features=5):
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_test)

    feature_importance = np.abs(shap_values.values).mean(axis=0)
    feature_names = shap_values.feature_names
    sorted_idx = np.argsort(feature_importance)[::-1][:max_features]

    explanation = {
        "method": "SHAP",
        "top_features": [
            {"feature": feature_names[i], "importance": round(feature_importance[i], 4)}
            for i in sorted_idx
        ],
        "shap_values": shap_values
    }
    return explanation


def explain_with_lime(model, X_train, X_test_row, feature_names, class_names=None, num_features=5):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names or ["class 0", "class 1"],
        mode="classification"
    )
    exp = explainer.explain_instance(X_test_row, model.predict_proba, num_features=num_features)

    lime_fig = exp.as_pyplot_figure()  # returns matplotlib Figure

    explanation = {
        "method": "LIME",
        "top_features": [
            {"feature": f, "importance": round(w, 4)}
            for f, w in exp.as_list()
        ],
        "lime_plot": lime_fig
    }
    return explanation
