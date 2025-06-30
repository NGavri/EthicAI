import streamlit as st
import pandas as pd
import joblib
import io
import shap
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import os
import json
from pathlib import Path

from fairnessMetrics import find_sensitive_features, eval_fairness_metrics, interpret_metrics
from privacyEvaluation import evaluate_privacy
from ethicalScore import calculate_ethical_score
from explainability import explain_with_shap, explain_with_lime
from reportGenerator import generate_text_report

st.set_page_config(page_title="EthicAI", page_icon= "logo.png", layout="wide")
st.title("EthicAI")
st.sidebar.title("EthicAI")

COUNTER_PATH = "evaluated_counter.json"

def read_model_count():
    try:
        with open(COUNTER_PATH, "r") as f:
            return json.load(f).get("count", 0)
    except Exception:
        # If file does not exist or corrupted, start from zero
        return 0

def increment_model_count():
    count = read_model_count() + 1
    with open(COUNTER_PATH, "w") as f:
        json.dump({"count": count}, f)

def ensure_binary_labels(y):
    if y.dtype == 'O' or y.dtype.name == 'category':
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        if len(le.classes_) > 2:
            st.error("Fairness metrics currently only support binary classification targets.")
            return None
        return y_enc
    else:
        unique_vals = set(y.unique()) if hasattr(y, 'unique') else set(y)
        if unique_vals <= {0, 1} or unique_vals <= {-1, 1}:
            return y
        else:
            y_bin = y.apply(lambda x: 1 if x == max(unique_vals) else 0) if hasattr(y, 'apply') else [1 if x == max(unique_vals) else 0 for x in y]
            return y_bin

def generate_pdf_report(report_text, shap_plot=None, lime_plot=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cross-platform compatible font path
    font_path = Path("fonts/DejaVuSans.ttf").resolve()

    # Debug: Check font path in both local and deployed environments
    st.write("Current working directory:", os.getcwd())
    st.write("Resolved font path:", font_path)
    st.write("Font file exists:", font_path.exists())

    try:
        pdf.add_font("DejaVu", "", str(font_path), uni=True)
        pdf.set_font("DejaVu", "", 12)
    except Exception as e:
        st.error(f"Failed to load custom font: {e}")
        pdf.set_font("Arial", "", 12)

    pdf.add_page()

    # Text first
    for line in report_text.split('\n'):
        pdf.multi_cell(0, 10, line)

    # SHAP plot
    if shap_plot:
        pdf.add_page()
        pdf.set_font("DejaVu", "", 16)
        pdf.cell(0, 10, "SHAP Summary Plot", ln=True)
        shap_path = "shap_temp.png"
        shap_plot.savefig(shap_path, bbox_inches="tight")
        pdf.image(shap_path, x=10, w=pdf.w - 20)
        os.remove(shap_path)

    # LIME plot
    if lime_plot:
        pdf.add_page()
        pdf.set_font("DejaVu", "", 16)
        pdf.cell(0, 10, "LIME Explanation Plot", ln=True)
        lime_path = "lime_temp.png"
        lime_plot.savefig(lime_path, bbox_inches="tight")
        pdf.image(lime_path, x=10, w=pdf.w - 20)
        os.remove(lime_path)

    pdf_output = pdf.output(dest='S').encode('latin1')
    return io.BytesIO(pdf_output)


def get_report_filename(model_name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    safe_model_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(" ", "_")
    return f"Ethical_Report_{safe_model_name}_{date_str}.pdf"


if 'page' not in st.session_state:
    st.session_state.page = "Home"

if st.sidebar.button("Home"):
    st.session_state.page = "Home"
if st.sidebar.button("Evaluation"):
    st.session_state.page = "Evaluation"
if st.sidebar.button("About"):
    st.session_state.page = "About"
if st.sidebar.button("Feedback"):
    st.session_state.page = "Feedback"

if st.session_state.page == "Home":
    # Display model evaluation count in big green text
    model_count = read_model_count()
    st.markdown(
    f"""
    <h3 style='font-weight:bold;'>
      Models Evaluated So Far: 
      <span style='color:green; font-size:48px; font-weight:bold;'>{model_count}</span>
    </h3>
    """,
    unsafe_allow_html=True
)

    st.subheader("Upload your model and dataset")

    model_file = st.file_uploader("Upload your AI model", type=["pkl", "joblib"])
    dataset_file = st.file_uploader("Upload your dataset", type=["csv", "json"])
    model_name_input = st.text_input("Model Name")

    if model_file and dataset_file:
        try:
            model = joblib.load(model_file)
            df = pd.read_csv(dataset_file) if dataset_file.name.endswith(".csv") else pd.read_json(dataset_file)

            st.write("Dataset preview:")
            st.dataframe(df.head())

            target_column = st.text_input("Target column name", value=df.columns[-1])

            if target_column not in df.columns:
                st.error("Invalid target column.")
            else:
                X = df.drop(columns=[target_column])
                y = df[target_column]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                sensitive_features = find_sensitive_features(X)
                if sensitive_features:
                    st.info(f"Automatically detected sensitive feature(s): {sensitive_features}")
                else:
                    st.warning("No sensitive features automatically detected.")
                    sensitive_features = st.multiselect("Select sensitive feature(s):", options=list(X.columns))

                if not sensitive_features:
                    st.error("Select at least one sensitive feature to continue.")
                else:
                    if st.button("Evaluate"):
                        with st.spinner("Evaluating model..."):
                            y_test_bin = ensure_binary_labels(y_test)
                            if y_test_bin is None:
                                st.stop()

                            y_pred = model.predict(X_test)
                            y_pred_bin = ensure_binary_labels(pd.Series(y_pred))
                            if y_pred_bin is None:
                                st.stop()

                            fairness_results = eval_fairness_metrics(model, X_test, y_test_bin, sensitive_features)
                            fairness_interpret = interpret_metrics(fairness_results)
                            privacy_results = evaluate_privacy(model, X_train, X_test, y_train, y_test)
                            score = calculate_ethical_score(fairness_results, privacy_results)

                            shap_result = explain_with_shap(model, X_train, X_test.head(10))
                            lime_result = explain_with_lime(model, X_train, X_test.iloc[0], feature_names=X_train.columns)

                            # SHAP plot
                            shap_fig, ax = plt.subplots()
                            shap.plots.bar(shap_result["shap_values"], show=False, ax=ax)

                            shap_explanation_text = (
                                "\nSHAP Explanation:\n"
                                "SHAP values indicate the average impact of each feature on the model output globally.\n"
                                "Higher absolute values mean higher importance.\n"
                            )
                            lime_explanation_text = (
                                "\nLIME Explanation:\n"
                                "LIME shows the local contribution of features for the selected instance.\n"
                                "Positive weights push the prediction towards the predicted class.\n"
                            )

                            # Generate report text
                            report = generate_text_report(
                                model_name_input,
                                fairness_results,
                                fairness_interpret,
                                privacy_results,
                                score,
                                shap_explanation=shap_result,
                                lime_explanation=lime_result,
                                shap_explanation_text=shap_explanation_text,
                                lime_explanation_text=lime_explanation_text
                            )

                        st.success("Evaluation Complete!")

                        # Increment model evaluation count after successful eval
                        increment_model_count()

                        # Display report
                        st.code(report)

                        # Graphs after report on website
                        st.markdown("### SHAP Summary Plot")
                        st.pyplot(shap_fig)
                        plt.close(shap_fig)

                        st.markdown("### LIME Explanation")
                        st.pyplot(lime_result["lime_plot"])
                        plt.close(lime_result["lime_plot"])

                        # PDF with plots
                        pdf_buffer = generate_pdf_report(
                            report,
                            shap_plot=shap_fig,
                            lime_plot=lime_result["lime_plot"]
                        )
                        filename = get_report_filename(model_name_input)

                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_buffer,
                            file_name=filename,
                            mime="application/pdf"
                        )
        except Exception as e:
            st.error(f"Something went wrong during evaluation: {e}")

elif st.session_state.page == "Evaluation":
    st.subheader("How the models are evaluated?")
    st.markdown("""
    - **Fairness Metrics**:
        - Statistical Parity Difference
        - Equal Opportunity Difference
        - Disparate Impact
        - Group-wise Accuracy

    - **Privacy Metrics**:
        - Membership Inference Attack Simulation
        - Differential Privacy Check
        - Data Leakage Check

    - **Ethical Score**: Weighted score combining fairness and privacy.

    - **Explainability**:
        - SHAP (Global Feature Importance)
        - LIME (Local Explanation)

    - **Report**: Automatically generated report summarizing all findings and scores. (Still improving the recommendations section. Progress is underway!)
                


    """)
    st.write("")    
    st.subheader("Why should you trust this evaluation?")
    st.markdown("""
    Good question.

    Every metric used here is backed by academic research and widely accepted in fairness and privacy audits. 
    The explainability tools (SHAP and LIME) are industry standards.
        
    That said, this tool is still evolving. It's built with transparency and accountability in mind, and the entire process 
    is kept open so **you can verify, interpret, and question** the results yourself.

    Trust isn't given; it’s built and I’m working on it continuously.
                
    """)

elif st.session_state.page == "About":
    st.subheader("About EthicAI")
    st.markdown("""
    EthicAI is a tool to ethically evaluate AI/ML models for **fairness**, **bias**, **privacy**, and **explainability**.
                
    Whether you're a student, researcher, or developer — EthicAI helps you build **responsible AI**.
                
    This version is just the beginning. I’m actively developing **new features and improvements** based on 
    **user feedback and the latest research** to make ethical AI evaluation more accessible and comprehensive.

    ---
    **Curious about AI beyond the usual?**  
    I also run a blog — [**Synapse and Steel**](https://synapseandsteel.wordpress.com/?_gl=1*1wk53el*_gcl_au*MTUyNDU2NzIzNy4xNzUxMTE3NzEx). It's part journal, part tech talk, and part “oops, I did that.”
    Feel free to explore and maybe chuckle once or twice.

    """)

elif st.session_state.page == "Feedback":
    st.subheader("Feedback")
    st.markdown("Your feedback is valuable. Please share your thoughts or suggestions below.")

    with st.form("feedback_form"):
        feedback_text = st.text_area("Enter your feedback here:")

        st.markdown("### Rate EthicAI")
        rating = st.radio(
            "How would you rate your experience?",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: "⭐" * x,
            horizontal=True
        )

        submit_feedback = st.form_submit_button("Submit Feedback")

    if submit_feedback:
        if not feedback_text.strip():
            st.error("Please enter some feedback before submitting.")
        else:
            try:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("feedback.txt", "a", encoding="utf-8") as f:
                    f.write(f"{timestamp} - Rating: {rating}\n")
                    f.write(f"{timestamp} - Feedback: {feedback_text.strip()}\n")
                    f.write("-" * 40 + "\n")
                st.success("Thank you for your feedback!")
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")
