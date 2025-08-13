# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).parent

@st.cache_data
def load_dataset(path=ROOT / "data" / "dataset.csv"):
    return pd.read_csv(path)

@st.cache_resource
def load_artifact(path=ROOT / "model.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Could not load model artifact: {e}")
        return None

def main():
    st.set_page_config(page_title="Diabetes Prediction", layout="wide")
    st.title("Diabetes Prediction (Pima Indians dataset)")
    st.write("A simple Streamlit app that loads a trained pipeline and predicts diabetes outcome.")

    artifact = load_artifact()
    df = load_dataset()

    menu = st.sidebar.selectbox("Page", ["Home / Data", "Visualize", "Predict", "Model Performance", "About"])

    if menu == "Home / Data":
        st.header("Dataset")
        st.write("Shape:", df.shape)
        st.dataframe(df.head(100))
        st.markdown("**Columns:** " + ", ".join(df.columns.tolist()))

    elif menu == "Visualize":
        st.header("Visualizations")
        numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
        col = st.selectbox("Histogram column", numeric_cols, index=numeric_cols.index("Glucose") if "Glucose" in numeric_cols else 0)
        fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
        st.plotly_chart(fig, use_container_width=True)

        if len(numeric_cols) >= 2:
            xcol = st.selectbox("X", numeric_cols, index=0)
            ycol = st.selectbox("Y", numeric_cols, index=1)
            fig2 = px.scatter(df, x=xcol, y=ycol, color="Outcome")
            st.plotly_chart(fig2, use_container_width=True)

    elif menu == "Predict":
        st.header("Make a prediction")
        if artifact is None:
            st.warning("Model artifact not found — run the training notebook and ensure model.pkl exists at project root.")
            return

        feat_names = artifact['feature_names']
        defaults = artifact.get('default_input', {})
        st.write("Enter values or use the defaults (medians from training set).")

        # Create input form
        input_data = {}
        cols = st.columns(4)
        for i, f in enumerate(feat_names):
            default_val = float(defaults.get(f, 0.0))
            # place inputs in columns to save space
            with cols[i % 4]:
                input_data[f] = st.number_input(f, value=default_val, format="%.4f")

        if st.button("Predict"):
            inp_df = pd.DataFrame([input_data], columns=feat_names)
            try:
                pipeline = artifact['pipeline']
                pred = pipeline.predict(inp_df)[0]
                st.markdown(f"### Prediction: **{'Diabetic (1)' if pred==1 else 'Non-diabetic (0)'}**")
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(inp_df)[0]
                    st.write("Probability (class 0, class 1):", [float(p) for p in proba])
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif menu == "Model Performance":
        st.header("Model Performance (saved metrics)")
        if artifact is None:
            st.warning("No artifact loaded.")
        else:
            metrics = artifact.get('metrics', {})
            st.write(metrics)
            # Optional: show confusion matrix by re-evaluating on a test split if you saved X_test,y_test (we didn't)
            # If you want to show confusion matrix from saved metrics, include it in the artifact during training.

    elif menu == "About":
        st.header("About")
        st.markdown("""
        - Dataset: Pima Indians Diabetes Dataset.
        - Target column: Outcome (0 = non-diabetic, 1 = diabetic).
        - Model: RandomForest (best from GridSearch) saved as a scikit-learn pipeline in model.pkl.
        - To retrain: open `notebooks/model_training.ipynb`, run all cells, and re-run the app.
        """)

if __name__ == "__main__":
    main()
