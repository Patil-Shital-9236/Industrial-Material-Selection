import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Industrial Material Selector", layout="wide")

# ------------------- Load Artifacts -------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_artifacts/similarity_model.joblib")
    preprocessor = joblib.load("model_artifacts/preprocessor.joblib")
    dataset = joblib.load("model_artifacts/full_dataset.joblib")
    return model, preprocessor, dataset

model, preprocessor, dataset = load_artifacts()

st.title("🏭 Industrial Material Selection")
st.markdown(
    "Provide parameters of your desired material and get **similar materials** "
    "from the industry database."
)

# ------------------- Input UI -------------------
st.sidebar.header("Material Parameters")

# Dynamically build inputs based on dataset columns
user_input = {}
for col in dataset.columns:
    if dataset[col].dtype in [np.float64, np.int64]:
        # Numeric input
        val = float(dataset[col].mean()) if pd.notna(dataset[col].mean()) else 0.0
        user_input[col] = st.sidebar.number_input(f"{col}", value=val)
    else:
        # Categorical input
        unique_vals = dataset[col].dropna().unique().tolist()
        default_val = unique_vals[0] if unique_vals else "Unknown"
        user_input[col] = st.sidebar.selectbox(f"{col}", options=unique_vals or ["Unknown"], index=0)

if st.sidebar.button("Find Similar Materials"):
    # Convert to DataFrame
    user_df = pd.DataFrame([user_input])

    # Transform with the same preprocessor
    user_transformed = preprocessor.transform(user_df)

    # Find nearest neighbors
    distances, indices = model.kneighbors(user_transformed, n_neighbors=6)

    st.subheader("🔎 Top Similar Materials")
    result_df = dataset.iloc[indices[0]].copy()
    result_df["Similarity Score"] = 1 - distances[0]  # higher is more similar
    st.dataframe(result_df.reset_index(drop=True))
else:
    st.info("Adjust parameters on the left and click **Find Similar Materials**.")

st.markdown("---")
st.caption("Industrial Material Selection System")
