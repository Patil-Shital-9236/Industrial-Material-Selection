# Industrial Material Selection — Project Files

Files created:
- preprocess_and_train.py : Preprocessing and training script (unsupervised similarity model)
- streamlit_app.py : Streamlit interface to pick a material or enter requirements and get recommendations
- model_artifacts/similarity_model.joblib : Saved scaler, NN model, feature columns, and original dataframe (created already)

How to run:
1. Ensure `materials_industrial.csv` is in the same folder.
2. Install requirements: `pip install scikit-learn pandas streamlit joblib`
3. (Optional) Recreate artifacts: `python preprocess_and_train.py`
4. Run the UI: `streamlit run streamlit_app.py`

Notes:
- Column names are preserved exactly as in the CSV. The preprocessing creates internal columns with suffixes `_num` or `_ord` but original column names remain untouched in the CSV and in the saved original_df.
- Dataset is small (25 rows); recommendations are based on available features and simple numeric extraction. For production, expand dataset and refine parsing/units.
