import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# 1️⃣ Load dataset
data = pd.read_csv('materials_industrial.csv')

# 2️⃣ Handle missing values
for col in data.columns:
    if data[col].dtype in ['float64', 'int64']:
        data[col] = data[col].fillna(data[col].mean())
    else:
        if not data[col].mode().empty:
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna("Unknown")

# 3️⃣ Identify column types
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

# 4️⃣ Build preprocessing pipeline
transformers = []
if numeric_cols:
    transformers.append(("num", StandardScaler(), numeric_cols))
if categorical_cols:
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

preprocessor = ColumnTransformer(transformers)

# 5️⃣ Fit-transform data
X = preprocessor.fit_transform(data)

# 6️⃣ Train Nearest Neighbors model
nn = NearestNeighbors(n_neighbors=6, metric='cosine')
nn.fit(X)

# 7️⃣ Save artifacts
os.makedirs("model_artifacts", exist_ok=True)
joblib.dump(nn, "model_artifacts/similarity_model.joblib")
joblib.dump(preprocessor, "model_artifacts/preprocessor.joblib")
joblib.dump(data, "model_artifacts/full_dataset.joblib")

print("✅ Training completed. Model artifacts saved in 'model_artifacts/'")
