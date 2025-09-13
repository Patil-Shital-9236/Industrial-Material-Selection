
# Industrial Material Selection System

An interactive **Machine Learning application** that helps engineers and manufacturers select
**industrial materials** based on multiple parameters.  
The system recommends **similar materials** from a dataset using a **Nearest Neighbors** model.

---

##  Features
- **Material Recommendation**: Enter key parameters and instantly get similar industrial materials.
-  **Mixed Data Handling**: Works with both numeric and categorical features.
- **Machine Learning Model**:
  - Preprocessing with `StandardScaler` + `OneHotEncoder`.
  - Similarity search using `sklearn.neighbors.NearestNeighbors`.
- **Streamlit Interface**: Clean, responsive, and easy-to-use web application.

---

## Project Structure
```

├─ materials\_industrial.csv        # Dataset of industrial materials and parameters
├─ preprocess\_and\_train.py          # Preprocessing and model training script
├─ streamlit\_app.py                  # Streamlit user interface
├─ model\_artifacts/                  # Saved model + preprocessor
└─ requirements.txt                   # Python dependencies

````

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/industrial-material-selection.git
cd industrial-material-selection
````

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python preprocess_and_train.py
```

### 5. Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

---

## How It Works

1. **Data Preprocessing**

   * Missing values are handled automatically.
   * Numeric features are scaled, categorical features are one-hot encoded.
2. **Model Training**

   * Uses `NearestNeighbors` (cosine metric) to compute material similarity.
3. **Material Selection**

   * User inputs desired material parameters via the Streamlit UI.
   * The model returns the top 5 similar materials along with similarity scores.

---

##  Tech Stack

* **Python 3.10+**
* [Streamlit](https://streamlit.io/) – Web interface
* [scikit-learn](https://scikit-learn.org/) – Machine learning
* pandas, numpy – Data handling

---

## Future Enhancements

*  Integration with larger industrial material databases.
* Advanced similarity metrics (e.g., weighted scoring).
* Deployment to cloud platforms like AWS / Heroku.

---

## Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you would like to change.

---

##  License

This project is licensed under the MIT License.

---
