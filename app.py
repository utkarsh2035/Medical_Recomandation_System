from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import re
import os
from sklearn.preprocessing import LabelEncoder
import ast

app = Flask(__name__)

# -------------------------------
# File Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
TRAINING_CSV = os.path.join(DATASETS_DIR, "medicine_training.csv")

SYMPTOMS_CSV = os.path.join(DATASETS_DIR, "symptoms_df.csv")
DESCRIPTION_CSV = os.path.join(DATASETS_DIR, "description.csv")
DIET_CSV = os.path.join(DATASETS_DIR, "diets.csv")
MEDICATION_CSV = os.path.join(DATASETS_DIR, "medications.csv")
PRECAUTION_CSV = os.path.join(DATASETS_DIR, "precautions_df.csv")
WORKOUT_CSV = os.path.join(DATASETS_DIR, "workout_df.csv")

# -------------------------------
# Load Model & Datasets
# -------------------------------
model = joblib.load(MODEL_PATH)
train_df = pd.read_csv(TRAINING_CSV)

label_col = next((c for c in train_df.columns if c.strip().lower() in ("prognosis", "disease", "diagnosis")), train_df.columns[-1])
feature_cols = [c for c in train_df.columns if c != label_col]
symptoms_dict = {col.strip().lower().replace(" ", "_"): idx for idx, col in enumerate(feature_cols)}

le = LabelEncoder()
le.fit(train_df[label_col].astype(str).values)
diseases_dict = {i: cls for i, cls in enumerate(le.classes_)}


def _read_csv_safe(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


description_df = _read_csv_safe(DESCRIPTION_CSV)
diet_df = _read_csv_safe(DIET_CSV)
medication_df = _read_csv_safe(MEDICATION_CSV)
precaution_df = _read_csv_safe(PRECAUTION_CSV)
workout_df = _read_csv_safe(WORKOUT_CSV)

# Normalize disease column in all datasets
for df in [description_df, diet_df, medication_df, precaution_df, workout_df]:
    for col in df.columns:
        if "disease" in col.lower():
            df.rename(columns={col: "Disease"}, inplace=True)
            break
    if "Disease" in df.columns:
        df["Disease"] = df["Disease"].astype(str).str.strip().str.lower()


# -------------------------------
# Utility Functions
# -------------------------------
def normalize_sym(sym: str) -> str:
    s = sym.strip().lower()
    s = re.sub(r'[^a-z0-9\s_]', '', s)
    s = re.sub(r'\s+', '_', s)
    return s


def clean_symptoms_input(user_input: str):
    if not isinstance(user_input, str):
        return []
    parts = re.split(r',|\n', user_input)
    cleaned = [normalize_sym(p) for p in parts if normalize_sym(p) in symptoms_dict]
    return cleaned


def build_input_vector(symptoms_list):
    vec = np.zeros(len(feature_cols), dtype=int)
    for s in symptoms_list:
        idx = symptoms_dict.get(s)
        if idx is not None:
            vec[idx] = 1
    return vec.reshape(1, -1)


def list_formatter(value):
    """Convert array-like or list string to clean numbered list"""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            value = [v.strip() for v in value.split(',') if v.strip()]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [f"{i + 1} : {str(v).strip()}" for i, v in enumerate(value) if str(v).strip()]
    if pd.notna(value):
        return [f"1 : {str(value).strip()}"]
    return ["Information not available."]


def get_disease_details(disease_name):
    disease_name = disease_name.strip().lower()

    # Description
    desc_row = description_df.loc[description_df["Disease"] == disease_name, "Description"]
    desc = desc_row.iloc[0] if not desc_row.empty else "Description not available."

    # Diet
    diet_row = diet_df.loc[diet_df["Disease"] == disease_name, "Diet"]
    diet = list_formatter(diet_row.iloc[0]) if not diet_row.empty else ["Diet not available."]

    # Medications
    meds_row = medication_df.loc[medication_df["Disease"] == disease_name, "Medication"]
    meds = list_formatter(meds_row.iloc[0]) if not meds_row.empty else ["Medications not available."]

    # Precautions
    precautions = []
    row = precaution_df.loc[precaution_df["Disease"] == disease_name]
    if not row.empty:
        for col in [c for c in precaution_df.columns if "Precaution" in c]:
            val = row.iloc[0][col]
            if pd.notna(val) and str(val).strip():
                precautions.append(str(val).strip())
    precautions = [f"{i + 1} : {v}" for i, v in enumerate(precautions)] if precautions else ["Precautions not available."]

    # Workout
    w_row = workout_df.loc[workout_df["Disease"] == disease_name, "workout"]
    workout = list_formatter(w_row.iloc[0]) if not w_row.empty else ["Workout suggestions not available."]

    return desc, diet, meds, precautions, workout


# -------------------------------
# Flask Routes
# -------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("symptoms", "")
    cleaned = clean_symptoms_input(user_input)

    if not cleaned:
        return render_template("index.html",
                               error="Please enter valid symptoms (e.g. 'itching, skin_rash').",
                               user_input=user_input)

    input_vec = build_input_vector(cleaned)
    try:
        pred_numeric = model.predict(input_vec)[0]
    except Exception as e:
        return render_template("index.html", error=f"Model error: {e}", user_input=user_input)

    disease_name = pred_numeric if isinstance(pred_numeric, str) else diseases_dict.get(int(pred_numeric), str(pred_numeric))

    desc, diet, meds, precautions, workout = get_disease_details(disease_name)

    return render_template("index.html",
                           disease=disease_name.title(),
                           description=desc,
                           diet=diet,
                           medicines=meds,
                           precautions=precautions,
                           workout=workout,
                           user_input=user_input)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
