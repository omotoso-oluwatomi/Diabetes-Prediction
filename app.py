import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==============================
# FILE PATHS
# ==============================
MODEL_PATH = "diabetes_model.pkl"
DATA_PATH = "diabetes.csv"
ACCESS_KEY = "medpass123"  # simple password for staff

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction sSystem")

# ==============================
# HELPER FUNCTIONS
# ==============================


def train_and_save_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    """Train the model on updated dataset and save it"""
    df = pd.read_csv(data_path)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    elif os.path.exists(DATA_PATH):
        st.warning("⚠️ Model not found, training new one...")
        return train_and_save_model()
    else:
        st.error("❌ No dataset found to train model.")
        return None


# Load model
model = load_model()

# ==============================
# SIDEBAR NAVIGATION
# ==============================
# page = st.sidebar.radio("Choose Page", ["🔍 Predict Diabetes", "📋 Add Medical Data"])
page = st.sidebar.radio("choose Page", ["Predict Diabetes"])

# ==============================
# PREDICTION PAGE
# ==============================
if page == "Predict Diabetes":
    st.header("Diabetes Risk Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    preg = 0 if gender.lower() == "male" else st.number_input("Pregnancies", 0, 20, 0)
    gluc = st.number_input("Glucose Level", 0, 300, 100)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 0, 120, 30)

    if st.button("Predict"):
        if model is None:
            st.error("⚠️ No trained model available.")
        else:
            features = np.array([[preg, gluc, bp, skin, insulin, bmi, dpf, age]])
            prediction = model.predict(features)[0]

            # probability = model.predict_proba(features)[0][1] * 100
            result = "🚫 No Diabetes" if prediction == 0 else "⚠️ Likely Diabetic"
            st.success(f"**Prediction:** {result}")
            # st.caption(f"Confidence: {prob:.2f}%")

# ==============================
# DATA ENTRY PAGE
# ==============================
# elif page == "📋 Add Medical Data":
# st.header("Add Verified Medical Data")
# st.info("⚠️ For authorized medical personnel only")

#     with st.form("data_entry_form"):
#         name = st.text_input("Staff Name (for record)")
#         password = st.text_input("Access Key", type="password")
#         preg = st.number_input("Pregnancies", 0, 20, 0)
#         gluc = st.number_input("Glucose Level", 0, 300, 100)
#         bp = st.number_input("Blood Pressure", 0, 200, 70)
#         skin = st.number_input("Skin Thickness", 0, 100, 20)
#         insulin = st.number_input("Insulin", 0, 900, 80)
#         bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
#         dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
#         age = st.number_input("Age", 0, 120, 30)
#         outcome = st.selectbox("Outcome", [0, 1])  # 0 = No Diabetes, 1 = Diabetes

#         submitted = st.form_submit_button("Submit Data")

#     if submitted:
#         if password != ACCESS_KEY:
#             st.error("❌ Invalid access key. Entry denied.")
#         else:
#             new_row = pd.DataFrame(
#                 [[preg, gluc, bp, skin, insulin, bmi, dpf, age, outcome]],
#                 columns=[
#                     "Pregnancies",
#                     "Glucose",
#                     "BloodPressure",
#                     "SkinThickness",
#                     "Insulin",
#                     "BMI",
#                     "DiabetesPedigreeFunction",
#                     "Age",
#                     "Outcome",
#                 ],
#             )

#             if os.path.exists(DATA_PATH):
#                 df = pd.read_csv(DATA_PATH)
#                 df = pd.concat([df, new_row], ignore_index=True)
#             else:
#                 df = new_row

#             df.to_csv(DATA_PATH, index=False)
#             st.success("✅ Data successfully added to dataset!")

#             # Retrain model
#             with st.spinner("Retraining model with updated data..."):
#                 new_model = train_and_save_model()
#             st.success("🧠 Model retrained and saved successfully!")

#             st.caption("New data has been used to improve prediction accuracy.")
