import streamlit as st
import numpy as np
import joblib

# Load saved model, scaler, and PCA
model = joblib.load(r"C:\Users\candy\Desktop\ML_project\Models\diabetes_best_model.pkl")
scaler = joblib.load(r"C:\Users\candy\Desktop\ML_project\Models\diabetes_scaler.pkl")
pca = joblib.load(r"C:\Users\candy\Desktop\ML_project\Models\diabetes_pca.pkl")

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details below:")

# Input fields
preg = st.number_input("Pregnancies", min_value=0.0)
glucose = st.number_input("Glucose Level", min_value=0.0)
bp = st.number_input("Blood Pressure", min_value=0.0)
skin = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0.0)

if st.button("Predict Diabetes"):
    user_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    # Step 1: scale
    user_scaled = scaler.transform(user_input)

    # Step 2: PCA transform
    user_pca = pca.transform(user_scaled)

    # Step 3: predict
    prediction = model.predict(user_pca)[0]
    confidence = model.predict_proba(user_pca)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ Diabetes Detected! (Confidence: {confidence:.2f})")
    else:
        st.success(f"✔️ No Diabetes Detected (Confidence: {confidence:.2f})")
