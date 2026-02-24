import streamlit as st
import numpy as np
import joblib

MODEL_PATH = r"C:\Users\candy\Desktop\ML_project\Models\lungs_best_model.pkl"
SCALER_PATH = r"C:\Users\candy\Desktop\ML_project\Models\lung_scaler.pkl"
PCA_PATH = r"C:\Users\candy\Desktop\ML_project\Models\lung_pca.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)

st.title("🫁 Lung Cancer Prediction System")
st.write("Enter the details below to predict lung cancer risk.")

gender = st.selectbox("Gender (M/F)", ["M", "F"])
age = st.number_input("Age", min_value=1, max_value=120, step=1)
smoking = st.selectbox("Smoking", [0, 1])
yellow_fingers = st.selectbox("Yellow Fingers", [0, 1])
anxiety = st.selectbox("Anxiety", [0, 1])
peer_pressure = st.selectbox("Peer Pressure", [0, 1])
chronic_disease = st.selectbox("Chronic Disease", [0, 1])
fatigue = st.selectbox("Fatigue", [0, 1])
allergy = st.selectbox("Allergy", [0, 1])
wheezing = st.selectbox("Wheezing", [0, 1])
alcohol_consumption = st.selectbox("Alcohol Consumption", [0, 1])
coughing = st.selectbox("Persistent Coughing", [0, 1])
shortness_breath = st.selectbox("Shortness of Breath", [0, 1])
swallowing_difficulty = st.selectbox("Difficulty Swallowing", [0, 1])
chest_pain = st.selectbox("Chest Pain", [0, 1])

gender = 1 if gender == "M" else 0

input_data = np.array([[
    gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
    chronic_disease, fatigue, allergy, wheezing, alcohol_consumption,
    coughing, shortness_breath, swallowing_difficulty, chest_pain
]])

if st.button("Predict"):

    scaled_input = scaler.transform(input_data)

    transformed_input = pca.transform(scaled_input)

    prediction = model.predict(transformed_input)[0]
    probability = model.predict_proba(transformed_input)[0][1] * 100

    st.subheader("🔍 Prediction Result:")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Lung Cancer ({probability:.2f}%)")
    else:
        st.success(f"🟢 Low Risk of Lung Cancer ({probability:.2f}%)")

    st.write("📊 Model Confidence:", f"**{probability:.2f}%**")
