import streamlit as st
import numpy as np
import joblib

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# ================================
# LOAD MODEL, SCALER, PCA
# ================================
model = joblib.load(r"C:\Users\candy\Desktop\ML_project\Models\heart_best_model.pkl")
scaler = joblib.load(r"C:\Users\candy\Desktop\ML_project\Models\heart_scaler.pkl")
pca = joblib.load(r"C:\Users\candy\Desktop\ML_project\Models\heart_pca.pkl")

# ================================
# CUSTOM CSS FOR ENHANCED UI
# ================================
st.markdown("""
<style>
/* Title */
h1 {
    color: #ff4d4d;
    text-align: center;
    font-weight: 800;
}

/* Input card */
.block-container {
    padding-top: 2rem;
}

.stNumberInput>div>div>input {
    border-radius: 8px;
}

/* Prediction cards */
.result-good {
    background-color: #e6ffe6;
    border-left: 5px solid #33cc33;
    padding: 15px;
    border-radius: 8px;
}

.result-bad {
    background-color: #ffe6e6;
    border-left: 5px solid #ff3333;
    padding: 15px;
    border-radius: 8px;
}

/* Button */
.stButton>button {
    border-radius: 10px;
    background-color: #ff4d4d;
    color: white;
    padding: 10px 20px;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #cc0000;
}
</style>
""", unsafe_allow_html=True)

# ================================
# PAGE TITLE
# ================================
st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient details below:")

# ================================
# INPUT FIELDS
# ================================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.number_input("Sex (1=Male, 0=Female)", min_value=0, max_value=1)
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0.0)
    chol = st.number_input("Cholesterol Level", min_value=0.0)
    fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", min_value=0, max_value=1)

with col2:
    restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=0.0)
    exang = st.number_input("Exercise Induced Angina (1=True, 0=False)", min_value=0, max_value=1)
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, format="%.2f")
    slope = st.number_input("Slope (0-2)", min_value=0, max_value=2)
    ca = st.number_input("Number of major vessels (0-4)", min_value=0, max_value=4)
    thal = st.number_input("Thal (0=Normal, 1=Fixed Defect, 2=Reversible)", min_value=0, max_value=2)

# ================================
# PREDICTION
# ================================
if st.button("Predict Heart Disease"):

    # Prepare input
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    # Scale
    scaled = scaler.transform(user_input)

    # PCA transform
    pca_features = pca.transform(scaled)

    # Predict
    prediction = model.predict(pca_features)[0]
    confidence = model.predict_proba(pca_features)[0][prediction]

    # Display result
    if prediction == 1:
        st.markdown(
            f"<div class='result-bad'><h3>⚠️ Heart Disease Detected</h3>"
            f"<p>Confidence: <b>{confidence:.2f}</b></p></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-good'><h3>✔️ No Heart Disease Detected</h3>"
            f"<p>Confidence: <b>{confidence:.2f}</b></p></div>",
            unsafe_allow_html=True
        )
