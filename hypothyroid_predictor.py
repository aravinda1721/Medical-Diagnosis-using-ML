import streamlit as st
import numpy as np
import joblib

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Hypothyroid Disease Predictor",
    page_icon="🧬",
    layout="centered"
)

# ================================
# LOAD MODEL + SCALER + PCA
# ================================
model = joblib.load(r"C:\Users\candy\Desktop\ML_project\Models\hypothyroid_best_model.pkl")
scaler = joblib.load(r"C:\Users\candy\Desktop\ML_project\Models\hypothyroid_scaler.pkl")
pca = joblib.load(r"C:\Users\candy\Desktop\ML_project\Models\hypothyroid_pca.pkl")

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
h1 { color:#0073e6; text-align:center; font-weight:900; }

.result-good {
    background:#e7ffe7; border-left:6px solid #1fa51f;
    padding:15px; border-radius:12px; margin-top:15px;
}
.result-bad {
    background:#ffe6e6; border-left:6px solid #ff3333;
    padding:15px; border-radius:12px; margin-top:15px;
}
.stButton>button {
    background:#0066cc; color:white; padding:10px 25px;
    border-radius:10px; font-size:16px;
}
.stButton>button:hover { background:#004d99; }
</style>
""", unsafe_allow_html=True)

# ================================
# UI TITLE
# ================================
st.title("🧬 Hypothyroid Disease Prediction System")
st.write("Enter patient test values below:")

# ================================
# INPUT FORM
# ================================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    sex = st.selectbox("Sex", ["Female", "Male"])
    on_thyroxine = st.selectbox("Currently taking Thyroxine?", ["No", "Yes"])
    t3_measured = st.selectbox("Was T3 measurement performed?", ["No", "Yes"])

with col2:
    TSH = st.number_input("TSH Level", min_value=0.0, format="%.3f")
    T3 = st.number_input("T3 Level", min_value=0.0, format="%.3f")
    TT4 = st.number_input("TT4 Level", min_value=0.0, format="%.3f")

# Convert categorical values
sex_val = 1 if sex == "Male" else 0
thyroxine_val = 1 if on_thyroxine == "Yes" else 0
t3_measured_val = 1 if t3_measured == "Yes" else 0

# ================================
# PREDICTION
# ================================
if st.button("Predict Hypothyroidism"):

    input_data = np.array([[age, sex_val, thyroxine_val, TSH, t3_measured_val, T3, TT4]])

    # Apply scaler + PCA
    scaled = scaler.transform(input_data)
    transformed = pca.transform(scaled)

    prediction = model.predict(transformed)[0]
    confidence = model.predict_proba(transformed)[0][prediction]

    if prediction == 1:
        st.markdown(
            f"<div class='result-bad'><h3>⚠️ Hypothyroidism Detected</h3>"
            f"<p>Confidence: <b>{confidence:.3f}</b></p></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-good'><h3>✔️ No Hypothyroidism Detected</h3>"
            f"<p>Confidence: <b>{confidence:.3f}</b></p></div>",
            unsafe_allow_html=True
        )
