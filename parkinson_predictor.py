import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Parkinson's Disease Predictor",
    page_icon="🧠",
    layout="centered"
)

model = joblib.load(r"C:\Users\candy\Desktop\ML_project\Models\parkinson_best_model.pkl")
scaler = joblib.load(r"C:\Users\candy\Desktop\ML_project\Models\parkinson_scaler.pkl")
pca = joblib.load(r"C:\Users\candy\Desktop\ML_project\Models\parkinson_pca.pkl")

st.markdown("""
<style>
h1 {
    color: #5a00b3;
    text-align: center;
    font-weight: 900;
}

/* Prediction good */
.result-good {
    background-color: #e8ffe8;
    border-left: 6px solid #33cc33;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
}

/* Prediction bad */
.result-bad {
    background-color: #ffe8e8;
    border-left: 6px solid #ff3333;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
}

.stButton>button {
    border-radius: 10px;
    background-color: #6a0dad;
    color: white;
    padding: 10px 25px;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #4b0082;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Parkinson's Disease Prediction System")
st.write("Enter the vocal and biomedical measurements below:")

col1, col2 = st.columns(2)

with col1:
    MDVP_Fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0)
    MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0)
    MDVP_Flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0)
    MDVP_Jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, format="%.6f")
    MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, format="%.6f")
    MDVP_RAP = st.number_input("MDVP:RAP", min_value=0.0, format="%.6f")
    MDVP_PPQ = st.number_input("MDVP:PPQ", min_value=0.0, format="%.6f")
    Jitter_DDP = st.number_input("Jitter:DDP", min_value=0.0, format="%.6f")
    MDVP_Shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, format="%.6f")
    MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, format="%.6f")
    Shimmer_APQ3 = st.number_input("Shimmer:APQ3", min_value=0.0, format="%.6f")

with col2:
    Shimmer_APQ5 = st.number_input("Shimmer:APQ5", min_value=0.0, format="%.6f")
    MDVP_APQ = st.number_input("MDVP:APQ", min_value=0.0, format="%.6f")
    Shimmer_DDA = st.number_input("Shimmer:DDA", min_value=0.0, format="%.6f")
    NHR = st.number_input("NHR", min_value=0.0, format="%.6f")
    HNR = st.number_input("HNR", min_value=0.0, format="%.6f")
    RPDE = st.number_input("RPDE", min_value=0.0)
    DFA = st.number_input("DFA", min_value=0.0)
    spread1 = st.number_input("spread1", format="%.6f")
    spread2 = st.number_input("spread2", format="%.6f")
    D2 = st.number_input("D2", min_value=0.0)
    PPE = st.number_input("PPE", min_value=0.0, format="%.6f")

if st.button("Predict Parkinson's Disease"):

    # Convert inputs to array
    features = np.array([[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter_percent,
                          MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
                          MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3,
                          Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR,
                          RPDE, DFA, spread1, spread2, D2, PPE]])

    # Scale
    scaled = scaler.transform(features)

    # PCA transform
    pca_features = pca.transform(scaled)

    # Prediction
    pred = model.predict(pca_features)[0]
    confidence = model.predict_proba(pca_features)[0][pred]

    if pred == 1:
        st.markdown(
            f"<div class='result-bad'><h3>⚠️ Parkinson's Detected</h3>"
            f"<p>Confidence: <b>{confidence:.3f}</b></p></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-good'><h3>✔️ No Parkinson's Detected</h3>"
            f"<p>Confidence: <b>{confidence:.3f}</b></p></div>",
            unsafe_allow_html=True
        )
