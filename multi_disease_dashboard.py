import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Smart Health Predictor", page_icon="🩺", layout="wide")

st.title("🧠 Smart Health Predictor Dashboard")
st.write("Choose a disease and enter patient details to analyze risk using trained ML models.")

# -------------------------------
# Disease Selection
# -------------------------------
disease = st.sidebar.selectbox(
    "Select Disease to Predict",
    ["Diabetes", "Heart Disease", "Parkinson's", "Hypothyroid", "Lung Cancer"]
)

st.sidebar.info(f"📌 You selected: **{disease}**")

# -------------------------------
# Load correct Models dynamically
# -------------------------------
def load_model(model_name):
    base = r"C:\Users\candy\Desktop\ML_project\Models"
    model = joblib.load(f"{base}\\{model_name}_best_model.pkl")
    scaler = joblib.load(f"{base}\\{model_name}_scaler.pkl")
    pca = joblib.load(f"{base}\\{model_name}_pca.pkl")
    return model, scaler, pca


# -------------------------------
# Dynamic UI Per Disease
# -------------------------------

def diabetes_form():
    preg = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    pedigree = st.number_input("Diabetes Pedigree", min_value=0.0)
    age = st.number_input("Age", min_value=1, max_value=120)
    return [preg, glucose, bp, skin, insulin, bmi, pedigree, age]


def heart_form():
    age = st.number_input("Age", min_value=1, max_value=100)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3)
    trestbps = st.number_input("Resting BP", min_value=60, max_value=200)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl", [0, 1])
    restecg = st.number_input("ECG Results (0-2)", min_value=0, max_value=2)
    thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0)
    slope = st.number_input("Slope (0-2)", min_value=0, max_value=2)
    ca = st.number_input("Number of Vessels (0-4)", min_value=0, max_value=4)
    thal = st.number_input("Thal (0-3)", min_value=0, max_value=3)
    return [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]


def parkinson_form():
    st.subheader("🎤 Parkinson's Voice Measurement Inputs")

    feature_labels = [
        'MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)',
        'MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)',
        'Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR',
        'RPDE','DFA','spread1','spread2','D2','PPE'
    ]

    values = []
    for label in feature_labels:
        values.append(st.number_input(label, value=0.0, step=0.01))

    return values


def hypo_form():
    age = st.number_input("Age", min_value=1, max_value=120)
    tsh = st.number_input("TSH Level")
    t3 = st.number_input("T3 Level")
    tt4 = st.number_input("TT4 Level")
    t4u = st.number_input("T4U Level")
    fti = st.number_input("FTI Level")
    sex = st.selectbox("Sex", [0, 1])
    return [age, tsh, t3, tt4, t4u, fti, sex]


def lung_form():
    gender = st.selectbox("Gender (M/F)", ["M", "F"])
    gender = 1 if gender == "M" else 0
    age = st.number_input("Age", 1, 120)
    sm = st.selectbox("Smoking", [0, 1])
    yf = st.selectbox("Yellow Fingers", [0, 1])
    anx = st.selectbox("Anxiety", [0, 1])
    pp = st.selectbox("Peer Pressure", [0, 1])
    cd = st.selectbox("Chronic Disease", [0, 1])
    fat = st.selectbox("Fatigue", [0, 1])
    allergy = st.selectbox("Allergy", [0, 1])
    wheeze = st.selectbox("Wheezing", [0, 1])
    alc = st.selectbox("Alcohol Consumption", [0, 1])
    cough = st.selectbox("Coughing", [0, 1])
    sob = st.selectbox("Shortness of Breath", [0, 1])
    swallow = st.selectbox("Swallowing Difficulty", [0, 1])
    chest = st.selectbox("Chest Pain", [0, 1])
    return [gender, age, sm, yf, anx, pp, cd, fat, allergy, wheeze, alc, cough, sob, swallow, chest]


# -------------------------------
# Build Form Based on Disease
# -------------------------------
forms = {
    "Diabetes": diabetes_form,
    "Heart Disease": heart_form,
    "Parkinson's": parkinson_form,
    "Hypothyroid": hypo_form,
    "Lung Cancer": lung_form
}

user_data = forms[disease]()

if st.button("🔍 Predict"):

    model, scaler, pca = load_model(disease.lower().replace(" ", "_"))

    scaled = scaler.transform([user_data])
    reduced = pca.transform(scaled)
    pred = model.predict(reduced)[0]
    prob = model.predict_proba(reduced)[0][1] * 100

    st.subheader("Prediction Result:")

    if pred == 1:
        st.error(f"⚠ HIGH RISK ({prob:.2f}%)")
    else:
        st.success(f"🟢 LOW RISK ({prob:.2f}%)")

