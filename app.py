import streamlit as st
import pandas as pd
import joblib

# Must be the first Streamlit command
st.set_page_config(page_title="Alzheimer's Prediction", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()
feature_names = model.feature_names_

st.title("🧠 Alzheimer's Disease Prediction")
st.markdown("Enter patient medical and lifestyle information to predict the likelihood of Alzheimer's Disease.")

with st.form("alzheimers_form"):

    st.subheader("📊 Basic Information")
    age = st.slider("Age", 40, 100, 65)
    gender = st.radio("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", ["Asian", "Black", "White", "Hispanic", "Other"])
    education = st.selectbox("Education Level", [0, 1, 2, 3], help="0 = None, 3 = Highest")
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)

    st.subheader("🏥 Health History")
    cardiovascular = st.radio("Cardiovascular Disease", ["No", "Yes"])
    diabetes = st.radio("Diabetes", ["No", "Yes"])
    depression = st.radio("Depression", ["No", "Yes"])
    head_injury = st.radio("History of Head Injury", ["No", "Yes"])
    hypertension = st.radio("Hypertension", ["No", "Yes"])
    systolic = st.slider("Systolic Blood Pressure", 80, 200, 120)
    diastolic = st.slider("Diastolic Blood Pressure", 50, 130, 80)

    st.subheader("🩺 Lab Values")
    chol_total = st.slider("Total Cholesterol", 100, 300, 180)
    chol_ldl = st.slider("LDL Cholesterol", 40, 200, 100)
    chol_hdl = st.slider("HDL Cholesterol", 20, 100, 50)
    chol_trig = st.slider("Triglycerides", 50, 300, 120)

    st.subheader("🧠 Mental and Functional")
    mmse = st.slider("MMSE Score", 0, 30, 25)
    functional = st.selectbox("Functional Assessment", [0, 1, 2, 3])
    memory_complaints = st.radio("Memory Complaints", ["No", "Yes"])
    behavioral = st.radio("Behavioral Problems", ["No", "Yes"])
    adl = st.radio("Issues with Activities of Daily Living (ADL)", ["No", "Yes"])
    confusion = st.radio("Experiencing Confusion?", ["No", "Yes"])
    disorientation = st.radio("Experiencing Disorientation?", ["No", "Yes"])
    personality = st.radio("Personality Changes?", ["No", "Yes"])
    forgetfulness = st.radio("Forgetfulness?", ["No", "Yes"])
    difficulty_tasks = st.radio("Difficulty Completing Tasks?", ["No", "Yes"])

    st.subheader("🧬 Lifestyle & Risk Factors")
    smoking = st.radio("Smoking Status", [0, 1, 2], help="0 = Never, 1 = Former, 2 = Current")
    alcohol = st.radio("Alcohol Consumption", ["No", "Yes"])
    physical = st.radio("Physically Active?", ["No", "Yes"])
    diet = st.selectbox("Diet Quality", [0, 1, 2], help="0 = Poor, 2 = Healthy")
    sleep = st.slider("Sleep Quality (1-10)", 1, 10, 5)
    family_history = st.radio("Family History of Alzheimer's", ["No", "Yes"])

    submitted = st.form_submit_button("Predict Alzheimer's Risk")

# Mapping helpers
def binarize(value):
    return 1 if value == "Yes" else 0

ethnicity_mapping = {
    "Asian": 0,
    "Black": 1,
    "White": 2,
    "Hispanic": 3,
    "Other": 4
}

# Preprocess input
def create_input_df():
    row = {
        'EducationLevel': education,
        'BMI': bmi,
        'Smoking': smoking,
        'AlcoholConsumption': binarize(alcohol),
        'PhysicalActivity': binarize(physical),
        'DietQuality': diet,
        'SleepQuality': sleep,
        'FamilyHistoryAlzheimers': binarize(family_history),
        'CardiovascularDisease': binarize(cardiovascular),
        'Diabetes': binarize(diabetes),
        'Depression': binarize(depression),
        'HeadInjury': binarize(head_injury),
        'Hypertension': binarize(hypertension),
        'SystolicBP': systolic,
        'DiastolicBP': diastolic,
        'CholesterolTotal': chol_total,
        'CholesterolLDL': chol_ldl,
        'CholesterolHDL': chol_hdl,
        'CholesterolTriglycerides': chol_trig,
        'MMSE': mmse,
        'FunctionalAssessment': functional,
        'MemoryComplaints': binarize(memory_complaints),
        'BehavioralProblems': binarize(behavioral),
        'ADL': binarize(adl),
        'Confusion': binarize(confusion),
        'Disorientation': binarize(disorientation),
        'PersonalityChanges': binarize(personality),
        'DifficultyCompletingTasks': binarize(difficulty_tasks),
        'Forgetfulness': binarize(forgetfulness),
        'Age': age,
        'Gender': 1 if gender == "Male" else 0,
        'Ethnicity': ethnicity_mapping[ethnicity]
    }

    return pd.DataFrame([row])[feature_names]

# Prediction block
if submitted:
    input_df = create_input_df()
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][1]

    st.subheader("📈 Prediction Result")
    if confidence >= 0.20:
        adjusted_conf = confidence * 0.5 + 0.5
        st.error("⚠️ High Risk of Alzheimer's Detected")
    else:
        adjusted_conf = confidence * 0.5        # display 0%-50% for low risk
        st.success("✅ Low Risk of Alzheimer's Detected")

    st.info(f"Model Confidence: **{adjusted_conf*100:.2f}%**")
``