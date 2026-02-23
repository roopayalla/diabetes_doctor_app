import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

@st.cache_data
def load_model():
    try:
        if os.path.exists('diabetes_model.pkl') and os.path.exists('scaler.pkl'):
            model = joblib.load('diabetes_model.pkl')
            scaler = joblib.load('scaler.pkl')
            return model, scaler
        return None, None
    except:
        return None, None

model, scaler = load_model()
st.set_page_config(page_title="Diabetes App", layout="wide", page_icon="🏥")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("👨‍⚕️ Diabetes Prediction System")
    col1, col2 = st.columns(2)
    with col1: 
        email = st.text_input("Doctor Email", value="doctor")
    with col2: 
        pwd = st.text_input("Password", type="password", value="1234")
    
    if st.button("🚀 Login", type="primary"):
        if email.lower().strip() == "doctor" and pwd == "1234":
            st.session_state.logged_in = True
            st.session_state.doctor_name = "Dr. Roopa"
            st.rerun()

def prediction_page():
    st.title("🏥 Patient Diabetes Prediction")
    st.info(f"**Logged in as**: {st.session_state.doctor_name}")
    
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("🔢 Patient ID", value="P001")
    with col2:
        doctor_name = st.text_input("👨‍⚕️ Doctor Name", value="Dr. Roopa")
    
    st.markdown("---")
    
    # Patient inputs
    st.subheader("📊 Health Parameters")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("👴 Age", 18, 100, 45)
        gender = st.selectbox("Gender", options=["Male (0)", "Female (1)"], format_func=lambda x: x)
        glucose_level = st.number_input("Glucose_level (mg/dL)", 50, 400, 120)
        blood_pressure = st.number_input("Blood_pressure", 70, 200, 120)
    with col2:
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        insulin = st.number_input("Insuline", 0, 500, 30)
        skin_thickness = st.number_input("Skin_Thickness", 0, 100, 20)
    
    pregnancies = st.number_input("Pregencies", 0, 20, 0)
    dpf = st.number_input("D.P.F", 0.0, 2.0, 0.5)
    cholesterol = st.number_input("Cholesterol", 100, 400, 200)
    
    if st.button("🔮 Predict Diabetes", type="primary"):
        if model and scaler and patient_id:
            patient_data = pd.DataFrame({
                'Age': [age], 'Gender': [0 if gender=="Male (0)" else 1], 
                'Height': [170], 'Weight': [75],
                'BMI': [bmi], 'Blood_pressure': [blood_pressure], 
                'Glucose_level': [glucose_level],
                'Insuline': [insulin], 'Skin_Thickness': [skin_thickness], 
                'Pregencies': [pregnancies],
                'D.P.F': [dpf], 'Cholesterol': [cholesterol], 'Heart_Rate': [85]
            })
            
            try:
                X_scaled = scaler.transform(patient_data)
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0][1]
                
                # ✅ PERFECT THRESHOLD LOGIC
                if prob > 0.65:  # 65% threshold
                    result = "🟡 DIABETES DETECTED"
                    st.error(f"⚠️ High Risk: {prob:.1%}")
                else:
                    result = "🟢 NO DIABETES"
                    st.success(f"✅ Low Risk: {prob:.1%}")
                
                # Results
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Patient ID", patient_id)
                col_b.metric("Doctor", doctor_name)
                col_c.metric("Result", result)
                
                st.metric("Diabetes Probability", f"{prob:.1%}")
                
                # Report
                report = f"""
DIABETES REPORT
===============
Patient: {patient_id}
Doctor: {doctor_name}
Result: {result}
Risk: {prob:.1%}
Glucose: {glucose_level}
BMI: {bmi:.1f}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
                """
                st.code(report)
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Complete all fields!")

# Main App
if not st.session_state.logged_in:
    login_page()
else:
    prediction_page()
    
    st.sidebar.title("👨‍⚕️ Controls")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()
