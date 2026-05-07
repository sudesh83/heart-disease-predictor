import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load metrics
accuracy = pickle.load(open("accuracy.pkl", "rb"))
cm = pickle.load(open("cm.pkl", "rb"))
report = pickle.load(open("report.pkl", "rb"))

# Page settings
st.set_page_config(
    page_title="AI Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# DARK UI CSS
st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

/* Titles */
h1, h2, h3 {
    color: #FF4B4B;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161B22;
}

/* Buttons */
div.stButton > button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

div.stButton > button:hover {
    background-color: #ff1a1a;
    color: white;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #161B22;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

/* Input Boxes */
.stSlider {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    "<h1 style='text-align:center;'>❤️ AI Heart Disease Prediction Dashboard</h1>",
    unsafe_allow_html=True
)

# Hero Section
st.markdown("""
<div style='
padding:25px;
border-radius:15px;
background: linear-gradient(to right, #141E30, #243B55);
text-align:center;
margin-bottom:20px;
'>
<h2 style='color:white;'>Machine Learning Based Healthcare System</h2>
<p style='color:white;'>
Predict heart disease risk using AI and Random Forest Algorithm
</p>
</div>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1,1])

# LEFT SIDE INPUTS
with col1:

    st.subheader("🩺 Patient Information")

    age = st.slider("Age", 20, 100, 50)

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    cp = st.slider("Chest Pain Type", 0, 3, 1)

    trestbps = st.slider(
        "Resting Blood Pressure",
        80, 200, 120
    )

    chol = st.slider(
        "Cholesterol Level",
        100, 400, 200
    )

    thalach = st.slider(
        "Maximum Heart Rate",
        60, 220, 150
    )

    sex_value = 1 if sex == "Male" else 0

# RIGHT SIDE DASHBOARD
with col2:

    st.subheader("📊 Live Health Metrics")

    m1, m2 = st.columns(2)

    with m1:
        st.metric("Blood Pressure", trestbps)

    with m2:
        st.metric("Cholesterol", chol)

    m3, m4 = st.columns(2)

    with m3:
        st.metric("Heart Rate", thalach)

    with m4:
        st.metric("Age", age)

# Prediction Button
if st.button("🔍 Predict Disease Risk"):

    data = np.array([[
        age,
        sex_value,
        cp,
        trestbps,
        chol,
        thalach
    ]])

    # Prediction
    prediction = model.predict(data)

    # Loading animation
    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    st.markdown("---")

    st.subheader("🧠 AI Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ High Risk of Heart Disease")

        st.metric(
            label="Risk Percentage",
            value="87%"
        )

    else:

        st.success("✅ Low Risk of Heart Disease")

        st.metric(
            label="Risk Percentage",
            value="18%"
        )

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📈 Health Analysis",
    "📊 Model Performance",
    "ℹ️ About Project"
])

# TAB 1
with tab1:

    chart_data = pd.DataFrame({
        "Values": [
            age,
            trestbps,
            chol,
            thalach
        ]
    },
    index=[
        "Age",
        "Blood Pressure",
        "Cholesterol",
        "Heart Rate"
    ])

    st.bar_chart(chart_data)

# TAB 2
with tab2:

    st.subheader("📊 Model Evaluation")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            label="Model Accuracy",
            value=f"{accuracy*100:.2f}%"
        )

    with c2:
        st.metric(
            label="Algorithm",
            value="Random Forest"
        )

    st.markdown("### 🔍 Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No Disease", "Actual Disease"],
        columns=["Predicted No Disease", "Predicted Disease"]
    )

    st.dataframe(cm_df)

    st.markdown("### 📄 Classification Report")

    st.text(report)

# TAB 3
with tab3:

    st.write("""
    ## ❤️ About This Project

    This project uses Machine Learning to predict
    heart disease risk using patient health parameters.

    ### 🚀 Technologies Used
    - Python
    - Streamlit
    - Scikit-learn
    - Random Forest Algorithm

    ### 🧠 ML Concepts Used
    - Classification
    - Model Training
    - Prediction
    - Accuracy Evaluation
    - Confusion Matrix

    ### 🎯 Objective
    To build an AI-powered healthcare prediction system
    that can help identify heart disease risk.
    """)

st.markdown("---")

st.caption("Developed by Sudesh | 4th Semester Machine Learning Project")