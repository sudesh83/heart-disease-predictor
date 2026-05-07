import pandas as pd
import streamlit as st
import pickle
import numpy as np
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD MODEL & METRICS
# =========================
model = pickle.load(open("model.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))
cm = pickle.load(open("cm.pkl", "rb"))
report = pickle.load(open("report.pkl", "rb"))

# =========================
# DARK UI CSS
# =========================
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #050816;
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0B1120;
    border-right: 1px solid #1E293B;
}

/* Titles */
h1, h2, h3, h4 {
    color: #FF4B4B !important;
}

/* Text */
p, label, div {
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #ff4b4b, #ff1a1a);
    color: white;
    border-radius: 12px;
    border: none;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

/* Button Hover */
.stButton > button:hover {
    background: linear-gradient(90deg, #ff1a1a, #cc0000);
    color: white;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #1F2937;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #111827;
    border-radius: 10px;
    padding: 10px 20px;
    color: white;
}

/* Progress Bar */
.stProgress > div > div {
    background-color: #FF4B4B;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown(
    "<h1 style='text-align:center;'>❤️ AI Heart Disease Prediction Dashboard</h1>",
    unsafe_allow_html=True
)

# =========================
# HERO SECTION
# =========================
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

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1,1])

# =========================
# LEFT SIDE INPUTS
# =========================
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

# =========================
# RIGHT SIDE METRICS
# =========================
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

# =========================
# PREDICTION BUTTON
# =========================
if st.button("🔍 Predict Disease Risk"):

    data = np.array([[
        age,
        sex_value,
        cp,
        trestbps,
        chol,
        thalach
    ]])

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

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs([
    "📈 Health Analysis",
    "📊 Model Performance",
    "ℹ️ About Project"
])

# =========================
# TAB 1 - CHART
# =========================
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

# =========================
# TAB 2 - PERFORMANCE
# =========================
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

# =========================
# TAB 3 - ABOUT
# =========================
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

# =========================
# FOOTER
# =========================
st.markdown("---")

st.caption("Developed by Sudesh | 4th Semester Machine Learning Project")