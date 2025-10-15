import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('placement_regressor.pkl', 'rb'))

# App Title
st.title("🎓 Placement Prediction App")

st.markdown("""
This tool predicts the **likelihood of student placement** based on academic and skill parameters.
""")

# Input form
cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
major_projects = st.slider("Major Projects", 0, 5, 1)
certifications = st.slider("Workshops / Certifications", 0, 10, 2)
mini_projects = st.slider("Mini Projects", 0, 5, 2)
skills = st.slider("Skills Rating (out of 10)", 1, 10, 6)
communication = st.slider("Communication Skill Rating (out of 10)", 1, 10, 7)
internship = st.slider("Internships", 0, 3, 1)
hackathon = st.slider("Hackathon Participation", 0, 5, 1)
perc_12 = st.slider("12th Percentage", 30, 100, 75)
backlogs = st.slider("No. of Backlogs", 0, 10, 0)

# Predict button
if st.button("Predict Placement Likelihood"):
    input_data = pd.DataFrame([[
        cgpa, major_projects, certifications, mini_projects, skills,
        communication, internship, hackathon, perc_12, backlogs
    ]], columns=[
        'CGPA', 'Major Projects', 'Workshops/Certificatios', 'Mini Projects',
        'Skills', 'Communication Skill Rating', 'Internship', 'Hackathon',
        '12th Percentage', 'backlogs'
    ])

    # 💡 Check average of inputs to catch low-score edge cases
    avg_input = input_data.values.flatten().mean()

    if avg_input < 2.0:
        prediction = 0.0
    else:
        prediction = np.clip(model.predict(input_data)[0], 0.0, 1.0)

    st.subheader(f"📊 Estimated Placement Probability: **{prediction * 100:.2f}%**")

    if prediction >= 0.7:
        st.success("✅ Great! The student is likely to get placed.")
    elif prediction >= 0.4:
        st.warning("⚠️ Moderate chance. Work on improving skills and projects.")
    else:
        st.error("❌ Low placement probability. Focus on CGPA, certifications, and internships.")



