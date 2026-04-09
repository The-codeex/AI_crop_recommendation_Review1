
#  last working code  of app.py


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# -------------------------------
# Background Image
# -------------------------------

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_path):
    img_base64 = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Dark overlay so text remains readable */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.55);
            z-index: 0;
        }}

        /* Ensure all content sits above overlay */
        .stApp > * {{
            position: relative;
            z-index: 1;
        }}

        /* White text throughout */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stText, label,
        .stSelectbox label, .stNumberInput label,
        .stTextInput label, p {{
            color: #FFFFFF !important;
        }}

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            padding: 4px;
        }}

        .stTabs [data-baseweb="tab"] {{
            color: #FFFFFF !important;
            font-weight: 600;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: rgba(255, 255, 255, 0.2) !important;
            border-radius: 8px;
        }}

        /* Input fields */
        .stSelectbox div[data-baseweb="select"] > div,
        .stTextInput input,
        .stNumberInput input {{
            background-color: rgba(255, 255, 255, 0.15) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 8px;
        }}

        /* Predict button */
        .stButton > button {{
            background-color: #2D6A4F;
            color: #FFFFFF;
            font-weight: 700;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            transition: background-color 0.2s;
        }}

        .stButton > button:hover {{
            background-color: #52B788;
            color: #FFFFFF;
        }}

        /* Dataframe */
        .stDataFrame {{
            background-color: rgba(0, 0, 0, 0.4) !important;
            border-radius: 10px;
        }}

        /* Success box */
        .stSuccess {{
            background-color: rgba(45, 106, 79, 0.7) !important;
            border-radius: 8px;
            color: #FFFFFF !important;
        }}

        /* Subheader */
        .stSubheader {{
            color: #FFFFFF !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background — make sure back-img.jpg is in the same folder as app.py
set_background("image/back-img.jpg")


# -------------------------------
# Tabs
# -------------------------------

tab1, tab2 = st.tabs([
    "💰 Profit Based Recommendation",
    "🌱 Soil Based Recommendation"
])


# -------------------------------
# Load trained models
# -------------------------------

yield_model   = joblib.load("models/yield_model.pkl")
le_state      = joblib.load("models/state_encoder.pkl")
le_crop       = joblib.load("models/crop_encoder.pkl")
le_season     = joblib.load("models/season_encoder.pkl")
le_Dist     = joblib.load("models/Dist_encoder.pkl")




# -------------------------------
# Tab 1 — Profit Based Recommendation
# -------------------------------

with tab1:
    st.title("AI-Based Crop Recommendation System")
    st.write("Predict the most profitable crop")
    default_state = "Odisha"
    default_dist = "KORAPUT"
    

    state       = st.selectbox("Select State", le_state.classes_ , index=list(le_state.classes_).index(default_state))
    season      = st.selectbox("Select Season", le_season.classes_)
    Dist      = st.selectbox("Select Dist", le_Dist.classes_ ,index=list(le_Dist.classes_).index(default_dist))
    # district    = st.text_input("Type District")
    area        = st.number_input("Area (hectares)", min_value=1.0)
    year        = st.number_input("Year", min_value=2000, max_value=2035, value=2026)
    temperature = st.number_input("Temperature (°C)")
    rainfall    = st.number_input("Rainfall (mm)")

    if st.button("Predict Best Crop"):

        state_encoded  = le_state.transform([state])[0]
        season_encoded = le_season.transform([season])[0]
        Dist_encoded = le_Dist.transform([Dist])[0]

        candidate_crops = list(le_crop.classes_)
        results = []

        for crop in candidate_crops:

            crop_encoded = le_crop.transform([crop])[0]

            yield_input = pd.DataFrame([{
                "State":       state_encoded,
                "District_Name":Dist_encoded,
                "Crop":        crop_encoded,
                "Season":      season_encoded,
                "Area":        area,
                "Rainfall":    rainfall,
                "Temperature": temperature
            }])

            yield_log       = yield_model.predict(yield_input)
            predicted_yield = max(0.1, np.expm1(yield_log)[0])

            results.append({
                "Crop":                                crop,
                "Predicted Yield [metric tones/hec] ": predicted_yield,
                "Predicted Price [(₹)/Q] ":            0,
                "Cost/hec [(₹)]":                      0,
                "Expected Profit (₹ Lakh)":            0
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            by="Predicted Yield [metric tones/hec] ", ascending=False
        )

        st.subheader("Crop Profit Prediction")
        st.dataframe(results_df)

        best_crop = results_df.iloc[0]["Crop"]
        # st.success(f"Recommended Crop: {best_crop}")
        st.success(f"Recommended Crop: NA")






































