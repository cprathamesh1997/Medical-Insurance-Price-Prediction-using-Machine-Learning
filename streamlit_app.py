import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import os

@st.cache_data
def load_data():
    try:
        csv_path = "Medical_insurance.csv"
        if not os.path.exists(csv_path):
            st.error(f"Error: '{csv_path}' not found. Please upload it to the root directory.")
            return None, None
        df = pd.read_csv(csv_path)
        df = df.drop(['region'], axis=1)
        df['sex'] = df['sex'].map({'male': 1, 'female': 0})
        df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
        X = df.drop('charges', axis=1)
        y = df['charges']
        return X, y
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


@st.cache_resource
def train_model(X, y):
    if X is None or y is None:
        return None
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=15, max_depth=3, gamma=0)
    model.fit(X_train, y_train)
    return model

def set_custom_style():
    st.markdown("""
    <style>
    /* Dark blue gradient background */
    .stApp {
        background: linear-gradient(to right, #1e3a8a, #3b82f6);
        padding: 20px;
    }
    /* Title with light blue glow */
    h1 {
        color: #ffffff !important;
        text-align: center !important;
        font-family: 'Arial', sans-serif !important;
        font-size: 36px !important;
        font-weight: bold !important;
        letter-spacing: 2px !important;
        text-shadow: 0 0 10px #87CEEB, 0 0 20px #87CEEB, 0 0 30px #87CEEB !important;
    }
    /* Description (for the main description under the title) */
    .stMarkdown p:not(.feature-desc):not(.warning):not(.footer) {
        color: #e0f7fa !important;
        font-size: 18px !important;
        text-align: center !important;
        font-family: 'Arial', sans-serif !important;
    }
    /* Form container */
    .stForm {
        background-color: rgba(255, 255, 255, 0.95) !important;
        padding: 15px !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    /* Button - More specific selector */
    div[data-testid="stForm"] .stButton>button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 5px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        transition: background-color 0.3s ease !important;
    }
    div[data-testid="stForm"] .stButton>button:hover {
        background-color: #1e40af !important;
    }
    /* Custom prediction result container */
    .prediction-result {
        background-color: #1e3a8a !important; /* Dark blue background */
        color: #ffffff !important; /* White text */
        border-radius: 5px !important;
        padding: 10px !important;
        font-size: 18px !important;
        font-weight: bolder !important;
        text-align: center !important;
        box-shadow: 0 0 10px #1e3a8a !important;
        border: 2px solid #87CEEB !important; /* Light blue border */
    }
    /* Input labels */
    .stNumberInput label, .stSelectbox label {
        color: #1e3a8a !important;
        font-weight: bold !important;
        font-family: 'Arial', sans-serif !important;
    }
    /* Feature descriptions - Custom class */
    .feature-desc {
        color: #333333 !important; /* Dark gray for visibility on white frame */
        font-size: 16px !important;
        font-style: italic !important;
        font-weight: 500 !important; /* Slightly bolder for readability */
        font-family: 'Arial', sans-serif !important;
        margin-top: -10px !important;
        margin-bottom: 15px !important;
    }
    /* Warning message - Red and bold */
    .warning {
        color: #FF073A !important;
        font-size: 14px !important;
        text-align: center !important;
        font-style: italic !important;
        font-weight: bold !important;
        margin-top: 10px !important;
    }
    /* Footer */
    .footer {
        color: #e0f7fa !important;
        font-size: 12px !important;
        text-align: center !important;
        margin-top: 20px !important;
        font-family: 'Arial', sans-serif !important;
    }
    /* Frame containers */
    .frame {
        background-color: rgba(255, 255, 255, 0.95) !important; /* Solid white background */
        padding: 15px !important;
        border-radius: 10px !important;
        margin-bottom: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
   
    st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🏥", layout="centered")

    set_custom_style()

    X, y = load_data()
    if X is None or y is None:
        return
    model = train_model(X, y)
    if model is None:
        return

    # Title and Description
    with st.container():
        st.markdown("<div class='frame'>", unsafe_allow_html=True)
        st.title("MEDICAL INSURANCE PRICE PREDICTION 🩺💰📊")
        st.write("Provide your details to estimate your insurance cost instantly!")
        st.markdown("</div>", unsafe_allow_html=True)

    #  Input Form
    with st.container():
        st.markdown("<div class='frame'>", unsafe_allow_html=True)
        with st.form(key='prediction_form'):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", min_value=5, max_value=80, value=30, step=1, help="Enter your age (5-80)")
                age = int(age)  # Ensure integer type
                st.markdown("<p class='feature-desc'>Older age often increases insurance costs due to higher health risks.</p>", unsafe_allow_html=True)
                sex = st.selectbox("Sex", ["Male", "Female"], help="Select your gender")
                st.markdown("<p class='feature-desc'>Gender can slightly affect costs due to differing health patterns.</p>", unsafe_allow_html=True)

            with col2:
                bmi = st.number_input("BMI", min_value=1.0, max_value=80.0, value=30.0, step=0.1, help="Enter your BMI (1-80)")
                bmi = float(bmi)  # Ensure float type
                st.markdown("<p class='feature-desc'>Higher BMI may raise costs due to obesity-related health risks.</p>", unsafe_allow_html=True)
                children = st.slider("Number of Children", 0, 10, 0, help="Slide to set number of dependents")
                st.markdown("<p class='feature-desc'>More children can increase costs due to family coverage needs.</p>", unsafe_allow_html=True)

            smoker = st.selectbox("Smoker?", ["No", "Yes"], help="Select smoking status")
            st.markdown("<p class='feature-desc'>Smoking significantly raises costs due to associated health issues.</p>", unsafe_allow_html=True)

            submit_button = st.form_submit_button(label="Predict Cost")
        st.markdown("</div>", unsafe_allow_html=True)
  
    with st.container():
        st.markdown("<div class='frame'>", unsafe_allow_html=True)
        if submit_button:
            with st.spinner("Calculating your cost..."):
                input_data = pd.DataFrame({
                    'age': [age],
                    'sex': [1 if sex == "Male" else 0],
                    'bmi': [bmi],
                    'children': [children],
                    'smoker': [1 if smoker == "Yes" else 0]
                })
                prediction = model.predict(input_data)[0]

            st.markdown(
                f"""
                <div class='prediction-result'>
                    Estimated Insurance Cost: ₹{prediction:,.2f}/-
                </div>
                """,
                unsafe_allow_html=True
            )
            # Warning message
            st.markdown("""
            <p class='warning'>
            Warning: For proper financial analysis, consult with an advisor. This estimate is generated by Medical Insurance Price Predictor and should not be considered an actual insurance quote.
            </p>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <p class='footer'>
    © 2025 Medical Insurance Price Predictor | Built with ❤ using Streamlit | Deployed on Hugging Face
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()