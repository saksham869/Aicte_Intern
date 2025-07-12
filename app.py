import streamlit as st
import joblib
import numpy as np

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="GHG Emission Predictor 🌱",
    page_icon="🌱",
    layout="centered"
)

# -------------------- Load Model & Scaler --------------------
try:
    model = joblib.load('models/rf_tuned_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    st.error(f"🚨 Failed to load model or scaler: {e}")
    st.stop()

# -------------------- Header --------------------
st.markdown("<h1 style='text-align: center;'>🌱 GHG Emission Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "This app predicts greenhouse gas (GHG) emissions (in kg CO₂e) "
    "based on various industrial emission factors using a machine learning model.",
    unsafe_allow_html=True
)
st.markdown("---")

# -------------------- Form Input --------------------
st.subheader("📥 Input Features")
with st.form("ghg_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        f1 = st.number_input("🏭 Industry Code", value=111)
        f2 = st.number_input("📅 Year", value=2016)
        f3 = st.number_input("💰 kg CO₂e per USD", value=0.01, step=0.01)
        f4 = st.number_input("🔗 Supply Chain Emission Factor", value=0.01, step=0.01)
        f5 = st.number_input("🚚 Transport Emission Factor", value=0.01, step=0.01)

    with col2:
        f6 = st.number_input("🏗️ Manufacturing Emission Factor", value=0.01, step=0.01)
        f7 = st.number_input("🔩 Material Emission Factor", value=0.01, step=0.01)
        f8 = st.number_input("⚡ Energy Emission Factor", value=0.01, step=0.01)
        f9 = st.number_input("🌫️ Other GHG Emissions", value=0.01, step=0.01)

    submitted = st.form_submit_button("🔍 Predict GHG Emission")

# -------------------- Prediction --------------------
if submitted:
    try:
        input_data = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.markdown("---")
        st.success(f"✅ **Predicted GHG Emission:** `{prediction:.2f} kg CO₂e`")

    except ValueError as ve:
        st.error(f"⚠️ Input Error: {ve}")
    except Exception as e:
        st.error(f"🚨 Unexpected Error: {e}")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(
    "<small>🧠 Built using Random Forest Regressor & Streamlit | "
    "📊 Data preprocessed with StandardScaler</small>",
    unsafe_allow_html=True
)
