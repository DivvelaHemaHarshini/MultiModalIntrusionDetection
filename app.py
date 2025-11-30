# ================================================
# 🛡 Intrusion Detection System - Streamlit App
# ================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
import plotly.express as px

@st.cache_resource
def load_models():
    rf = joblib.load("random_forest_model.pkl")
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgboost_model.json")
    cnn = load_model("cnn_model.keras")
    le = joblib.load("label_encoder.pkl")
    scaler = joblib.load("newscaler.pkl")
    return rf, xgb_model, cnn, le, scaler

rf_model, xgb_model, cnn_model, label_encoder, scaler = load_models()

st.set_page_config(page_title="IDS System", layout="wide")

st.markdown("""
    <style>
.main-title {
    text-align: center;
    font-size: 40px !important;
    font-weight: bold;
    background: linear-gradient(90deg, #ff6a00, #ee0979);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
        .subheader { color: #2E86C1; font-weight: bold; font-size: 20px; }
        .section { background-color: #f8f9fa; padding: 20px; border-radius: 10px; }
        .footer-text { text-align: center; color: gray; margin-top: 40px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='main-title'>🛡 Intrusion Detection System (IDS)</p>", unsafe_allow_html=True)
st.write("Upload a CSV file **or** manually enter network features to detect possible attacks.")

# -------------------------------
# 📂 CSV UPLOAD SECTION (FIRST)
# -------------------------------
st.markdown("<p class='subheader'>📂 Upload CSV File</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV (must contain all 37 features)", type=["csv"])
data = None

feature_list = [
    "Destination Port","Flow Duration",
    "Total Fwd Packets","Total Backward Packets",
    "Total Length of Fwd Packets","Total Length of Bwd Packets",
    "Fwd Packet Length Mean","Fwd Packet Length Std",
    "Bwd Packet Length Mean","Bwd Packet Length Std",
    "Flow Bytes/s","Flow Packets/s",
    "Flow IAT Mean","Flow IAT Std",
    "Fwd IAT Mean","Bwd IAT Mean",
    "FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count",
    "ACK Flag Count","URG Flag Count",
    "Fwd Header Length","Bwd Header Length",
    "Fwd Packets/s","Bwd Packets/s",
    "Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std",
    "Average Packet Size","Avg Fwd Segment Size","Avg Bwd Segment Size",
    "Init_Win_bytes_forward","Init_Win_bytes_backward",
    "Active Mean","Idle Mean"
]

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("📁 CSV Loaded Successfully!")
    st.dataframe(data.head())

    missing_cols = set(feature_list) - set(data.columns)
    if missing_cols:
        st.error(f"❌ Missing columns in CSV: {missing_cols}")
        data = None

st.write("---")

# -------------------------------
# 🧾 MANUAL INPUT SECTION (SECOND)
# -------------------------------
manual_input = {}

with st.expander("📝 Enter Features Manually"):
    st.write("Fill in the network traffic feature values:")
    
    # Create two-column layout
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(feature_list):
        if i % 2 == 0:
            manual_input[feature] = col1.number_input(f"{feature}", value=0.0)
        else:
            manual_input[feature] = col2.number_input(f"{feature}", value=0.0)


# -------------------------------
# 🚀 PREDICTION SECTION
# -------------------------------
if st.button("🚀 Predict Attack"):
    X_scaled = scaler.transform(data)
    
    X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    rf_pred = rf_model.predict(X_scaled)
    dtest = xgb.DMatrix(X_scaled, feature_names=feature_list)
    xgb_pred = xgb_model.predict(dtest).astype(int)
    cnn_pred = np.argmax(cnn_model.predict(X_cnn), axis=1)

    rf_label = label_encoder.inverse_transform(rf_pred)
    xgb_label = label_encoder.inverse_transform(xgb_pred)
    cnn_label = label_encoder.inverse_transform(cnn_pred)

    final_result = []
    for i in range(len(data)):
        votes = [rf_label[i], xgb_label[i], cnn_label[i]]
        if votes.count("BENIGN") >= 2:
            final_result.append(("🟢 SAFE", "No Intrusion"))
        else:
            attack_type = max(set(votes), key=votes.count)
            final_result.append(("🔴 ATTACK DETECTED", attack_type))

    result_df = pd.DataFrame({
        "Random Forest": rf_label,
        "XGBoost": xgb_label,
        "CNN-LSTM": cnn_label,
        "Final Status": [x[0] for x in final_result],
        "Attack Type": [x[1] for x in final_result]
    })

    st.write("### 📌 Prediction Results")
    st.dataframe(result_df)

        
    st.markdown("## 📊 IDS Dashboard Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows", len(data))
    attack_count = sum(result_df["Final Status"] == "🔴 ATTACK DETECTED")
    col2.metric("Total Attacks", attack_count)
    col3.metric("Benign Flows", len(data) - attack_count)

    # Attack Types Distribution
    attack_types = result_df[result_df["Final Status"] == "🔴 ATTACK DETECTED"]["Attack Type"].value_counts()
    if not attack_types.empty:
        fig_attack = px.pie(values=attack_types.values, names=attack_types.index, title="Attack Types Distribution")
        st.plotly_chart(fig_attack)

    # Filter results by final status
    # status_filter = st.selectbox("Filter by Final Status", options=["All", "🟢 SAFE", "🔴 ATTACK DETECTED"])
    # if status_filter != "All":
    #     filtered_df = result_df[result_df["Final Status"] == status_filter]
    # else:
    #     filtered_df = result_df
    # st.dataframe(filtered_df)

    # Model Confidence Plot for single row input
    if len(data) == 1:
        st.subheader("Model Confidence Levels Comparison")

        prob_df = pd.DataFrame({
            "Class": label_encoder.classes_,
            "Random Forest": rf_label[0],
            "XGBoost": xgb_label[0],  # Ensure xgb_probs_2class created for binary/multiclass
            "CNN-LSTM": cnn_label[0]
        })

        prob_long = prob_df.melt(id_vars="Class", var_name="Model", value_name="Confidence")

        fig_conf = px.bar(
            prob_long,
            x="Class",
            y="Confidence",
            color="Model",
            barmode="group",
            title="Model Confidence Levels"
        )
        fig_conf.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig_conf)


    


# st.markdown("<p class='footer-text'>Developed for Final Year Project | Intrusion Detection System</p>", unsafe_allow_html=True)
