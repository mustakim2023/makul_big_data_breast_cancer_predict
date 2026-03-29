import pandas as pd
import streamlit as st
from joblib import load
import base64

# ================= BACKGROUND =================
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    img_base64 = get_base64("gambar.jpg")

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)
except:
    pass

# ================= TITLE =================
st.title('Aplikasi Prediksi Breast Cancer')
st.subheader('MUSTAKIM - 204032510007')

# ================= LOAD MODEL =================
model = load('model_prediksi_breast_cancer.joblib')

st.write('Silahkan masukkan data Pasien')

# ================= INPUT =================
feature_names = [
    'concave_points_worst',
    'radius_worst',
    'concave_points_mean',
    'perimeter_worst',
    'texture_worst',
    'area_worst',
    'concavity_worst',
    'area_mean',
    'smoothness_worst',
    'concavity_mean'
]

input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature, min_value=0.0, value=0.0)

# ================= PREDIKSI =================
if st.button('Tes Prediksi'):
    df_input = pd.DataFrame([input_data])

    prediction = model.predict(df_input)

    if prediction[0] == 1:
        st.error('Cancer Ganas')
    else:
        st.success('Cancer Jinak')
