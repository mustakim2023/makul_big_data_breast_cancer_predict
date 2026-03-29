# Mengimpor library
import pandas as pd
import streamlit as st
from joblib import load
import base64

# Menghilangkan warning
import warnings
warnings.filterwarnings("ignore")

#  ================= BACKGROUND IMAGE =================

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64("gambar.jpg")

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

/*  CARD PUTIH */
.block-container {{
    background-color: rgba(255, 255, 255, 0.9);
    padding: 30px;
    border-radius: 15px;
    max-width: 900px;
    margin-top: 30px;
}}

/* input agar lebih jelas */
input, .stNumberInput input {{
    background-color: white !important;
}}

/* label lebih tegas */
label {{
    font-weight: bold;
    color: black !important;
}}

</style>
""", unsafe_allow_html=True)

#  ================= END BACKGROUND =================


# Membuat judul
st.title('TUGAS KULIAH BIG DATA')
st.title('MMPJJ25A - TELKOM UNIVERSITY')
st.title('Aplikasi Machine Learning untuk Prediksi Breast Cancer')

# Menambah subheader
st.subheader('MUSTAKIM - 204032510007')

# Load model
my_model = load('model_prediksi_breast_cancer.joblib')

# Menulis text
st.write('Silahkan masukkan data Pasien')

# ================= INPUT =================

# Baris Pertama
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        concave_points_worst = st.number_input('concave_points_worst', min_value=0.0, value=0.0)
    with col2:
        radius_worst = st.number_input('radius_worst', min_value=0.0, value=0.0)

# Baris Kedua
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        concave_points_mean = st.number_input('concave_points_mean', min_value=0.0, value=0.0)
    with col2:
        perimeter_worst = st.number_input('perimeter_worst', min_value=0.0, value=0.0)

# Baris Ketiga
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        texture_worst = st.number_input('texture_worst', min_value=0.0, value=0.0)
    with col2:
        area_worst = st.number_input('area_worst', min_value=0.0, value=0.0)

# Baris Keempat
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        concavity_worst = st.number_input('concavity_worst', min_value=0.0, value=0.0)
    with col2:
        area_mean = st.number_input('area_mean', min_value=0.0, value=0.0)

# Baris Kelima
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        smoothness_worst = st.number_input('smoothness_worst', min_value=0.0, value=0.0)
    with col2:
        concavity_mean = st.number_input('concavity_mean', min_value=0.0, value=0.0)

# ================= PREDIKSI =================

if st.button('Tes Prediksi Keganasan Breast Cancer'):

    # 🔥 PERBAIKAN UTAMA: gunakan DataFrame
    input_data = pd.DataFrame([{
        'concave_points_worst': concave_points_worst,
        'radius_worst': radius_worst,
        'concave_points_mean': concave_points_mean,
        'perimeter_worst': perimeter_worst,
        'texture_worst': texture_worst,
        'area_worst': area_worst,
        'concavity_worst': concavity_worst,
        'area_mean': area_mean,
        'smoothness_worst': smoothness_worst,
        'concavity_mean': concavity_mean
    }])

    # Prediksi
    cancer_predict = my_model.predict(input_data)

    # Hasil
    if cancer_predict[0] == 1:
        cancer_diagnosis = 'Cancer Ganas'
    else:
        cancer_diagnosis = 'Cancer Jinak'

    st.success(cancer_diagnosis)
