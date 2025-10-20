import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime

# === Konfigurasi Dashboard ===
st.set_page_config(
    page_title="Dashboard Cuaca Bandung",
    layout="wide",
    page_icon="🌤️"
)

# --- Styling CSS ---
st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #E0F7FA 0%, #80DEEA 100%); }
        h1, h2, h3, h4 { color: #004D40; font-family: 'Segoe UI', sans-serif; }
        .metric-box { background: white; padding: 15px; border-radius: 15px; box-shadow: 0px 3px 10px rgba(0,0,0,0.1); text-align: center; }
        .footer { text-align: center; padding: 10px; font-size: 0.85em; color: #004D40; }
    </style>
""", unsafe_allow_html=True)

# === Load Dataset ===
@st.cache_data
def load_data():
    df = pd.read_csv("cuaca_bandung_mapped_feature_engineered.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    choices = ['Tidak Hujan', 'Hujan Ringan', 'Hujan Sedang', 'Hujan Lebat']
    df['Kategori Hujan'] = pd.cut(df['Curah Hujan'], bins=[-1,0,10,30,1000], labels=choices)
    df['Musim'] = df['Bulan'].apply(lambda x: 'Hujan' if x in [11,12,1,2,3,4] else 'Kemarau')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File dataset tidak ditemukan!")
    st.stop()

# === Load Model dan Scaler ===
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_prediksi_hujan.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_model()

# === SIDEBAR FILTER ===
st.sidebar.title("Pengaturan Dashboard")
st.sidebar.markdown("Filter data cuaca:")

min_date = df["Tanggal"].min().date()
max_date = df["Tanggal"].max().date()
date_range = st.sidebar.date_input("Pilih rentang tanggal:", [min_date, max_date])

kategori_options = ['Semua'] + list(df['Kategori Hujan'].cat.categories)
kategori_hujan = st.sidebar.selectbox("Filter Kategori Hujan:", kategori_options)

musim_options = ['Semua'] + df['Musim'].unique().tolist()
musim_filter = st.sidebar.selectbox("Filter Musim:", musim_options)

# === Terapkan Filter ===
df_filtered = df.copy()
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_filtered[(df_filtered["Tanggal"].dt.date >= start_date) &
                              (df_filtered["Tanggal"].dt.date <= end_date)]
if kategori_hujan != 'Semua':
    df_filtered = df_filtered[df_filtered['Kategori Hujan'] == kategori_hujan]
if musim_filter != 'Semua':
    df_filtered = df_filtered[df_filtered['Musim'] == musim_filter]

# === HEADER ===
st.title("Dashboard Analitik & Prediksi Cuaca — Bandung")
st.markdown("Analisis data historis dan prediksi hujan di Kota Bandung secara interaktif.")

# === Tabs ===
tab1, tab2, tab3 = st.tabs(["Ringkasan", "Visualisasi", "Prediksi"])

# --- TAB 1: Ringkasan ---
with tab1:
    st.subheader("Ringkasan Data Cuaca")
    if not df_filtered.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='metric-box'><h4>Suhu Rata-rata</h4><h2>{df_filtered['Suhu Rata-rata'].mean():.1f} °C</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'><h4>Suhu Maksimum</h4><h2>{df_filtered['Suhu Maksimum'].mean():.1f} °C</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-box'><h4>Suhu Minimum</h4><h2>{df_filtered['Suhu Minimum'].mean():.1f} °C</h2></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='metric-box'><h4>Curah Hujan</h4><h2>{df_filtered['Curah Hujan'].mean():.1f} mm</h2></div>", unsafe_allow_html=True)

        st.markdown("Contoh Data")
        st.dataframe(df_filtered, use_container_width=True, height=400)
    else:
        st.warning("Tidak ada data untuk filter yang dipilih.")

# --- TAB 2: Visualisasi ---
with tab2:
    if not df_filtered.empty:
        st.subheader("Ringkasan Cepat Data")
        total_hari = len(df_filtered)
        avg_suhu = df_filtered['Suhu Rata-rata'].mean()
        avg_curah = df_filtered['Curah Hujan'].mean()
        hujan_counts = df_filtered['Kategori Hujan'].value_counts(normalize=True) * 100
        st.markdown(f"""
        Total Hari: {total_hari}  
        Suhu Rata-rata: {avg_suhu:.1f} °C  
        Curah Hujan Rata-rata: {avg_curah:.1f} mm  
        Hari Tidak Hujan: {hujan_counts.get('Tidak Hujan',0):.1f}%  
        Hari Hujan Ringan: {hujan_counts.get('Hujan Ringan',0):.1f}%  
        Hari Hujan Sedang: {hujan_counts.get('Hujan Sedang',0):.1f}%  
        Hari Hujan Lebat: {hujan_counts.get('Hujan Lebat',0):.1f}%
        """)

        st.subheader("Tren Suhu & Curah Hujan")
        fig_trend = px.line(df_filtered, x='Tanggal', y=['Suhu Rata-rata', 'Curah Hujan'],
                            labels={'value':'Nilai','variable':'Parameter'}, title="Tren Cuaca Bandung")
        st.plotly_chart(fig_trend, use_container_width=True)

        st.subheader("Korelasi Variabel Cuaca")
        numeric_cols = ['Suhu Maksimum', 'Suhu Minimum', 'Suhu Rata-rata', 'Curah Hujan', 'Kelembaban', 'Kecepatan Angin_Max', 'Kecepatan Angin_Avg']
        corr = df_filtered[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Heatmap Korelasi Cuaca", color_continuous_scale='Blues')
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Distribusi Curah Hujan")
        fig_hist = px.histogram(df_filtered, x="Curah Hujan", color="Kategori Hujan", nbins=20, title="Distribusi Curah Hujan")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Box Plot Suhu per Musim")
        fig_box = px.box(df_filtered, x="Musim", y="Suhu Rata-rata", color="Musim", title="Box Plot Suhu Rata-rata per Musim")
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Proporsi Kategori Hujan")
        pie_data = df_filtered['Kategori Hujan'].value_counts().reset_index()
        pie_data.columns = ['Kategori Hujan','Jumlah']
        fig_pie = px.pie(pie_data, names='Kategori Hujan', values='Jumlah', title="Proporsi Kategori Hujan")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Suhu Rata-rata vs Curah Hujan")
        fig_scatter = px.scatter(df_filtered, x="Suhu Rata-rata", y="Curah Hujan", color="Kategori Hujan",
                                 size="Kecepatan Angin_Max", hover_data=["Tanggal"], title="Suhu vs Curah Hujan")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Tidak ada data untuk filter yang dipilih.")

# --- TAB 3: Prediksi ---
with tab3:
    st.subheader("Prediksi Hujan / Tidak Hujan")
    if model is not None and scaler is not None:
        col1, col2, col3 = st.columns(3)
        suhu_min = col1.number_input("Suhu Minimum (°C)", value=22.0)
        suhu_max = col2.number_input("Suhu Maksimum (°C)", value=30.0)
        suhu_avg = col3.number_input("Suhu Rata-rata (°C)", value=25.0)
        kelembaban = col1.number_input("Kelembaban (%)", value=75.0)
        angin_max = col2.number_input("Kecepatan Angin Maks (m/s)", value=5.0)
        angin_avg = col3.number_input("Kecepatan Angin Rata-rata (m/s)", value=3.0)
        arah_angin = col1.number_input("Arah Angin (°)", value=250.0)

        if st.button("Prediksi"):
            try:
                input_data = [[suhu_min, suhu_max, suhu_avg, kelembaban, angin_max, angin_avg, arah_angin]]
                input_scaled = scaler.transform(input_data)
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0][1]
                if pred == 1:
                    st.success(f"Prediksi: HUJAN (Probabilitas {prob*100:.1f}%)")
                else:
                    st.info(f"Prediksi: TIDAK HUJAN (Probabilitas {prob*100:.1f}%)")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
    else:
        st.error("Model atau scaler tidak ditemukan!")

# === Footer ===
st.markdown("<div class='footer'>Dashboard Cuaca Bandung — dibuat dengan Streamlit</div>", unsafe_allow_html=True)
