import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime

# === Konfigurasi Dashboard ===
st.set_page_config(
    page_title="🌦️ Dashboard Cuaca Bandung",
    layout="wide",
    page_icon="🌤️"
)

# 🌈 --- Styling Custom CSS ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #E0F7FA 0%, #80DEEA 100%);
        }
        h1, h2, h3, h4 {
            color: #004D40;
            font-family: 'Segoe UI', sans-serif;
        }
        .metric-box {
            background: white;
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .footer {
            text-align: center;
            padding: 10px;
            font-size: 0.85em;
            color: #004D40;
        }
    </style>
""", unsafe_allow_html=True)

# === Load Dataset ===
@st.cache_data
def load_data():
    # Mengganti nama file sesuai dengan hasil preprocessing
    df = pd.read_csv("cuaca_bandung_mapped_feature_engineered.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ File `cuaca_bandung_mapped_feature_engineered.csv` tidak ditemukan. Pastikan file ada di folder yang sama.")
    st.stop()

# === Load Model dan Scaler ===
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_prediksi_hujan.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except:
        # Mengembalikan None jika file model atau scaler tidak ditemukan
        return None, None

model, scaler = load_model()

# === SIDEBAR ===
st.sidebar.title("🔧 Pengaturan Dashboard")
st.sidebar.markdown("Gunakan filter berikut untuk menyesuaikan tampilan data:")

# Filter Tahun
tahun_list = sorted(df['Tahun'].unique())
tahun = st.sidebar.selectbox("Pilih Tahun", tahun_list)

# Filter Bulan
bulan_list = sorted(df['Bulan'].unique())
bulan = st.sidebar.multiselect("Pilih Bulan", bulan_list, default=[])

# Terapkan Filter
df_filtered = df[df['Tahun'] == tahun]
if bulan:
    df_filtered = df_filtered[df_filtered['Bulan'].isin(bulan)]

# === HEADER ===
st.title("🌦️ Dashboard Analitik & Prediksi Cuaca — Bandung")
st.markdown("> Dashboard ini menyajikan **analisis data historis cuaca** dan **prediksi hujan** di Kota Bandung secara interaktif ☔")

# === Tabs Navigasi ===
tab1, tab2, tab3 = st.tabs(["📌 Ringkasan", "📈 Visualisasi", "🤖 Prediksi"])

# === TAB 1: RINGKASAN ===
with tab1:
    st.subheader("📌 Ringkasan Data Cuaca")

    if not df_filtered.empty:
        # Menampilkan Metrik Cuaca Rata-rata
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='metric-box'><h4>🌡️ Suhu Rata-rata</h4><h2>{df_filtered['Suhu Rata-rata'].mean():.1f} °C</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'><h4>🔥 Suhu Maksimum</h4><h2>{df_filtered['Suhu Maksimum'].mean():.1f} °C</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-box'><h4>❄️ Suhu Minimum</h4><h2>{df_filtered['Suhu Minimum'].mean():.1f} °C</h2></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='metric-box'><h4>🌧️ Curah Hujan</h4><h2>{df_filtered['Curah Hujan'].mean():.1f} mm</h2></div>", unsafe_allow_html=True)

        st.markdown("### 📝 Contoh Data")
        st.dataframe(df_filtered.head(10), use_container_width=True)
    else:
        st.warning("⚠️ Tidak ada data untuk filter yang dipilih.")

# === TAB 2: VISUALISASI ===
with tab2:
    st.subheader("📊 Tren Cuaca")

    if not df_filtered.empty:
        # Grafik Tren Suhu & Curah Hujan
        fig_trend = px.line(
            df_filtered,
            x='Tanggal',
            y=['Suhu Rata-rata', 'Curah Hujan'],
            labels={'value': 'Nilai', 'variable': 'Parameter'},
            title="📈 Grafik Suhu & Curah Hujan"
        )
        fig_trend.update_layout(legend_title_text='Parameter', template='plotly_white')
        st.plotly_chart(fig_trend, use_container_width=True)

        # Korelasi Variabel Cuaca Numerik
        st.subheader("🔍 Korelasi Variabel Cuaca")
        numeric_cols = ['Suhu Maksimum', 'Suhu Minimum', 'Suhu Rata-rata',
                        'Curah Hujan', 'Kelembaban', 'Kecepatan Angin_Max',
                        'Kecepatan Angin_Avg']
        corr = df_filtered[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Heatmap Korelasi Cuaca")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("⚠️ Tidak ada data untuk filter yang dipilih.")

# === TAB 3: PREDIKSI ===
with tab3:
    st.subheader("🤖 Prediksi Hujan / Tidak Hujan")

    if model is not None and scaler is not None:
        st.info("Masukkan parameter cuaca untuk melakukan prediksi:")
        col1, col2, col3 = st.columns(3)

        # Input untuk Prediksi
        suhu_min = col1.number_input("Suhu Minimum (°C)", value=22.0)
        suhu_max = col2.number_input("Suhu Maksimum (°C)", value=30.0)
        suhu_avg = col3.number_input("Suhu Rata-rata (°C)", value=25.0)

        kelembaban = col1.number_input("Kelembaban (%)", value=75.0)
        angin_max = col2.number_input("Kecepatan Angin Maks (m/s)", value=5.0)
        angin_avg = col3.number_input("Kecepatan Angin Rata-rata (m/s)", value=3.0)
        arah_angin = col1.number_input("Arah Angin (°)", value=250.0)

        if st.button("🚀 Prediksi"):
            try:
                # Siapkan data input sesuai urutan fitur yang digunakan saat training
                input_data = [[suhu_min, suhu_max, suhu_avg, kelembaban, angin_max, angin_avg, arah_angin]]
                input_scaled = scaler.transform(input_data)
                pred = model.predict(input_scaled)[0]
                # Ambil probabilitas untuk kelas positif (Hujan = 1)
                prob = model.predict_proba(input_scaled)[0][1]

                if pred == 1:
                    st.success(f"🌧 **Prediksi: HUJAN** (Probabilitas {prob*100:.1f}%)")
                else:
                    st.info(f"☀️ **Prediksi: TIDAK HUJAN** (Probabilitas {prob*100:.1f}%)")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
    else:
        st.warning("⚠️ Model prediksi atau scaler belum tersedia. Silakan upload file model (`.pkl`) dan scaler:")

        # Bagian untuk Upload Model
        uploaded_model = st.file_uploader("📤 Upload Model Prediksi (.pkl)", type=["pkl"])
        uploaded_scaler = st.file_uploader("📤 Upload Scaler (.pkl)", type=["pkl"])

        if uploaded_model and uploaded_scaler:
            try:
                with open("model_prediksi_hujan.pkl", "wb") as f:
                    f.write(uploaded_model.getbuffer())
                with open("scaler.pkl", "wb") as f:
                    f.write(uploaded_scaler.getbuffer())
                st.success("✅ Model & Scaler berhasil diupload. Silakan refresh halaman.")
            except Exception as e:
                 st.error(f"Terjadi kesalahan saat menyimpan file: {e}")


# === Footer ===
st.markdown("<div class='footer'>✨ Dashboard Cuaca Bandung — dibuat dengan ❤️ dan Streamlit</div>", unsafe_allow_html=True)
