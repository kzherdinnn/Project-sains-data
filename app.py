import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime
import numpy as np # Import numpy for array handling
import matplotlib.pyplot as plt # Import matplotlib for seaborn plots
import seaborn as sns # Import seaborn for statistical plots

# === Konfigurasi Dashboard ===
st.set_page_config(
    page_title="Dashboard Cuaca Bandung",
    layout="wide",
    page_icon="🌤️"
)

# === SIDEBAR: Tema ===
st.sidebar.title("🎨 Pengaturan Tampilan")
theme_mode = st.sidebar.radio("Mode Tampilan:", ["Terang ☀️", "Gelap 🌙"])

# --- CSS Berdasarkan Mode Tema ---
if theme_mode == "Terang ☀️":
    background_color = "linear-gradient(135deg, #E0F7FA 0%, #80DEEA 100%)"
    text_color = "#004D40"
    box_color = "white"
else:
    background_color = "linear-gradient(135deg, #263238 0%, #37474F 100%)"
    text_color = "#ECEFF1"
    box_color = "#455A64"

st.markdown(f"""
    <style>
        .stApp {{
            background: {background_color};
        }}
        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: {text_color} !important;
            font-family: 'Segoe UI', sans-serif;
        }}
        .metric-box {{
            background: {box_color};
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0px 3px 10px rgba(0,0,0,0.15);
            text-align: center;
        }}
        .footer {{
            text-align: center;
            padding: 10px;
            font-size: 0.85em;
            color: {text_color};
        }}
    </style>
    """, unsafe_allow_html=True)

# === Load Dataset ===
@st.cache_data
def load_data():
    # Load the feature engineered data which is used for EDA and can be basis for dashboard display
    df = pd.read_csv("cuaca_bandung_mapped_feature_engineered.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    # Ensure 'Kategori Hujan' and 'Musim' are categorical with correct categories
    # based on the feature engineering step, if they are not already
    if 'Kategori Hujan' in df.columns:
        # Get categories from the label encoder used in modeling
        try:
            le_label_loaded = joblib.load("label_encoder_balanced.pkl")
            df['Kategori Hujan'] = pd.Categorical(df['Kategori Hujan'], categories=le_label_loaded.classes_, ordered=False)
        except FileNotFoundError:
             st.warning("Label encoder for target not found, 'Kategori Hujan' categories might be incomplete.")
             # Fallback: use unique values from data if encoder not found
             df['Kategori Hujan'] = pd.Categorical(df['Kategori Hujan'], categories=df['Kategori Hujan'].unique(), ordered=False)

    if 'Musim' in df.columns:
         # Get categories from the musim encoder if available, otherwise from data
         try:
             le_musim_loaded = joblib.load("musim_encoder.pkl")
             # Note: The encoder stores encoded values, not the original string categories.
             # We need the original string categories for the dashboard filter.
             # Let's rely on the unique values in the dataframe for the dashboard filter.
             df['Musim'] = pd.Categorical(df['Musim'], categories=df['Musim'].unique().tolist(), ordered=False)
         except FileNotFoundError:
              st.warning("Musim encoder not found, 'Musim' categories might be incomplete.")
              df['Musim'] = pd.Categorical(df['Musim'], categories=df['Musim'].unique().tolist(), ordered=False)

    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File dataset (cuaca_bandung_mapped_feature_engineered.csv) tidak ditemukan!")
    st.stop()

# === Load Model, Scaler, dan Encoder ===
@st.cache_resource
def load_model_artifacts():
    try:
        # Load the correct model and scaler files
        model = joblib.load("model_rf_kategori_balanced.pkl")
        scaler = joblib.load("scaler_rf_kategori_balanced.pkl")
        le_label = joblib.load("label_encoder_balanced.pkl") # Load label encoder for target
        # Load musim encoder if it was used and saved
        le_musim = None
        try:
             le_musim = joblib.load("musim_encoder.pkl")
        except FileNotFoundError:
             st.warning("Musim encoder (musim_encoder.pkl) tidak ditemukan. Prediksi mungkin tidak akurat jika 'Musim' digunakan sebagai fitur kategorikal.")

        return model, scaler, le_label, le_musim
    except FileNotFoundError as e:
        st.error(f"File model atau scaler tidak ditemukan: {e}")
        return None, None, None, None

model, scaler, le_label, le_musim = load_model_artifacts()


# === SIDEBAR FILTER ===
st.sidebar.title("🔍 Filter Data Cuaca")

min_date = df["Tanggal"].min().date()
max_date = df["Tanggal"].max().date()
date_range = st.sidebar.date_input("Pilih rentang tanggal:", [min_date, max_date])

# Use categories from the loaded dataframe for filter options
kategori_options = ['Semua'] + (list(df['Kategori Hujan'].cat.categories) if 'Kategori Hujan' in df.columns else [])
kategori_hujan = st.sidebar.selectbox("Filter Kategori Hujan:", kategori_options)

musim_options = ['Semua'] + (df['Musim'].unique().tolist() if 'Musim' in df.columns else [])
musim_filter = st.sidebar.selectbox("Filter Musim:", musim_options)

# === Terapkan Filter ===
df_filtered = df.copy()
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_filtered[(df_filtered["Tanggal"].dt.date >= start_date) &
                             (df_filtered["Tanggal"].dt.date <= end_date)]

if kategori_hujan != 'Semua' and 'Kategori Hujan' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Kategori Hujan'] == kategori_hujan]
if musim_filter != 'Semua' and 'Musim' in df_filtered.columns:
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
        if 'Suhu Rata-rata' in df_filtered.columns:
            col1.markdown(f"<div class='metric-box'><h4>Suhu Rata-rata</h4><h2>{df_filtered['Suhu Rata-rata'].mean():.1f} °C</h2></div>", unsafe_allow_html=True)
        if 'Suhu Maksimum' in df_filtered.columns:
            col2.markdown(f"<div class='metric-box'><h4>Suhu Maksimum</h4><h2>{df_filtered['Suhu Maksimum'].mean():.1f} °C</h2></div>", unsafe_allow_html=True)
        if 'Suhu Minimum' in df_filtered.columns:
            col3.markdown(f"<div class='metric-box'><h4>Suhu Minimum</h4><h2>{df_filtered['Suhu Minimum'].mean():.1f} °C</h2></div>", unsafe_allow_html=True)
        if 'Curah Hujan' in df_filtered.columns:
            col4.markdown(f"<div class='metric-box'><h4>Curah Hujan</h4><h2>{df_filtered['Curah Hujan'].mean():.1f} mm</h2></div>", unsafe_allow_html=True)

        st.markdown("Contoh Data")
        st.dataframe(df_filtered, use_container_width=True, height=400)
    else:
        st.warning("Tidak ada data untuk filter yang dipilih.")

# --- TAB 2: Visualisasi ---
with tab2:
    if not df_filtered.empty:
        st.subheader("Tren Suhu & Curah Hujan")
        # Check if necessary columns exist before plotting
        if 'Tanggal' in df_filtered.columns and 'Suhu Rata-rata' in df_filtered.columns and 'Curah Hujan' in df_filtered.columns:
             fig_trend = px.line(df_filtered, x='Tanggal', y=['Suhu Rata-rata', 'Curah Hujan'],
                                 labels={'value':'Nilai','variable':'Parameter'}, title="Tren Cuaca Bandung")
             fig_trend.update_layout(template='plotly_dark' if theme_mode == "Gelap 🌙" else 'plotly_white')
             st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("Kolom yang diperlukan untuk plot tren tidak lengkap.")

        st.subheader("Distribusi Variabel Numerik")
        numeric_cols_for_dist = [col for col in df_filtered.select_dtypes(include=np.number).columns.tolist() if col not in ['Bulan', 'Tahun']]
        if numeric_cols_for_dist:
            n_cols = 3 # Number of columns in the subplot grid
            n_rows = (len(numeric_cols_for_dist) + n_cols - 1) // n_cols
            fig_dist, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
            axes = axes.flatten() # Flatten the array for easy iteration

            for i, col in enumerate(numeric_cols_for_dist):
                sns.histplot(df_filtered[col], kde=True, bins=20, ax=axes[i])
                axes[i].set_title(f'Distribusi {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Frekuensi")

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                fig_dist.delaxes(axes[j])

            plt.tight_layout()
            st.pyplot(fig_dist) # Use st.pyplot() to display matplotlib figures
        else:
            st.warning("Tidak ada kolom numerik yang dipilih untuk analisis distribusi pada filter yang dipilih.")


        st.subheader("Korelasi Variabel Cuaca")
        numeric_cols_corr = [col for col in ['Suhu Maksimum', 'Suhu Minimum', 'Suhu Rata-rata', 'Curah Hujan', 'Kelembaban', 'Kecepatan Angin_Max', 'Kecepatan Angin_Avg', 'SS', 'Arah_Angin_deg', 'Suhu_Rata_Calc', 'Suhu_7d_mean', 'Hujan_7d_sum', 'Bulan'] if col in df_filtered.columns]
        if numeric_cols_corr: # Only plot heatmap if there are columns
            corr = df_filtered[numeric_cols_corr].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Heatmap Korelasi Cuaca", color_continuous_scale='Blues')
            fig_corr.update_layout(template='plotly_dark' if theme_mode == "Gelap 🌙" else 'plotly_white')
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Tidak ada kolom numerik yang tersedia untuk heatmap korelasi pada filter yang dipilih.")

        st.subheader("Hubungan Suhu vs Kelembaban terhadap Hujan")
        if 'Suhu_Rata_Calc' in df_filtered.columns and 'Kelembaban' in df_filtered.columns and 'Is_Rain' in df_filtered.columns:
             fig_scatter_temp_hum = px.scatter(df_filtered, x='Suhu_Rata_Calc', y='Kelembaban', color='Is_Rain',
                                              title="Hubungan Suhu Rata-rata dan Kelembaban terhadap Hujan",
                                              labels={"Suhu_Rata_Calc": "Suhu Rata-rata (°C)", "Kelembaban": "Kelembaban (%)", "Is_Rain": "Hujan?"})
             fig_scatter_temp_hum.update_layout(template='plotly_dark' if theme_mode == "Gelap 🌙" else 'plotly_white')
             st.plotly_chart(fig_scatter_temp_hum, use_container_width=True)
        else:
             st.warning("Kolom yang diperlukan untuk plot scatter Suhu vs Kelembaban tidak lengkap pada filter yang dipilih.")


        st.subheader("Rata-rata Curah Hujan per Bulan")
        if 'Bulan' in df_filtered.columns and 'Curah Hujan' in df_filtered.columns:
             fig_barplot_bulan = px.bar(df_filtered.groupby('Bulan')['Curah Hujan'].mean().reset_index(),
                                        x='Bulan', y='Curah Hujan', title="Rata-rata Curah Hujan per Bulan",
                                        labels={"Bulan": "Bulan", "Curah Hujan": "Curah Hujan Rata-rata (mm)"})
             fig_barplot_bulan.update_layout(xaxis={'tickmode':'array', 'tickvals': list(range(1,13)), 'ticktext': ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des']},
                                             template='plotly_dark' if theme_mode == "Gelap 🌙" else 'plotly_white')
             st.plotly_chart(fig_barplot_bulan, use_container_width=True)
        else:
             st.warning("Kolom 'Bulan' atau 'Curah Hujan' tidak ditemukan untuk plot curah hujan per bulan pada filter yang dipilih.")

        st.subheader("Perbandingan Suhu Rata-rata per Musim")
        if 'Musim' in df_filtered.columns and 'Suhu_Rata_Calc' in df_filtered.columns:
            # Use matplotlib/seaborn for boxplot as plotly express boxplot might be less intuitive for this
            plt.figure(figsize=(8,5))
            sns.boxplot(x='Musim', y='Suhu_Rata_Calc', data=df_filtered, palette='autumn')
            plt.title("Perbandingan Suhu Rata-rata per Musim")
            plt.xlabel("Musim")
            plt.ylabel("Suhu (°C)")
            plt.xticks(rotation=0)
            st.pyplot(plt) # Display using st.pyplot

        else:
             st.warning("Kolom 'Musim' atau 'Suhu_Rata_Calc' tidak ditemukan untuk plot suhu per musim pada filter yang dipilih.")

        st.subheader("Hubungan Suhu Rata-rata vs Curah Hujan")
        if 'Suhu_Rata_Calc' in df_filtered.columns and 'Curah Hujan' in df_filtered.columns:
            fig_scatter_temp_rain = px.scatter(df_filtered, x='Suhu_Rata_Calc', y='Curah Hujan',
                                               title='Hubungan Suhu vs Curah Hujan',
                                               labels={'Suhu_Rata_Calc': 'Suhu Rata-rata (°C)', 'Curah Hujan': 'Curah Hujan (mm)'})
            fig_scatter_temp_rain.update_layout(template='plotly_dark' if theme_mode == "Gelap 🌙" else 'plotly_white')
            st.plotly_chart(fig_scatter_temp_rain, use_container_width=True)
        else:
             st.warning("Kolom 'Suhu_Rata_Calc' atau 'Curah Hujan' tidak ditemukan untuk plot scatter pada filter yang dipilih.")


        st.subheader("Hubungan Intensitas Hujan dengan Kecepatan Angin Rata-rata")
        if 'Kategori Hujan' in df_filtered.columns and 'Kecepatan Angin_Avg' in df_filtered.columns:
            # Use matplotlib/seaborn for boxplot
            plt.figure(figsize=(8,5))
            sns.boxplot(x='Kategori Hujan', y='Kecepatan Angin_Avg', data=df_filtered, palette='viridis')
            plt.title("Hubungan Intensitas Hujan dengan Kecepatan Angin Rata-rata")
            plt.xlabel("Kategori Hujan")
            plt.ylabel("Kecepatan Angin (m/s)")
            plt.xticks(rotation=15)
            st.pyplot(plt) # Display using st.pyplot
        else:
             st.warning("Kolom 'Kategori Hujan' atau 'Kecepatan Angin_Avg' tidak ditemukan untuk plot boxplot pada filter yang dipilih.")


    else:
        st.warning("Tidak ada data untuk filter yang dipilih.")


# --- TAB 3: Prediksi ---
with tab3:
    st.subheader("Prediksi Kategori Hujan Harian")
    if model is not None and scaler is not None and le_label is not None: # Check if encoders are also loaded
        st.write("Masukkan nilai parameter cuaca untuk prediksi:")

        # Define the feature names exactly as used in the model training
        # This list MUST match the order and names in X_train during model training
        feature_names_for_prediction = [
            'Suhu Maksimum', 'Suhu Minimum', 'Suhu Rata-rata',
            'Kelembaban', 'Curah Hujan', 'SS', 'Kecepatan Angin_Max',
            'Arah_Angin_deg', 'Kecepatan Angin_Avg', 'Suhu_Rata_Calc',
            'Suhu_7d_mean', 'Hujan_7d_sum', 'Bulan', 'Musim'
        ]

        # Create input fields for each feature
        input_values = {}
        col1, col2, col3 = st.columns(3)

        # Arrange inputs into columns
        input_values['Suhu Maksimum'] = col1.number_input("Suhu Maksimum (°C)", value=30.0, format="%.1f")
        input_values['Suhu Minimum'] = col2.number_input("Suhu Minimum (°C)", value=22.0, format="%.1f")
        input_values['Suhu Rata-rata'] = col3.number_input("Suhu Rata-rata (°C)", value=25.0, format="%.1f")

        input_values['Kelembaban'] = col1.number_input("Kelembaban (%)", value=75.0, format="%.1f")
        input_values['Curah Hujan'] = col2.number_input("Curah Hujan (mm)", value=0.0, format="%.1f") # Input for current day rainfall
        input_values['SS'] = col3.number_input("SS (jam)", value=5.0, format="%.1f")

        input_values['Kecepatan Angin_Max'] = col1.number_input("Kecepatan Angin Maks (m/s)", value=5.0, format="%.1f")
        input_values['Arah_Angin_deg'] = col2.number_input("Arah Angin (°)", value=250.0, format="%.1f")
        input_values['Kecepatan Angin_Avg'] = col3.number_input("Kecepatan Angin Rata-rata (m/s)", value=3.0, format="%.1f")

        # These require some thought for real-time input - maybe average of last 7 days from available data?
        # For simplicity in a demo, let's allow manual input or default to a value
        # Recalculate Suhu_Rata_Calc based on user input for max/min temp
        input_values['Suhu_Rata_Calc'] = (input_values['Suhu Maksimum'] + input_values['Suhu Minimum']) / 2.0
        input_values['Suhu_7d_mean'] = col2.number_input("Rata-rata Suhu 7 Hari (°C)", value=float(df['Suhu_7d_mean'].mean()), format="%.1f") # Placeholder/Example with dataset mean
        input_values['Hujan_7d_sum'] = col3.number_input("Total Curah Hujan 7 Hari (mm)", value=float(df['Hujan_7d_sum'].mean()), format="%.1f") # Placeholder/Example with dataset mean

        # Month and Musim can be derived from a date input, but for simplicity, let's input directly or derive from current date
        # Using current date for simplicity, or could add a date input
        current_month = datetime.now().month
        input_values['Bulan'] = col1.number_input("Bulan (1-12)", value=current_month, min_value=1, max_value=12, step=1)

        # Derive Musim from Bulan input
        def get_musim(bulan):
             if bulan in [12, 1, 2, 3, 4]:
                 return 'Musim Hujan'
             elif bulan in [5, 6, 7, 8, 9, 10]:
                 return 'Musim Kemarau'
             else:
                 return 'Peralihan' # Should not happen with month 1-12

        musim_str = get_musim(input_values['Bulan'])
        st.write(f"Musim Terdeteksi: **{musim_str}**")

        # Encode Musim if the encoder is available
        musim_encoded_value = None
        if le_musim is not None:
            try:
                # le_musim expects a list-like input
                musim_encoded_value = le_musim.transform([musim_str])[0]
                input_values['Musim'] = musim_encoded_value # Add encoded value to input_values
            except ValueError as e:
                 st.error(f"Gagal meng-encode Musim '{musim_str}'. Encoder mungkin tidak mengenali kategori ini. Error: {e}")
                 # Handle this error - maybe stop prediction or use a default value
                 musim_encoded_value = None # Indicate encoding failed

        # Ensure 'Musim' key is present in input_values even if encoding failed,
        # using the string value for now, although model expects numeric.
        # If encoding failed, prediction will likely fail unless Musim is dropped or handled.
        # Let's add a check before prediction.
        if le_musim is None and 'Musim' in feature_names_for_prediction:
             st.warning("Musim encoder tidak dimuat, kolom 'Musim' mungkin tidak diproses dengan benar untuk prediksi.")
             # If 'Musim' is in feature_names_for_prediction, it needs to be a number.
             # If le_musim failed, input_values['Musim'] is still the string.
             # This will cause scaler.transform to fail.
             # A robust solution would require either the encoder or dropping the feature.
             # For this update, we assume le_musim is available and works or Musim is not critical.
             # Let's add a check before creating the DataFrame.


        if st.button("Prediksi Cuaca Harian"):
            if scaler is not None and model is not None and le_label is not None:
                try:
                    # Create DataFrame for prediction
                    # Ensure the order of columns matches feature_names_for_prediction
                    input_df_pred = pd.DataFrame([input_values], columns=feature_names_for_prediction)

                    # --- Preprocessing steps similar to training ---
                    # Handle Musim Encoding if successful (already done above)
                    # If le_musim was None or encoding failed, and Musim is a feature,
                    # input_df_pred['Musim'] might be a string. Check type.
                    if 'Musim' in input_df_pred.columns and input_df_pred['Musim'].dtype == 'object':
                         st.error("Kolom 'Musim' tidak dalam format numerik setelah preprocessing. Prediksi dibatalkan.")
                         st.stop()


                    # Identify numeric columns for scaling in the input DataFrame
                    numeric_cols_input = input_df_pred.select_dtypes(include=np.number).columns.tolist()
                    # Ensure all numeric features expected by the scaler are present
                    expected_numeric_cols_by_scaler = scaler.feature_names_in_ # Use feature_names_in_ from scaler

                    if not set(expected_numeric_cols_by_scaler).issubset(set(input_df_pred.columns)):
                         missing_for_scaling = set(expected_numeric_cols_by_scaler) - set(input_df_pred.columns)
                         st.error(f"Kolom numerik yang diperlukan untuk scaling tidak lengkap: {missing_for_scaling}. Prediksi dibatalkan.")
                         st.stop()

                    # Reorder input_df_pred columns to match scaler's expected order
                    input_df_pred_ordered = input_df_pred[expected_numeric_cols_by_scaler]

                    # Apply scaling
                    input_scaled_values = scaler.transform(input_df_pred_ordered)
                    input_scaled_df = pd.DataFrame(input_scaled_values, columns=expected_numeric_cols_by_scaler)

                    # --- Prediction ---
                    # The model expects features in the same order as during training.
                    # The columns of input_scaled_df should match the order of X_train.columns.
                    # The order of expected_numeric_cols_by_scaler from the scaler should match X_train.columns.
                    # Let's double check the order.
                    # Based on cell cIMEcnqulvyq, X_train columns are:
                    # ['Suhu Maksimum', 'Suhu Minimum', 'Suhu Rata-rata', 'Kelembaban', 'Curah Hujan', 'SS', 'Kecepatan Angin_Max', 'Arah_Angin_deg', 'Kecepatan Angin_Avg', 'Suhu_Rata_Calc', 'Suhu_7d_mean', 'Hujan_7d_sum', 'Bulan', 'Musim']
                    # This should match the order of `feature_names_for_prediction`.

                    # If the order is confirmed to match, proceed.
                    # If not, we might need to reindex input_scaled_df.
                    # For safety, let's ensure column order matches model's expected order.
                    # model.feature_names_in_ holds the expected feature names and their order
                    if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
                         if list(input_scaled_df.columns) != list(model.feature_names_in_):
                              st.warning("Urutan kolom data input setelah scaling tidak cocok dengan urutan kolom yang diharapkan model. Mencoba menyesuaikan urutan.")
                              try:
                                   input_scaled_df = input_scaled_df[list(model.feature_names_in_)]
                              except KeyError as ke:
                                   st.error(f"Gagal menyesuaikan urutan kolom untuk model: {ke}. Pastikan semua fitur model ada di data input.")
                                   st.stop()
                              except Exception as e:
                                   st.error(f"Terjadi error tak terduga saat menyesuaikan urutan kolom: {e}")
                                   st.stop()

                         else:
                             st.info("Urutan kolom data input cocok dengan yang diharapkan model.")
                    else:
                         st.warning("Model does not have 'feature_names_in_'. Assuming input column order is correct.")
                         # If model doesn't have feature_names_in_, rely on the order of `feature_names_for_prediction`


                    pred_encoded = model.predict(input_scaled_df)[0]
                    proba = model.predict_proba(input_scaled_df)[0]

                    # --- Interpret Results ---
                    # Ensure le_label is loaded
                    if le_label is not None:
                        label_pred = le_label.inverse_transform([pred_encoded])[0]

                        st.markdown(f"#### ✨ Hasil Prediksi:")
                        st.info(f"Prediksi Kategori Hujan: **{label_pred}**")

                        # Display probabilities for all categories
                        st.markdown("##### Probabilitas per Kategori:")
                        prob_data = {'Kategori Hujan': le_label.classes_, 'Probabilitas': proba}
                        prob_df = pd.DataFrame(prob_data).sort_values('Probabilitas', ascending=False)
                        # Format probability as percentage
                        prob_df['Probabilitas'] = prob_df['Probabilitas'].apply(lambda x: f"{x*100:.2f}%")
                        st.dataframe(prob_df, use_container_width=True)

                        # Optional: Binary prediction (Hujan/Tidak Hujan)
                        st.markdown("##### Prediksi Biner (Hujan / Tidak Hujan):")
                        # Sum probabilities for 'Hujan Ringan', 'Hujan Sedang', 'Hujan Lebat'
                        hujan_categories = ['Hujan Ringan', 'Hujan Sedang', 'Hujan Lebat']
                        prob_hujan_total = sum(proba[le_label.transform([cat])[0]] for cat in hujan_categories if cat in le_label.classes_)
                        prob_tidak_hujan = proba[le_label.transform(['Tidak Hujan'])[0]] if 'Tidak Hujan' in le_label.classes_ else 0

                        pred_biner = "Hujan" if prob_hujan_total > prob_tidak_hujan else "Tidak Hujan"

                        st.success(f"Prediksi Biner: **{pred_biner}**")
                        st.info(f"Probabilitas Hujan Total: {prob_hujan_total*100:.2f}%")
                        st.info(f"Probabilitas Tidak Hujan: {prob_tidak_hujan*100:.2f}%")

                    else:
                        st.error("Label Encoder tidak dimuat. Tidak dapat menginterpretasikan hasil prediksi.")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
                    st.warning("Pastikan nilai input sesuai dan file model/scaler/encoder berhasil dimuat.")
            else:
                st.error("Model, Scaler, atau Encoder tidak ditemukan. Tidak dapat melakukan prediksi.")


    else:
        st.warning("Model, Scaler, atau Encoder belum berhasil dimuat.")

# === Footer ===
st.markdown("<div class='footer'>Dashboard Cuaca Bandung — dibuat dengan Streamlit</div>", unsafe_allow_html=True)
