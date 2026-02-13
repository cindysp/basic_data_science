
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title='Prediksi Gaji Vokasi', layout='centered')
st.title('Prediksi Gaji Peserta Pelatihan Vokasi')
st.write('Aplikasi untuk memprediksi gaji awal peserta pelatihan vokasi berdasarkan data yang dimasukkan.')

# --- Fungsi untuk Memuat Model dan Sumber Daya --- #
@st.cache_resource
def load_all_resources():
    try:
        with open('gradient_boosting_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

        with open('scaler.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)

        # Hardcoding unique values based on df_bersih.unique() from notebook state
        pendidikan_unique = ['SMA', 'D3', 'SMK', 'S1'] # Order from df_bersih.unique() in earlier cells
        jurusan_unique = ['Administrasi', 'Desain Grafis', 'Otomotif', 'Teknik Las', 'Teknik Listrik'] # Order from df_bersih.unique() in earlier cells

        le_pendidikan = LabelEncoder()
        le_pendidikan.fit(pendidikan_unique)

        le_jurusan = LabelEncoder()
        le_jurusan.fit(jurusan_unique)

        # Hardcoding x_train_columns based on x_train.head() output in the notebook
        x_train_columns = [
            'Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
            'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
            'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
        ]
        return loaded_model, loaded_scaler, le_pendidikan, le_jurusan, x_train_columns
    except FileNotFoundError as e:
        st.error(f"Error: File model atau scaler tidak ditemukan. Pastikan file-file .pkl berada di direktori yang sama dengan app.py. Detail: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat sumber daya. Detail: {e}")
        st.stop()

# Memuat semua sumber daya
loaded_model, loaded_scaler, le_pendidikan, le_jurusan, x_train_columns = load_all_resources()

# --- Fungsi Pra-pemrosesan untuk Data Baru --- #
def preprocess_new_data(data):
    new_df = pd.DataFrame([data])

    # Terapkan Pemetaan Jenis_Kelamin
    mapping_gender = {'Pria': 'Laki-laki', 'L': 'Laki-laki', 'Perempuan': 'Wanita', 'P': 'Wanita'}
    new_df['Jenis_Kelamin'] = new_df['Jenis_Kelamin'].replace(mapping_gender)

    # Lakukan Label Encoding untuk 'Pendidikan' dan 'Jurusan'
    new_df['Pendidikan_encoded'] = le_pendidikan.transform(new_df['Pendidikan'])
    new_df['Jurusan_encoded'] = le_jurusan.transform(new_df['Jurusan'])

    # Lakukan One-Hot Encoding untuk 'Jenis_Kelamin' dan 'Status_Bekerja'
    ohe_gender = pd.get_dummies(new_df['Jenis_Kelamin'], prefix='Jenis_Kelamin')
    ohe_status = pd.get_dummies(new_df['Status_Bekerja'], prefix='Status_Bekerja')

    # Pastikan semua kolom dummy yang diharapkan ada, isi dengan 0 jika tidak ada
    expected_gender_cols = ['Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita']
    for col in expected_gender_cols:
        if col not in ohe_gender.columns:
            ohe_gender[col] = 0
    ohe_gender = ohe_gender[expected_gender_cols].astype(int)

    expected_status_cols = ['Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']
    for col in expected_status_cols:
        if col not in ohe_status.columns:
            ohe_status[col] = 0
    ohe_status = ohe_status[expected_status_cols].astype(int)

    # Gabungkan semua fitur menjadi satu DataFrame
    processed_df = new_df[['Usia', 'Durasi_Jam', 'Nilai_Ujian']].copy()
    processed_df['Pendidikan'] = new_df['Pendidikan_encoded']
    processed_df['Jurusan'] = new_df['Jurusan_encoded']
    processed_df = pd.concat([processed_df, ohe_gender, ohe_status], axis=1)

    # Pastikan urutan kolom sesuai dengan data pelatihan (x_train_columns)
    for col in x_train_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    processed_df = processed_df[x_train_columns]

    # Lakukan Scaling pada semua fitur
    scaled_features = loaded_scaler.transform(processed_df)
    scaled_df = pd.DataFrame(scaled_features, columns=processed_df.columns)

    return scaled_df

# --- Desain UI Streamlit --- #
st.header('Input Data Peserta Baru')

col1, col2 = st.columns(2)

pendidikan_options = le_pendidikan.classes_.tolist()
jurusan_options = le_jurusan.classes_.tolist()
jenis_kelamin_options = ['Laki-laki', 'Wanita']
status_bekerja_options = ['Belum Bekerja', 'Sudah Bekerja']

with col1:
    usia = st.number_input('Usia', min_value=18, max_value=60, value=25, step=1)
    durasi_jam = st.number_input('Durasi Pelatihan (Jam)', min_value=20, max_value=100, value=50, step=1)
    nilai_ujian = st.number_input('Nilai Ujian', min_value=0.0, max_value=100.0, value=75.0, step=0.1, format="%.1f")

with col2:
    pendidikan = st.selectbox('Pendidikan', options=pendidikan_options)
    jurusan = st.selectbox('Jurusan', options=jurusan_options)
    jenis_kelamin = st.selectbox('Jenis Kelamin', options=jenis_kelamin_options)
    status_bekerja = st.selectbox('Status Bekerja', options=status_bekerja_options)


if st.button('Prediksi Gaji Awal'):
    new_data = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }

    # Pra-pemrosesan dan prediksi
    processed_data = preprocess_new_data(new_data)
    predicted_gaji = loaded_model.predict(processed_data)[0]

    st.subheader('Hasil Prediksi')
    st.success(f'Prediksi Gaji Awal: **Rp {predicted_gaji * 1_000_000:,.2f}**')

st.markdown('---')
st.markdown('''
### Cara Menjalankan Aplikasi ini:
1. Pastikan Anda telah menyimpan `gradient_boosting_model.pkl` dan `scaler.pkl` di direktori yang sama dengan `app.py`.
2. Buka terminal atau command prompt.
3. Navigasi ke direktori tempat `app.py` disimpan.
4. Jalankan perintah: `streamlit run app.py`
5. Aplikasi akan terbuka di browser web Anda.
''')


if __name__ == '__main__':
    # main function content is now directly in the script, as Streamlit runs top-to-bottom.
    # The if __name__ == '__main__': block is often not strictly needed for basic Streamlit apps,
    # but it's good practice for modules.
    pass
