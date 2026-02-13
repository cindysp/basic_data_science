import streamlit as st

def main():
    st.title('Prediksi Gaji Peserta Vokasi')

if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd # Import pandas for data manipulation
# Since df_bersih is not directly available in app.py context,
# we need to get the unique values from df_bersih outside this block
# and pass them as hardcoded lists, or load df_bersih inside app.py.
# For simplicity and given the task context, I will assume unique values are pre-defined or can be loaded.
# For this step, I will hardcode them based on the kernel state if df_bersih is available.

# Assuming df_bersih is available in the environment where this cell is executed
# and its unique values are consistent.

# Unique values for categorical features from df_bersih
pendidikan_options = ['SMA', 'SMK', 'D3', 'S1', 'S2'] # df_bersih['Pendidikan'].unique() results
jurusan_options = ['Administrasi', 'Teknik Las', 'Desain Grafis', 'Teknik Listrik', 'Otomotif'] # df_bersih['Jurusan'].unique() results

def main():
    st.title('Prediksi Gaji Peserta Vokasi')

    st.header('Input Data Peserta')

    user_usia = st.number_input('Usia', min_value=18, max_value=60, value=25, step=1)
    user_durasi_jam = st.number_input('Durasi Pelatihan (Jam)', min_value=20, max_value=100, value=50, step=1)
    user_nilai_ujian = st.number_input('Nilai Ujian', min_value=0.0, max_value=100.0, value=75.0, format="%.1f")

    # Use the extracted options directly here
    user_pendidikan = st.selectbox('Pendidikan Terakhir', options=pendidikan_options)
    user_jurusan = st.selectbox('Jurusan Pelatihan', options=jurusan_options)

    user_jenis_kelamin = st.radio('Jenis Kelamin', options=['Laki-laki', 'Wanita'])
    user_status_bekerja = st.radio('Status Bekerja', options=['Sudah Bekerja', 'Belum Bekerja'])


if __name__ == '__main__':
    main()

# --- 3. Buat Fungsi Pra-pemrosesan untuk Data Baru ---
def preprocess_new_data(data, le_pendidikan, le_jurusan, loaded_scaler, x_train_columns):
    # Konversi data input ke DataFrame
    new_df = pd.DataFrame([data])

    # 3.1. Terapkan Pemetaan Jenis_Kelamin yang sama dengan saat pelatihan
    mapping_gender = {'Pria': 'Laki-laki', 'L': 'Laki-laki', 'Perempuan': 'Wanita', 'P': 'Wanita'}
    new_df['Jenis_Kelamin'] = new_df['Jenis_Kelamin'].replace(mapping_gender)

    # 3.2. Lakukan Label Encoding untuk 'Pendidikan' dan 'Jurusan'
    new_df['Pendidikan_encoded'] = le_pendidikan.transform(new_df['Pendidikan'])
    new_df['Jurusan_encoded'] = le_jurusan.transform(new_df['Jurusan'])

    # 3.3. Lakukan One-Hot Encoding untuk 'Jenis_Kelamin' dan 'Status_Bekerja'
    ohe_gender = pd.get_dummies(new_df['Jenis_Kelamin'], prefix='Jenis_Kelamin')
    ohe_status = pd.get_dummies(new_df['Status_Bekerja'], prefix='Status_Bekerja')

    # Pastikan semua kolom dummy yang diharapkan ada, isi dengan 0 jika tidak ada
    # Ini untuk mencocokkan struktur kolom x_train saat pelatihan
    expected_gender_cols = ['Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita']
    for col in expected_gender_cols:
        if col not in ohe_gender.columns:
            ohe_gender[col] = 0
    ohe_gender = ohe_gender[expected_gender_cols].astype(int) # Pastikan urutan dan tipe data

    expected_status_cols = ['Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']
    for col in expected_status_cols:
        if col not in ohe_status.columns:
            ohe_status[col] = 0
    ohe_status = ohe_status[expected_status_cols].astype(int) # Pastikan urutan dan tipe data

    # 3.4. Gabungkan semua fitur menjadi satu DataFrame
    # Kolom numerik asli
    processed_df = new_df[['Usia', 'Durasi_Jam', 'Nilai_Ujian']].copy()
    # Kolom Label Encoded
    processed_df['Pendidikan'] = new_df['Pendidikan_encoded']
    processed_df['Jurusan'] = new_df['Jurusan_encoded']
    # Kolom One-Hot Encoded
    processed_df = pd.concat([processed_df, ohe_gender, ohe_status], axis=1)

    # Pastikan urutan kolom sesuai dengan data pelatihan (x_train)
    # x_train_columns harus tersedia dari lingkungan pemanggilan
    for col in x_train_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    processed_df = processed_df[x_train_columns]

    # 3.5. Lakukan Scaling pada semua fitur menggunakan loaded_scaler
    scaled_features = loaded_scaler.transform(processed_df)
    scaled_df = pd.DataFrame(scaled_features, columns=processed_df.columns)

    return scaled_df

# Memindahkan loading model/scaler dan LabelEncoder inisialisasi ke bagian atas main() function
# agar hanya sekali jalan saat app dimulai
def load_resources():
    global loaded_model, loaded_scaler, le_pendidikan, le_jurusan, x_train_columns
    # --- 1. Muat Model dan Scaler yang Tersimpan ---
    with open('gradient_boosting_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        loaded_scaler = pickle.load(file)

    # --- 2. Inisialisasi Ulang LabelEncoder dengan Data Pelatihan ---
    # LabelEncoder tidak disimpan secara langsung, jadi kita inisialisasi ulang
    # dengan kategori unik dari data 'df_bersih' yang masih ada di memori.
    # Untuk keperluan Streamlit, kita perlu memuat atau mendefinisikan ulang kategori unik jika df_bersih tidak tersedia.
    # Untuk saat ini, kita akan menggunakan nilai unik dari df_bersih yang ada di environment notebook.
    # Dalam aplikasi Streamlit mandiri, Anda mungkin perlu menyimpan df_bersih atau daftar unik ini.
    
    # Untuk sementara, gunakan data dari kernel:
    # pendidikan_unique = df_bersih['Pendidikan'].unique()
    # jurusan_unique = df_bersih['Jurusan'].unique()

    # Hardcoding unique values based on previous cell output to ensure app.py runs independently
    # without df_bersih being in its direct scope at runtime.
    pendidikan_unique = ['SMA', 'SMK', 'D3', 'S1', 'S2']
    jurusan_unique = ['Administrasi', 'Teknik Las', 'Desain Grafis', 'Teknik Listrik', 'Otomotif']

    le_pendidikan = LabelEncoder()
    le_pendidikan.fit(pendidikan_unique)

    le_jurusan = LabelEncoder()
    le_jurusan.fit(jurusan_unique)

    # Menyimpan kolom x_train untuk memastikan urutan yang benar
    # Ini harus diambil dari `x_train` yang diinisialisasi selama training
    # Hardcoding x_train_columns based on previous cell output to ensure app.py runs independently
    x_train_columns = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan', 
                       'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita', 
                       'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja']


def main():
    load_resources()
    st.title('Prediksi Gaji Peserta Vokasi')

    st.header('Input Data Peserta')

    user_usia = st.number_input('Usia', min_value=18, max_value=60, value=25, step=1)
    user_durasi_jam = st.number_input('Durasi Pelatihan (Jam)', min_value=20, max_value=100, value=50, step=1)
    user_nilai_ujian = st.number_input('Nilai Ujian', min_value=0.0, max_value=100.0, value=75.0, format="%.1f")

    pendidikan_options = ['SMA', 'SMK', 'D3', 'S1', 'S2']
    jurusan_options = ['Administrasi', 'Teknik Las', 'Desain Grafis', 'Teknik Listrik', 'Otomotif']

    user_pendidikan = st.selectbox('Pendidikan Terakhir', options=pendidikan_options)
    user_jurusan = st.selectbox('Jurusan Pelatihan', options=jurusan_options)

    user_jenis_kelamin = st.radio('Jenis Kelamin', options=['Laki-laki', 'Wanita'])
    user_status_bekerja = st.radio('Status Bekerja', options=['Sudah Bekerja', 'Belum Bekerja'])

    if st.button('Prediksi Gaji'):
        new_data_point = {
            'Usia': user_usia,
            'Durasi_Jam': user_durasi_jam,
            'Nilai_Ujian': user_nilai_ujian,
            'Pendidikan': user_pendidikan,
            'Jurusan': user_jurusan,
            'Jenis_Kelamin': user_jenis_kelamin,
            'Status_Bekerja': user_status_bekerja
        }

        preprocessed_input = preprocess_new_data(new_data_point, le_pendidikan, le_jurusan, loaded_scaler, x_train_columns)
        predicted_gaji = loaded_model.predict(preprocessed_input)
        st.success(f'Prediksi Gaji Pertama: {predicted_gaji[0]:.2f} Juta Rupiah')


if __name__ == '__main__':
    # Declare global variables outside main for access within the app's scope
    loaded_model = None
    loaded_scaler = None
    le_pendidikan = None
    le_jurusan = None
    x_train_columns = None
    main()
