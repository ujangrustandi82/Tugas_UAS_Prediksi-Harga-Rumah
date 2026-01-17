

# Proyek Machine Learning End-to-End
## Prediksi Harga Rumah

### Deskripsi Proyek
Proyek ini bertujuan untuk membangun sistem prediksi harga rumah menggunakan Machine Learning
yang di-deploy sebagai API menggunakan FastAPI dan diakses melalui antarmuka web sederhana.

---

### Dataset
Dataset berisi data properti rumah dengan variabel:
- bedrooms
- bathrooms
- sqft_living
- floors

Target (Y):
- price (harga rumah)

---

### Preprocessing Data
- Menghapus data duplikat
- Menangani missing value
- Standardisasi fitur numerik
- Pembagian data train (80%) dan test (20%)

---

### Algoritma
Algoritma yang digunakan adalah **Random Forest Regression** karena mampu
menangani data non-linear dan memberikan akurasi yang baik.

---

### Evaluasi Model
- R2 Score
- RMSE (Root Mean Squared Error)

---

### Deployment
Model disimpan dalam format `.pkl` dan di-load oleh FastAPI.
Endpoint:
- GET `/`
- POST `/predict`

---

### Frontend
Frontend dibuat menggunakan HTML dan JavaScript
untuk mengirim data ke API dan menampilkan hasil prediksi.

---

### Cara Menjalankan
1. Install library:
pip install fastapi uvicorn pandas scikit-learn numpy

markdown
Copy code

2. Jalankan API:
uvicorn main:app --reload

yaml
Copy code

3. Buka `index.html` di browser

---

### Author
Nama : Ujang Rustandi  
Mata Kuliah : Machine Learning  
Tahun : 2026




