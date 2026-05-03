# cnn_restnet18
## Comparative Study of Regularized CNN and ResNet‑18 for Indonesian Rupiah Banknote Classification

### 📌 Deskripsi Proyek
Proyek ini merupakan eksperimen dan studi komparatif (perbandingan) antara dua arsitektur *Deep Learning* untuk tugas pengenalan dan klasifikasi uang kertas Rupiah Indonesia. Model yang dibandingkan adalah:

1. **Regularized CNN (Dibuat dari Nol / Scratch)**: Arsitektur *Convolutional Neural Network* (CNN) ringan yang dibangun secara mandiri. Model ini dilengkapi dengan teknik regularisasi seperti *Batch Normalization* dan *Dropout* untuk mencegah terjadinya *overfitting* pada dataset berukuran kecil/menengah.
2. **ResNet-18 (Transfer Learning / Fine-Tuned)**: Model *state-of-the-art* turunan ImageNet yang dioptimalkan ulang (*fine-tuning*) menggunakan teknik *differential learning rate*. Lapisan konvolusi awal mempertahankan fitur visual dasar, sementara lapisan akhir (klasifikator) dilatih khusus untuk membedakan nominal Rupiah.

Tujuan utama proyek ini adalah membandingkan kecepatan pelatihan, konvergensi, serta akurasi dari kedua pendekatan tersebut, dan membuktikan keandalan ResNet-18 yang terbukti mampu mencapai tingkat akurasi mendekati 100% pada eksperimen ini.

### 💰 Kelas Target (Nominal Uang)
Sistem telah dilatih untuk mendeteksi 7 jenis pecahan uang kertas Rupiah:
- Rp 1.000
- Rp 2.000
- Rp 5.000
- Rp 10.000
- Rp 20.000
- Rp 50.000
- Rp 100.000

### 🚀 Berkas Utama
- `comparative_cnn_resnet18.ipynb`: Notebook Jupyter yang memuat metodologi riset secara utuh; meliputi pembagian dataset (train/val/test), augmentasi data, implementasi PyTorch dari kedua model, proses *training universal* dengan *early stopping*, hingga visualisasi metrik performa (*confusion matrix* dan grafik perbandingan kurva latihan).
- `webcam_test.py`: Skrip pengujian waktu-nyata (*real-time*) yang mengintegrasikan model final ResNet-18 dengan OpenCV. Digunakan untuk membuktikan kinerja model secara langsung menggunakan kamera (webcam) laptop Anda.
- `hasil_komparasi/`: Direktori tempat penyimpanan bobot model terbaik (format `.pth`) serta laporan visualisasi grafik hasil pelatihan.

### 💻 Cara Menggunakan Pengujian Webcam
Untuk menguji coba model langsung menggunakan kamera laptop Anda, ikuti langkah berikut:

1. Buka Terminal / Anaconda Prompt.
2. Aktifkan *environment* conda Anda yang memiliki dependensi PyTorch dan OpenCV:
   ```bash
   conda activate env_deeplearning
   ```
3. Eksekusi skrip Python berikut:
   ```bash
   python webcam_test.py
   ```
4. Arahkan lembaran uang Rupiah asli Anda ke depan kamera. Prediksi akan muncul di layar beserta persentase akurasinya (kepercayaan/confidence).
5. Tekan tombol **`q`** pada *keyboard* Anda untuk keluar dari jendela kamera.

### 📚 Teknologi / Pustaka yang Digunakan
- **Python 3.x**
- **PyTorch & Torchvision**: Framework *deep learning* untuk konstruksi, pelatihan model, dan augmentasi data.
- **OpenCV (`cv2`)**: Untuk penangkapan video real-time.
- **Scikit-Learn**: Untuk ekstraksi metrik perbandingan klasifikasi.
- **Matplotlib & Seaborn**: Untuk visualisasi grafis laporan model.
