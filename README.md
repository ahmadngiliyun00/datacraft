
![Logo DataCraft](https://github.com/ahmadngiliyun00/datacraft/blob/main/static/img/Logo-DataCraft-Horizontal.png?raw=true)

# DataCraft - Sentiment Analysis Web App 🚀

DataCraft adalah aplikasi berbasis web yang digunakan untuk melakukan analisis sentimen menggunakan Naive Bayes dan Support Vector Machine (SVM). Aplikasi ini mendukung pemrosesan data otomatis, pembuatan model, serta visualisasi hasil dengan berbagai metrik evaluasi.

![Tentang DataCraft](https://github.com/ahmadngiliyun00/datacraft/blob/main/static/img/Tentang-Aplikasi.png?raw=true)

## ✨ Fitur Utama

 - **Eksplorasi Data**: Mengunggah dataset dari file `CSV`/`XLSX` dan
   melakukan eksplorasi awal.
   
-  **Pra-Pemrosesan Data**: Membersihkan teks, stemming dengan `Sastrawi`,
   serta menghapus stopwords.
   
-  **Pemodelan Data**: Melatih model `Naive Bayes` dan `SVM`, serta
   mengevaluasi performanya.
   
-  **Interpretasi Hasil**: Menyediakan grafik distribusi sentimen, word
   cloud, serta metrik evaluasi model.
---
## 🛠 Instalasi & Menjalankan Aplikasi

1. **Clone Repository**

		git clone https://github.com/username/repository.git
		cd repository

2. **Buat Virtual Environment** (Opsional, tapi Direkomendasikan)

		python -m venv venv
		source venv/bin/activate  # Untuk Linux/macOS
		venv\Scripts\activate     # Untuk Windows

3. **Instal Dependensi**

	Pastikan kamu menggunakan `Python 3.8` atau lebih baru, lalu jalankan:

		pip install -r requirements.txt

	Jika `requirements.txt` belum ada, kamu bisa membuatnya dengan:

		pip freeze > requirements.txt

4. **Jalankan Aplikasi**

		python app.py

	Kemudian buka http://127.0.0.1:5000/ di browser.

---
## 📚 Dependensi yang Dibutuhkan

Aplikasi ini membutuhkan library berikut:

- Flask → Framework web untuk backend

- Pandas → Manipulasi data CSV/XLSX

- Matplotlib & Seaborn → Visualisasi data (grafik distribusi sentimen, confusion matrix)

- NLTK & Sastrawi → Pemrosesan teks (stopwords & stemming bahasa Indonesia)

- WordCloud → Pembuatan word cloud dari sentimen

- Scikit-Learn → Model machine learning (Naive Bayes & SVM)

---
## 📌 Instalasi Manual

1. Buat dan Aktifkan Virtual Environment (Opsional, tapi Direkomendasikan)

	Sebelum menginstal dependensi, lebih baik menggunakan virtual environment agar lingkungan kerja tetap bersih.

		# Buat virtual environment (hanya pertama kali)
		python -m venv venv

		# Aktifkan virtual environment
		# Windows
		venv\Scripts\activate

		# macOS/Linux
		source venv/bin/activate

2. Instal Semua Dependensi dari requirements.txt

	Jalankan perintah berikut untuk menginstal semua dependensi yang diperlukan:

		pip install -r requirements.txt

3. Jalankan Aplikasi

	Setelah instalasi selesai, jalankan aplikasi dengan perintah berikut:

		python app.py

	Akses aplikasi di browser melalui http://127.0.0.1:5000/.


Atau Instalasi tanpa menggunakan `requirements.txt`:

	pip install Flask pandas matplotlib seaborn nltk wordcloud scikit-learn Sastrawi

---
## 📂 Struktur Direktori

	📆 DataCraft
	 └─├📂 data
	   └─├📂 uploaded      # Folder untuk menyimpan dataset yang diunggah
	     ├📂 processed     # Folder untuk menyimpan dataset yang telah diproses
	     ├📂 modeled       # Folder untuk menyimpan model yang telah dilatih
	 └─├📂 static
	      └📂 img           # Folder untuk menyimpan visualisasi (grafik, word cloud)
	 └📂 templates       # Folder untuk file HTML
	 └📄 app.py          # Script utama untuk menjalankan aplikasi Flask
	 └📄 requirements.txt # Dependensi Python
	 └📄 README.md       # Dokumentasi proyek ini

---
## 📧 Kontribusi & Dukungan

Jika kamu menemukan bug atau memiliki ide fitur baru, silakan buat issue atau pull request di GitHub.

Happy coding! 🚀🔥