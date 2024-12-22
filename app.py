import os
import io
import pandas as pd
import re
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import subprocess
from werkzeug.utils import secure_filename
from datetime import datetime
import secrets
from wordcloud import WordCloud
import csv  # Modul csv harus diimpor
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Atur secret key untuk keamanan
app.secret_key = secrets.token_hex(16)  # Gunakan string acak yang kuat dalam produksi

# Konfigurasi direktori untuk menyimpan dataset
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'data', 'uploaded')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

# Fungsi pengecekan ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk menghapus emoji
def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Fungsi untuk menghapus tautan
def remove_links(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

# Fungsi untuk menghapus simbol atau karakter khusus
def remove_special_characters(text):
    return re.sub(r'[^\w\s]', '', text)

# Fungsi untuk menghapus tag HTML dan entitas
def remove_html_tags_and_entities(text):
    if isinstance(text, str):
        text = re.sub(r'<.*?>', '', text)  # Hapus tag HTML
        text = re.sub(r'&[a-z]+;', '', text)  # Hapus entitas HTML
    return text

# Fungsi untuk menghapus underscore
def remove_underscores(text):
    if isinstance(text, str):
        return text.replace('_', '')  # Mengganti underscore dengan string kosong
    return text

# Fungsi utama pembersihan data
def clean_comment(text):
    text = str(text)  # Konversi nilai menjadi string
    text = remove_emoji(text)  # Hapus emoji
    text = remove_links(text)  # Hapus tautan
    text = remove_html_tags_and_entities(text)  # Hapus tag HTML dan entitas
    text = remove_special_characters(text)  # Hapus simbol
    text = remove_underscores(text)  # Hapus underscore
    text = text.strip()  # Hapus spasi berlebih
    return text.lower()  # Ubah ke huruf kecil

# Route untuk halaman index
@app.route('/')
def index():
    return render_template('index.html', title="Dashboard")

@app.route('/about')
def about():
    return render_template('about.html', title="Tentang Aplikasi")

# Route untuk halaman Membuat Dataset
@app.route('/create-dataset', methods=['GET','POST'])
def create_dataset():
    if request.method == 'POST':
        # Ambil input dari form
        keyword = request.form['keyword']
        tweet_limit = request.form['tweet_limit']
        auth_token = request.form['auth_token']

        # Tentukan direktori penyimpanan
        save_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(save_dir, exist_ok=True)

        # Nama file
        safe_keyword = re.sub(r'[^\w\s]', '_', keyword)  # Ganti karakter khusus jadi '_'
        safe_keyword = re.sub(r'\s+', '_', safe_keyword.strip())  # Hilangkan spasi berlebih
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{safe_keyword}_{timestamp}.csv"
        file_path = os.path.join(save_dir, filename)

        # Pindah ke direktori kerja
        os.chdir(save_dir)

        # Perintah scraping
        command = (
            f"npx -y tweet-harvest@2.6.1 "
            f"-o \"{filename}\" -s \"{keyword}\" --tab \"LATEST\" -l {tweet_limit} --token {auth_token}"
        )
        result = os.system(command)

        # Kembalikan ke direktori awal
        os.chdir(os.path.dirname(__file__))

        # Validasi file secara manual ke dalam direktori 'data/tweets-data'
        manual_path = os.path.join(os.getcwd(), "data", "tweets-data", filename)


        # Validasi file
        if os.path.exists(manual_path):
            return render_template('01_create_dataset.html',
                                title="Buat Dataset",
                                success_message=f"Dataset berhasil dibuat: {filename}",
                                download_link=url_for('download_file', filename=filename),
                                keyword=keyword, tweet_limit=tweet_limit, auth_token=auth_token)
        else:
            flash(f"Gagal membuat dataset. Periksa kembali inputan atau token. {file_path}", "error")
            return redirect(url_for('create_dataset'))
        
    return render_template('01_create_dataset.html', title="Buat Dataset")

# Route untuk mengunduh file yang sudah dibuat
@app.route('/download/<filename>')
def download_file(filename):
    if url_for('create_dataset'):
        try:
            download_dir = os.path.join(os.getcwd(), "data", "tweets-data")
            return send_from_directory(download_dir, filename, as_attachment=True)
        except Exception as e:
            flash("Gagal mengunduh file.", "error")
            return redirect(url_for('create_dataset'))
    elif url_for('clean_dataset'):
        try:
            download_dir = os.path.join(os.getcwd(), "data", "processed")
            return send_from_directory(download_dir, filename, as_attachment=True)
        except Exception as e:
            flash("Gagal mengunduh file.", "error")
            return redirect(url_for('clean_dataset'))

# Route untuk halaman Unggah Dataset
@app.route('/upload-dataset', methods=['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada file yang diunggah.', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('Pilih file terlebih dahulu.', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Penamaan file menjadi dataset_0_raw.csv
            filename = "dataset_0_raw.csv"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Jika file bukan CSV, konversi ke CSV
            if not filename.endswith('.csv'):
                try:
                    data = pd.read_excel(filepath)
                    data.to_csv(filepath, index=False, sep=',')  # Konversi ke CSV dengan pembatas koma
                except Exception as e:
                    flash(f"Error saat konversi file: {str(e)}", 'error')
                    return redirect(request.url)

            # Validasi pembatas CSV hanya jika file sudah CSV
            else:
                try:
                    # Pastikan pembatas CSV benar
                    with open(filepath, 'r') as f:
                        sample = f.read(1024)
                        detected_delimiter = csv.Sniffer().sniff(sample).delimiter
                    if detected_delimiter != ',':
                        data = pd.read_csv(filepath, delimiter=detected_delimiter)
                        data.to_csv(filepath, index=False, sep=',')  # Ubah pembatas menjadi koma
                except Exception as e:
                    flash(f"Error saat membaca file CSV: {str(e)}", 'error')
                    return redirect(request.url)

            # Ubah nama kolom `full_text` menjadi `Tweet` jika ada
            data = pd.read_csv(filepath)
            if 'full_text' in data.columns:
                data.rename(columns={'full_text': 'Tweet'}, inplace=True)
            
            # Tambahkan kolom `Tweet Length` untuk menghitung jumlah kata dalam kolom `Tweet`
            if 'Tweet' in data.columns:
                data['Tweet Length'] = data['Tweet'].apply(lambda x: len(str(x).split()))
            else:
                flash("Kolom 'Tweet' tidak ditemukan dalam dataset!", "error")
                return redirect(request.url)

            # Simpan kembali file setelah perubahan
            data.to_csv(filepath, index=False)

            flash('Dataset berhasil diunggah dan disimpan sebagai dataset_0_raw.csv!', 'success')
            # Redirect ke halaman details_dataset
            return redirect(url_for('details_dataset'))

    return render_template('02_upload_dataset.html', title="Unggah Dataset")

# Route untuk Rincian Dataset
@app.route('/details-dataset', methods=['GET', 'POST'])
def details_dataset():
    try:
        # Semua angka ditampilkan dengan dua desimal.
        pd.options.display.float_format = '{:.2f}'.format
        
        # Ambil file dari dropdown (jika ada pilihan file dari user)
        selected_file = None
        if request.method == 'POST':
            selected_file = request.form.get('selected_file')
        
        # Default ke file utama jika tidak ada file yang dipilih
        if not selected_file:
            selected_file = "dataset_0_raw.csv"
        
        # Path file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
        
        # Validasi keberadaan file
        if not os.path.exists(filepath):
            flash(f"File {selected_file} tidak ditemukan di sistem.", "error")
            return render_template('03_details_dataset.html', title="Rincian Dataset", file_details=os.listdir(app.config['UPLOAD_FOLDER']))
        
        # Membaca dataset
        data = pd.read_csv(filepath)
        
        # Pastikan kolom 'Tweet' atau 'full_text' ada
        if 'full_text' in data.columns:
            data.rename(columns={'full_text': 'Tweet'}, inplace=True)
        
        if 'Tweet' not in data.columns:
            flash("Kolom 'Tweet' tidak ditemukan dalam dataset.", "error")
            return render_template('03_details_dataset.html', title="Rincian Dataset", file_details=os.listdir(app.config['UPLOAD_FOLDER']))

        # Tambahkan kolom 'Tweet Length'
        data['Tweet Length'] = data['Tweet'].apply(lambda x: len(str(x).split()))

        # Ringkasan dataset
        data_head = data.head().to_html(classes='table table-striped', index=False)
        # Tangkap output dari data.info()
        buffer = io.StringIO()
        data.info(buf=buffer)
        data_info = buffer.getvalue()
        data_description = data.describe().to_html(classes='table table-striped')
        data_shape = data.shape  # Dimensi dataset
        # Jumlah nilai unik
        data_unique = pd.DataFrame(data.nunique(), columns=['Unique Values']).reset_index()
        data_unique.rename(columns={'index': 'Column'}, inplace=True)
        data_unique_html = data_unique.to_html(classes='table table-striped', index=False)
        duplicate_count = data.duplicated().sum()  # Jumlah duplikat
        null_count = data.isnull().sum().sum()  # Jumlah nilai kosong

        # Deteksi elemen yang perlu dibersihkan
        def detect_emoji(text):
            if isinstance(text, str):
                emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"
                    u"\U0001F300-\U0001F5FF"
                    u"\U0001F680-\U0001F6FF"
                    u"\U0001F700-\U0001F77F"
                    u"\U0001F780-\U0001F7FF"
                    u"\U0001F800-\U0001F8FF"
                    u"\U0001F900-\U0001F9FF"
                    u"\U0001FA00-\U0001FA6F"
                    u"\U0001FA70-\U0001FAFF"
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)
                return bool(emoji_pattern.search(text))
            return False

        emoji_tweets = len(data[data['Tweet'].apply(detect_emoji)])
        links = len(data[data['Tweet'].str.contains("http|www|<a", na=False)])
        symbols = len(data[data['Tweet'].str.contains(r'[^\w\s]', na=False)])
        empty_tweets = len(data[data['Tweet'].str.strip() == ''])
        only_numbers = len(data[data['Tweet'].str.match(r'^\d+$', na=False)])
        tweets_with_numbers = len(data[data['Tweet'].str.contains(r'\d', na=False)])
        short_tweets = len(data[data['Tweet Length'] < 3])

        # Visualisasi distribusi panjang Tweet
        chart_path = os.path.join('static', 'img', 'tweet_0_length_distribution.png')
        plt.figure(figsize=(16, 9))
        data['Tweet Length'].plot(kind='hist', bins=30, title='Distribusi Panjang Tweet', color='blue', edgecolor='black')
        plt.xlabel('Jumlah Kata')
        plt.ylabel('Jumlah Tweet')
        plt.savefig(chart_path, bbox_inches='tight', facecolor='white')
        plt.close()

        # Visualisasi WordCloud
        wordcloud_path = os.path.join('static', 'img', 'tweet_0_wordcloud.png')
        text = ' '.join(data['Tweet'].dropna())
        wordcloud = WordCloud(width=1280, height=720, background_color='white').generate(text)
        plt.figure(figsize=(16, 9))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(wordcloud_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return render_template(
            '03_details_dataset.html',
            title="Rincian Dataset",
            file_details=os.listdir(app.config['UPLOAD_FOLDER']),
            selected_file=selected_file,
            data_head=data_head,
            data_info=data_info,
            data_description=data_description,
            data_shape=data_shape,
            data_unique=data_unique_html,
            duplicate_count=duplicate_count,
            null_count=null_count,
            emoji_tweets=emoji_tweets,
            links=links,
            symbols=symbols,
            empty_tweets=empty_tweets,
            only_numbers=only_numbers,
            tweets_with_numbers=tweets_with_numbers,
            short_tweets=short_tweets,
            chart_path=chart_path,
            wordcloud_path=wordcloud_path
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "error")
        return render_template('03_details_dataset.html', title="Rincian Dataset", file_details=os.listdir(app.config['UPLOAD_FOLDER']))

# Route untuk Pembersihan Data
@app.route('/clean-dataset', methods=['GET', 'POST'])
def clean_dataset():
    try:
        # Ambil daftar file di direktori upload
        uploaded_files_list = os.listdir(app.config['UPLOAD_FOLDER'])
        uploaded_files_list = [f for f in uploaded_files_list if allowed_file(f)]

        selected_file = None
        if len(uploaded_files_list) == 1:
            selected_file = uploaded_files_list[0]
        elif request.method == 'POST':
            selected_file = request.form.get('selected_file')

        if not selected_file:
            flash("Silakan pilih dataset untuk dibersihkan.", "error")
            return render_template('11_cleaned_dataset.html', file_details=uploaded_files_list, title="Pembersihan Data")

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
        output_file = "dataset_1_cleaned.csv"
        output_path = os.path.join(os.getcwd(), 'data', 'processed', output_file)

        # Pastikan direktori processed ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Membaca dataset
        data = pd.read_csv(input_path)

        # Pastikan kolom 'Tweet' atau 'Review' ada
        if 'Review' in data.columns:
            data.rename(columns={'Review': 'Tweet'}, inplace=True)

        if 'Tweet' not in data.columns:
            flash("Kolom 'Tweet' tidak ditemukan dalam dataset.", "error")
            return render_template('11_cleaned_dataset.html', title="Pembersihan Data")

        # Terapkan pembersihan
        data['Cleaned_Tweet'] = data['Tweet'].apply(clean_comment)

        # Hapus duplikat dan nilai kosong
        data = data.drop_duplicates(subset='Cleaned_Tweet')
        data = data.dropna(subset=['Cleaned_Tweet'])

        # Hapus teks dengan panjang <= 3 huruf
        data = data[data['Cleaned_Tweet'].str.len() > 3]

        # Simpan dataset yang telah dibersihkan
        data.to_csv(output_path, index=False)

        # Informasi dataset
        data_head = data.head().to_html(classes='table table-striped', index=False)
        data_description = data.describe().to_html(classes='table table-striped')
        data_shape = data.shape
        duplicate_count = data.duplicated().sum()
        null_count = data.isnull().sum().sum()

        # Tampilkan WordCloud
        wordcloud_path = os.path.join('static', 'img', 'tweet_1_wordcloud_cleaned.png')
        text = ' '.join(data['Cleaned_Tweet'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(wordcloud_path)
        plt.close()

        return render_template(
            '11_cleaned_dataset.html',
            title="Pembersihan Data",
            data_head=data_head,
            data_description=data_description,
            data_shape=data_shape,
            duplicate_count=duplicate_count,
            null_count=null_count,
            wordcloud_path=wordcloud_path,
            file_details=uploaded_files_list,
            selected_file=selected_file,
            download_link=url_for('download_file', output_file=output_file),
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "error")
        return render_template('11_cleaned_dataset.html', file_details=uploaded_files_list, title="Pembersihan Data")

@app.route('/pra-pemrosesan')
def pra_pemrosesan():
    return render_template('about.html', title="Tentang Aplikasi")
@app.route('/pemodelan')
def pemodelan():
    return render_template('pemodelan.html', title="Tentang Aplikasi")
@app.route('/evaluasi')
def evaluasi():
    return render_template('evaluasi.html', title="Tentang Aplikasi")
@app.route('/analisis-hasil')
def analisis_hasil():
    return render_template('analisis_hasil.html', title="Tentang Aplikasi")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)