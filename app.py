import os
import io
import pandas as pd
import re
import matplotlib.pyplot as plt
import secrets
import csv
import nltk
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from datetime import datetime
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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
        filename = f"dataset_0_{safe_keyword}_{timestamp}.csv"
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
    # Tentukan direktori berdasarkan nama file
    if filename.startswith("dataset_0"):
        download_dir = os.path.join(os.getcwd(), "data", "tweets-data")
    elif filename.startswith("dataset_1"):
        download_dir = os.path.join(os.getcwd(), "data", "processed")
    elif filename.startswith("dataset_2"):
        download_dir = os.path.join(os.getcwd(), "data", "processed")
    elif filename.startswith("dataset_3"):
        download_dir = os.path.join(os.getcwd(), "data", "processed")
    elif filename.startswith("dataset_4"):
        download_dir = os.path.join(os.getcwd(), "data", "processed")
    elif filename.startswith("dataset_5"):
        download_dir = os.path.join(os.getcwd(), "data", "processed")
    elif filename.startswith("dataset_6"):
        download_dir = os.path.join(os.getcwd(), "data", "processed")
    elif filename.startswith("dataset_7"):
        download_dir = os.path.join(os.getcwd(), "data", "processed")
    elif filename.startswith("dataset_8"):
        download_dir = os.path.join(os.getcwd(), "data", "processed")
    elif filename.startswith("dataset_9"):
        download_dir = os.path.join(os.getcwd(), "data", "processed")
    else:
        flash("File tidak valid untuk diunduh.", "error")
        return redirect(url_for('clean_dataset'))

    try:
        return send_from_directory(download_dir, filename, as_attachment=True)
    except Exception as e:
        flash("Gagal mengunduh file.", "error")
        return redirect(request.referrer)

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

            try:
                # Jika file bukan CSV, konversi ke CSV
                if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                    try:
                        # Gunakan engine openpyxl untuk membaca file .xlsx
                        data = pd.read_excel(filepath, engine='openpyxl')
                    except Exception as e1:
                        # Jika gagal dengan openpyxl, coba xlrd untuk file .xls
                        try:
                            data = pd.read_excel(filepath, engine='xlrd')
                        except Exception as e2:
                            flash(f"Error saat membaca file Excel: {str(e2)}", 'error')
                            return redirect(request.url)
                    
                    # Konversi ke CSV dengan pembatas koma
                    data.to_csv(filepath, index=False, sep=',')
                else:
                    # Validasi pembatas CSV
                    with open(filepath, 'r') as f:
                        sample = f.read(1024)
                        try:
                            detected_delimiter = csv.Sniffer().sniff(sample).delimiter
                        except csv.Error:
                            detected_delimiter = ','  # Default fallback delimiter

                    # Baca file CSV dengan pembatas yang terdeteksi
                    data = pd.read_csv(filepath, delimiter=detected_delimiter)
                    if detected_delimiter != ',':
                        # Simpan ulang file CSV dengan pembatas koma
                        data.to_csv(filepath, index=False, sep=',')
            except Exception as e:
                flash(f"Error saat membaca atau mengonversi file: {str(e)}", 'error')
                return redirect(request.url)

            # Ubah nama kolom `full_text` menjadi `Tweet` jika ada
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
        data_description = data.describe().round(2).to_html(classes='table table-striped')
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
        def clean_comment(text):
            text = str(text)  # Konversi ke string
            text = remove_emoji(text)  # Hapus emoji
            text = remove_links(text)  # Hapus tautan
            text = remove_html_tags_and_entities(text)  # Hapus tag HTML dan entitas
            text = remove_special_characters(text)  # Hapus simbol
            text = text.strip()  # Hapus spasi berlebih
            return text.lower()  # Ubah ke huruf kecil

        # Terapkan fungsi pembersihan
        data['Cleaned_Tweet'] = data['Tweet'].apply(clean_comment)

        # Hapus duplikat dan nilai kosong
        data = data.drop_duplicates(subset='Cleaned_Tweet')
        data = data.dropna(subset=['Cleaned_Tweet'])

        # Hapus teks dengan panjang <= 3 huruf
        data = data[data['Cleaned_Tweet'].str.len() > 3]

        # Hitung jumlah kata dari Cleaned_Tweet
        data['Cleaned_Length'] = data['Cleaned_Tweet'].apply(lambda x: len(x.split()))

        # Simpan dataset yang telah dibersihkan
        data.to_csv(output_path, index=False)
        
        data_head = data[['Tweet', 'Cleaned_Tweet']].head().to_html(classes='table table-striped', index=False)
        data_description = data.describe().round(2).to_html(classes='table table-striped')
        data_shape = data.shape
        duplicate_count = data.duplicated().sum()
        null_count = data.isnull().sum().sum()

        # Distribusi panjang Cleaned_Tweet
        chart_path = os.path.join('static', 'img', 'tweet_1_length_distribution_cleaned.png')
        plt.figure(figsize=(16, 9))
        data['Cleaned_Length'].plot(kind='hist', bins=30, color='blue', edgecolor='black', title='Distribusi Panjang Cleaned Tweet')
        plt.xlabel('Jumlah Kata')
        plt.ylabel('Frekuensi')
        plt.savefig(chart_path, bbox_inches='tight', facecolor='white')
        plt.close()

        # Tampilkan WordCloud
        wordcloud_path = os.path.join('static', 'img', 'tweet_1_wordcloud_cleaned.png')
        text = ' '.join(data['Cleaned_Tweet'].dropna())
        wordcloud = WordCloud(width=1280, height=720, background_color='white').generate(text)
        plt.figure(figsize=(16, 9))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(wordcloud_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return render_template(
            '11_cleaned_dataset.html',
            title="Pembersihan Data",
            data_head=data_head,
            data_description=data_description,
            data_shape=data_shape,
            duplicate_count=duplicate_count,
            null_count=null_count,
            chart_path=chart_path,
            wordcloud_path=wordcloud_path,
            file_details=uploaded_files_list,
            selected_file=selected_file,
            download_link=url_for('download_file', filename=output_file)
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "error")
        return render_template('11_cleaned_dataset.html', file_details=uploaded_files_list, title="Pembersihan Data")

# Route untuk Normalisasi Data
@app.route('/normalize-dataset', methods=['GET', 'POST'])
def normalize_dataset():
    try:
        # Ambil daftar file di direktori processed
        processed_files_list = os.listdir(os.path.join(os.getcwd(), 'data', 'processed'))
        processed_files_list = [f for f in processed_files_list if allowed_file(f)]

        selected_file = 'dataset_1_cleaned.csv'
        input_path = os.path.join(os.getcwd(), 'data', 'processed', selected_file)
        output_file = "dataset_2_normalized.csv"
        output_path = os.path.join(os.getcwd(), 'data', 'processed', output_file)

        # Pastikan direktori processed ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Membaca dataset
        data = pd.read_csv(input_path)

        # Pastikan kolom 'Cleaned_Tweet' ada
        if 'Cleaned_Tweet' not in data.columns:
            flash("Kolom 'Cleaned_Tweet' tidak ditemukan dalam dataset.", "error")
            return render_template('12_normalized_dataset.html', file_details=processed_files_list, title="Normalisasi Data")

        # Normalisasi Data
        def normalize_text(text):
            text = text.lower()  # Konversi ke huruf kecil
            text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
            text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
            return text

        data['Normalized_Tweet'] = data['Cleaned_Tweet'].apply(normalize_text)

        # Hitung jumlah kata dari Cleaned_Tweet
        data['Normalized_Length'] = data['Normalized_Tweet'].apply(lambda x: len(x.split()))

        # Simpan dataset yang telah dinormalisasi
        data.to_csv(output_path, index=False)

        # Informasi dataset
        data_head = data[['Cleaned_Tweet', 'Normalized_Tweet']].head().to_html(classes='table table-striped', index=False)
        data_description = data.describe().to_html(classes='table table-striped')
        data_shape = data.shape

        # Distribusi panjang Normalized_Tweet
        chart_path = os.path.join('static', 'img', 'tweet_2_length_distribution_cleaned.png')
        plt.figure(figsize=(16, 9))
        data['Normalized_Length'].plot(kind='hist', bins=30, color='blue', edgecolor='black', title='Distribusi Panjang Normalized Tweet')
        plt.xlabel('Jumlah Kata')
        plt.ylabel('Frekuensi')
        plt.savefig(chart_path, bbox_inches='tight', facecolor='white')
        plt.close()

        # Tampilkan WordCloud
        wordcloud_path = os.path.join('static', 'img', 'tweet_2_wordcloud_normalized.png')
        text = ' '.join(data['Normalized_Tweet'].dropna())
        wordcloud = WordCloud(width=1280, height=720, background_color='white').generate(text)
        plt.figure(figsize=(16, 9))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(wordcloud_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return render_template(
            '12_normalized_dataset.html',
            title="Normalisasi Data",
            data_head=data_head,
            data_description=data_description,
            data_shape=data_shape,
            file_details=processed_files_list,
            selected_file=selected_file,
            download_link=url_for('download_file', filename=output_file),
            chart_path=chart_path,
            wordcloud_path=wordcloud_path
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "error")
        return render_template('12_normalized_dataset.html', file_details=processed_files_list, title="Normalisasi Data")

# Route untuk Tokenisasi Data
@app.route('/tokenize-dataset', methods=['GET', 'POST'])
def tokenize_dataset():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        # Ambil daftar file di direktori processed
        processed_files_list = os.listdir(os.path.join(os.getcwd(), 'data', 'processed'))
        processed_files_list = [f for f in processed_files_list if allowed_file(f)]

        selected_file = 'dataset_2_normalized.csv'
        input_path = os.path.join(os.getcwd(), 'data', 'processed', selected_file)
        output_file = "dataset_3_tokenized.csv"
        output_path = os.path.join(os.getcwd(), 'data', 'processed', output_file)

        # Pastikan direktori processed ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Membaca dataset
        data = pd.read_csv(input_path)

        # Pastikan kolom 'Normalized_Tweet' ada
        if 'Normalized_Tweet' not in data.columns:
            flash("Kolom 'Normalized_Tweet' tidak ditemukan dalam dataset.", "error")
            return render_template('13_tokenized_dataset.html', file_details=processed_files_list, title="Tokenisasi Data")

        # Tokenisasi Data
        

        # Fungsi tokenisasi dengan nltk
        def tokenize_text(text):
            tokens = word_tokenize(text)
            return '["' + '", "'.join(tokens) + '"]'

        data['Tokenized_Tweet'] = data['Normalized_Tweet'].apply(tokenize_text)

        # Tambahkan kolom jumlah token
        data['Token_Count'] = data['Tokenized_Tweet'].apply(len)

        # Simpan dataset yang telah ditokenisasi
        data.to_csv(output_path, index=False)

        # Informasi dataset
        data_head = data[['Normalized_Tweet', 'Tokenized_Tweet']].head().to_html(classes='table table-striped', index=False)
        data_shape = data.shape

        # Tampilkan WordCloud
        wordcloud_path = os.path.join('static', 'img', 'tweet_3_wordcloud_tokenized.png')
        text = ' '.join(data['Normalized_Tweet'].dropna())
        wordcloud = WordCloud(width=1280, height=720, background_color='white').generate(text)
        plt.figure(figsize=(16, 9))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(wordcloud_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return render_template(
            '13_tokenized_dataset.html',
            title="Tokenisasi Data",
            data_head=data_head,
            data_shape=data_shape,
            file_details=processed_files_list,
            selected_file=selected_file,
            download_link=url_for('download_file', filename=output_file),
            wordcloud_path=wordcloud_path
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "error")
        return render_template('13_tokenized_dataset.html', file_details=processed_files_list, title="Tokenisasi Data")

# Route untuk Penghapusan Stopwords
@app.route('/remove-stopwords', methods=['GET', 'POST'])
def remove_stopwords():
    try:
        nltk.download('stopwords', quiet=True)
        # Ambil daftar file di direktori processed
        processed_files_list = os.listdir(os.path.join(os.getcwd(), 'data', 'processed'))
        processed_files_list = [f for f in processed_files_list if allowed_file(f)]

        selected_file = 'dataset_3_tokenized.csv'
        input_path = os.path.join(os.getcwd(), 'data', 'processed', selected_file)
        output_file = "dataset_4_no_stopwords.csv"
        output_path = os.path.join(os.getcwd(), 'data', 'processed', output_file)

        # Pastikan direktori processed ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Membaca dataset
        data = pd.read_csv(input_path)

        # Pastikan kolom 'Tokenized_Tweet' ada
        if 'Tokenized_Tweet' not in data.columns:
            flash("Kolom 'Tokenized_Tweet' tidak ditemukan dalam dataset.", "error")
            return render_template('14_no_stopwords_dataset.html', file_details=processed_files_list, title="Penghapusan Stopwords")

        # Inisialisasi stopwords
        stop_words = set(stopwords.words('indonesian'))  # Anda bisa mengganti dengan bahasa lain jika diperlukan

        # Ambil kata-kata kostum dari input form (POST request)
        manual_stopwords = ['gelo', 'mentri2', 'yg', 'ga', 'udh', 'aja', 'kaga', 'bgt', 'spt', 'sdh',
                            'dr', 'utan', 'tuh', 'budi', 'bodi', 'p', 'psi_id', 'fufufafa', 'pln', 'lu',
                            'krn','dah','jd','tdk','dll','golkar_id', 'dlm', 'ri', 'jg', 'ni', 'sbg',
                            'tp', 'nih', 'gini', 'jkw', 'nggak', 'bs', 'pk', 'ya', 'gk', 'gw', 'gua',
                            'klo', 'msh', 'blm', 'gue', 'sih', 'pa', 'dgn', 'n', 'skrg', 'pake', 'si',
                            'dg', 'utk', 'deh', 'tu', 'hrt', 'ala', 'mdy', 'moga', 'tau', 'liat', 'orang2',
                            'jadi']
        custom_stopwords = request.form.get('custom_stopwords')

        # Gabungkan manual_stopwords dengan custom_stopwords
        if custom_stopwords:
            # Pisahkan kata-kata dari input pengguna berdasarkan koma dan gabungkan dengan manual_stopwords
            custom_stopwords_list = [word.strip().lower() for word in custom_stopwords.split(',')]
            stop_words.update(manual_stopwords)  # Tambahkan stopwords manual
            stop_words.update(custom_stopwords_list)  # Tambahkan stopwords dari input pengguna
        else:
            stop_words.update(manual_stopwords)  # Tambahkan hanya stopwords manual jika tidak ada input

        # Fungsi untuk menghapus stopwords dari token
        def remove_stopwords_from_tokens(tokens):
            try:
                # Konversi token dari string ke list menggunakan eval
                tokens = eval(tokens)  # Mengubah string token menjadi list Python
                filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
                return '["' + '", "'.join(filtered_tokens) + '"]'  # Kembalikan dalam bentuk list string
            except Exception as e:
                return '[]'  # Jika ada error, kembalikan list kosong

        # Terapkan penghapusan stopwords
        data['No_Stopwords_Tweet'] = data['Tokenized_Tweet'].apply(remove_stopwords_from_tokens)

        # Tambahkan kolom jumlah token setelah penghapusan stopwords
        data['No_Stopwords_Count'] = data['No_Stopwords_Tweet'].apply(lambda x: len(eval(x)))

        # Simpan dataset hasil penghapusan stopwords
        data.to_csv(output_path, index=False)

        # Informasi dataset
        data_head = data[['Tokenized_Tweet', 'No_Stopwords_Tweet']].head().to_html(classes='table table-striped', index=False)
        data_shape = data.shape

        # Tampilkan WordCloud
        wordcloud_path = os.path.join('static', 'img', 'tweet_4_wordcloud_no_stopwords.png')
        text = ' '.join([' '.join(eval(tokens)) for tokens in data['No_Stopwords_Tweet'] if eval(tokens)])
        if len(text) > 0:
            wordcloud = WordCloud(width=1280, height=720, background_color='white').generate(text)
            plt.figure(figsize=(16, 9))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.savefig(wordcloud_path, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            wordcloud_path = None  # Jika tidak ada kata valid, tidak membuat WordCloud

        return render_template(
            '14_no_stopwords_dataset.html',
            title="Penghapusan Stopwords",
            data_head=data_head,
            data_shape=data_shape,
            file_details=processed_files_list,
            selected_file=selected_file,
            download_link=url_for('download_file', filename=output_file),
            wordcloud_path=wordcloud_path
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "error")
        return render_template('14_no_stopwords_dataset.html', file_details=processed_files_list, title="Penghapusan Stopwords")

@app.route('/stemming-dataset', methods=['GET', 'POST'])
def stemming_dataset():
    try:
        # Inisialisasi Stemmer Sastrawi
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        
        # Ambil daftar file di direktori processed
        processed_files_list = os.listdir(os.path.join(os.getcwd(), 'data', 'processed'))
        processed_files_list = [f for f in processed_files_list if allowed_file(f)]

        selected_file = 'dataset_4_no_stopwords.csv'
        input_path = os.path.join(os.getcwd(), 'data', 'processed', selected_file)
        output_file = "dataset_5_stemmed.csv"
        output_path = os.path.join(os.getcwd(), 'data', 'processed', output_file)

        # Pastikan direktori processed ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Membaca dataset
        data = pd.read_csv(input_path)

        # Pastikan kolom 'No_Stopwords_Tweet' ada
        if 'No_Stopwords_Tweet' not in data.columns:
            flash("Kolom 'No_Stopwords_Tweet' tidak ditemukan dalam dataset.", "error")
            return render_template('15_stemmed_dataset.html', file_details=processed_files_list, title="Stemming Data")

        # Fungsi untuk stemming menggunakan Sastrawi
        def stem_tokens(tokens):
            if isinstance(tokens, str):
                # Konversi string token kembali menjadi satu kalimat
                sentence = ' '.join(eval(tokens))
                # Stem kalimat
                stemmed_sentence = stemmer.stem(sentence)
                # Ubah kembali menjadi token
                return '["' + '", "'.join(stemmed_sentence.split()) + '"]'
            return tokens

        # Terapkan stemming
        data['Stemmed_Tweet'] = data['No_Stopwords_Tweet'].apply(stem_tokens)

        # Tambahkan kolom jumlah token setelah stemming
        data['Stemmed_Count'] = data['Stemmed_Tweet'].apply(lambda x: len(eval(x)))

        # Simpan dataset hasil stemming
        data.to_csv(output_path, index=False)

        # Informasi dataset
        data_head = data[['No_Stopwords_Tweet', 'Stemmed_Tweet']].head().to_html(classes='table table-striped', index=False)
        data_shape = data.shape

        # Tampilkan WordCloud
        wordcloud_path = os.path.join('static', 'img', 'tweet_5_wordcloud_stemmed.png')
        text = ' '.join([' '.join(eval(tokens)) for tokens in data['Stemmed_Tweet'] if isinstance(tokens, str)])
        wordcloud = WordCloud(width=1280, height=720, background_color='white').generate(text)
        plt.figure(figsize=(16, 9))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(wordcloud_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return render_template(
            '15_stemmed_dataset.html',
            title="Stemming Data",
            data_head=data_head,
            data_shape=data_shape,
            file_details=processed_files_list,
            selected_file=selected_file,
            download_link=url_for('download_file', filename=output_file),
            wordcloud_path=wordcloud_path
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "error")
        return render_template('15_stemmed_dataset.html', file_details=processed_files_list, title="Stemming Data")

@app.route('/label-dataset', methods=['GET', 'POST'])
def label_dataset():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        # Ambil daftar file di direktori processed
        processed_files_list = os.listdir(os.path.join(os.getcwd(), 'data', 'processed'))
        processed_files_list = [f for f in processed_files_list if allowed_file(f)]

        selected_file = 'dataset_5_stemmed.csv'
        input_path = os.path.join(os.getcwd(), 'data', 'processed', selected_file)
        output_file = "dataset_6_labeled.csv"
        output_path = os.path.join(os.getcwd(), 'data', 'processed', output_file)

        # Pastikan direktori processed ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Membaca dataset
        data = pd.read_csv(input_path)

        # Pastikan kolom 'Stemmed_Tweet' ada
        if 'Stemmed_Tweet' not in data.columns:
            flash("Kolom 'Stemmed_Tweet' tidak ditemukan dalam dataset.", "error")
            return render_template('16_labeled_dataset.html', file_details=processed_files_list, title="Pelabelan Data")

        # Daftar kata positif dan negatif (custom)
        positive_words = ['baik', 'bagus', 'hebat', 'senang', 'positif', 'sukses', 'puas', 'luar biasa', 'bijak', 
                        'maju', 'bersih', 'solid', 'kuat', 'efisien', 'mudah', 'nyaman', 'aman', 'sehat', 'stabil',
                        'berani', 'dukung','percaya', 'menang', 'bener','atur', 'pangan', 'peduli', 'kerja',]
        negative_words = ['buruk', 'jelek', 'gagal', 'sedih', 'negatif', 'kecewa', 'menyedihkan', 'masalah',
                        'gemuk', 'kotor', 'pecat', 'korup', 'korupsi', 'salah', 'bayar', 'apbn', 'moga', 'hrt',
                        'jelek', 'jilat', 'maling', 'parah', 'mati', 'berat', 'bayang', 'gendut', 'buang',
                        'gak', 'tunggu', 'biaya', 'kuasa','ganti', 'gebrak', 'rusak','koruptor', 'lapor',
                        'jokowi', 'gibran', 'anies', 'ahok', 'sandi']

        # Ambil kata-kata custom dari form input
        if request.method == 'POST':
            custom_positive = request.form.get('custom_positive', '')
            custom_negative = request.form.get('custom_negative', '')
            if custom_positive:
                positive_words.extend([word.strip().lower() for word in custom_positive.split(',')])
            if custom_negative:
                negative_words.extend([word.strip().lower() for word in custom_negative.split(',')])

        # Inisialisasi analyzer VADER
        analyzer = SentimentIntensityAnalyzer()

        # Fungsi untuk menentukan sentimen
        def determine_sentiment_vader(tokens):
            tokens = eval(tokens)  # Pastikan token dalam format list
            text = ' '.join(tokens)  # Gabungkan token menjadi teks
            # Prioritas pada custom kata-kata
            if any(word in tokens for word in positive_words):
                return 'positif'
            elif any(word in tokens for word in negative_words):
                return 'negatif'
            # Gunakan VADER jika tidak ditemukan kata custom
            score = analyzer.polarity_scores(text)
            if score['compound'] >= 0.05:
                return 'positif'
            elif score['compound'] <= -0.05:
                return 'negatif'
            else:
                return 'netral'

        # Terapkan pelabelan sentimen menggunakan VADER dan custom kata
        data['Sentimen'] = data['Stemmed_Tweet'].apply(determine_sentiment_vader)

        # Hitung jumlah setiap sentimen
        sentiment_counts = data['Sentimen'].value_counts().to_dict()

        # Simpan dataset yang telah dilabeli
        data.to_csv(output_path, index=False)

        # Informasi dataset
        data_head = data[['Stemmed_Tweet', 'Sentimen']].head().to_html(classes='table table-striped', index=False)
        data_shape = data.shape
        
        # Visualisasi distribusi sentimen
        plt.figure(figsize=(16, 9))
        # plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red', 'blue'])
        plt.bar(
            ['Positif', 'Negatif', 'Netral'],
            [sentiment_counts.get('positif', 0), sentiment_counts.get('negatif', 0), sentiment_counts.get('netral', 0)],
            color=['green', 'red', 'gray']
        )
        plt.xlabel('Sentimen', fontsize=12)
        plt.ylabel('Jumlah', fontsize=12)
        plt.title('Distribusi Sentimen', fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Simpan gambar distribusi
        distribution_path = os.path.join('static', 'img', 'tweet_6_length_distribution_labelled.png')
        plt.savefig(distribution_path, bbox_inches='tight', facecolor='white')
        plt.close()

        # Tampilkan WordCloud berdasarkan sentimen
        wordcloud_paths = {}
        for sentiment in ['positif', 'negatif', 'netral']:
            sentiment_text = ' '.join(data[data['Sentimen'] == sentiment]['Stemmed_Tweet'].apply(lambda x: ' '.join(eval(x))))
            if sentiment_text.strip():  # Pastikan teks tidak kosong
                wordcloud = WordCloud(width=1280, height=720, background_color='white').generate(sentiment_text)
                wordcloud_path = os.path.join('static', 'img', f'tweet_6_wordcloud_sentiment_{sentiment}.png')
                wordcloud.to_file(wordcloud_path)
                wordcloud_paths[sentiment] = wordcloud_path
            else:
                wordcloud_paths[sentiment] = None  # Tidak ada WordCloud untuk kategori ini

        return render_template(
            '16_labeled_dataset.html',
            title="Pelabelan Data",
            data_head=data_head,
            data_shape=data_shape,
            file_details=processed_files_list,
            selected_file=selected_file,
            download_link=url_for('download_file', filename=output_file),
            wordcloud_paths=wordcloud_paths,
            sentiment_counts=sentiment_counts,
            distribution_path=distribution_path
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "error")
        return render_template('16_labeled_dataset.html', file_details=processed_files_list, title="Pelabelan Data")

@app.route('/label-encode', methods=['GET', 'POST'])
def label_encode():
    try:
        # Ambil daftar file di direktori processed
        processed_files_list = os.listdir(os.path.join(os.getcwd(), 'data', 'processed'))
        processed_files_list = [f for f in processed_files_list if allowed_file(f)]

        selected_file = 'dataset_6_labeled.csv'
        input_path = os.path.join(os.getcwd(), 'data', 'processed', selected_file)
        output_file = "dataset_7_encoded.csv"
        output_path = os.path.join(os.getcwd(), 'data', 'processed', output_file)

        # Pastikan direktori processed ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Membaca dataset
        data = pd.read_csv(input_path)

        # Pastikan kolom 'Sentimen' ada
        if 'Sentimen' not in data.columns:
            flash("Kolom 'Sentimen' tidak ditemukan dalam dataset.", "error")
            return render_template('17_encoded_dataset.html', file_details=processed_files_list, title="Label Encoding")

        # Mapping untuk Label Encoding
        sentiment_mapping = {'positif': 1, 'netral': 0, 'negatif': -1}
        data['Encoded_Sentimen'] = data['Sentimen'].map(sentiment_mapping)
        
        # Hitung distribusi label encoding
        encoded_counts = data['Encoded_Sentimen'].value_counts().to_dict()

        # Simpan dataset dengan label encoded
        data.to_csv(output_path, index=False)

        # Informasi dataset
        data_head = data[['Stemmed_Tweet', 'Sentimen', 'Encoded_Sentimen']].head().to_html(classes='table table-striped', index=False)
        data_shape = data.shape
        
        # Membuat Grafik Distribusi Label Encoding
        distribution_path = os.path.join('static', 'img', 'tweet_7_length_distribution_encoded.png')
        plt.figure(figsize=(16, 9))
        plt.bar(
            ['Negatif (-1)', 'Netral (0)', 'Positif (1)'],
            [encoded_counts.get(-1, 0), encoded_counts.get(0, 0), encoded_counts.get(1, 0)],
            color=['red', 'gray', 'green']
        )
        plt.title('Distribusi Label Encoding')
        plt.xlabel('Label Encoding')
        plt.ylabel('Jumlah')
        plt.savefig(distribution_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return render_template(
            '17_encoded_dataset.html',
            title="Label Encoding",
            data_head=data_head,
            data_shape=data_shape,
            file_details=processed_files_list,
            selected_file=selected_file,
            encoded_counts=encoded_counts,
            download_link=url_for('download_file', filename=output_file),
            distribution_path=distribution_path
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "error")
        return render_template('17_encoded_dataset.html', file_details=processed_files_list, title="Label Encoding")

@app.route('/feature-representation', methods=['GET', 'POST'])
def feature_representation():
    try:
        # Ambil daftar file di direktori processed
        processed_files_list = os.listdir(os.path.join(os.getcwd(), 'data', 'processed'))
        processed_files_list = [f for f in processed_files_list if allowed_file(f)]

        selected_file = 'dataset_7_encoded.csv'
        input_path = os.path.join(os.getcwd(), 'data', 'processed', selected_file)
        output_file = "dataset_8_tfidf_features.csv"
        output_path = os.path.join(os.getcwd(), 'data', 'processed', output_file)

        # Pastikan direktori processed ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Membaca dataset
        data = pd.read_csv(input_path)

        # Pastikan kolom 'Stemmed_Tweet' ada
        if 'Stemmed_Tweet' not in data.columns:
            flash("Kolom 'Stemmed_Tweet' tidak ditemukan dalam dataset.", "error")
            return render_template('18_tfidf_dataset.html', file_details=processed_files_list, title="Representasi Fitur")
        
        # Konversi kolom 'Stemmed_Tweet' ke string
        data['Stemmed_Tweet'] = data['Stemmed_Tweet'].astype(str).fillna("")

        # Terapkan TF-IDF pada kolom Stemmed_Tweet
        tfidf = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\b\w+\b',
            lowercase=True,
            max_features=1000
        )
        tfidf_matrix = tfidf.fit_transform(data['Stemmed_Tweet'].apply(lambda x: ' '.join(eval(x))))
        feature_names = tfidf.get_feature_names_out()

        # Konversi hasil TF-IDF ke DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        tfidf_df['Sentimen'] = data['Sentimen']

        # Simpan dataset dengan fitur TF-IDF
        tfidf_df.to_csv(output_path, index=False)

        # Informasi dataset
        data_shape = tfidf_df.shape

        # Distribusi fitur TF-IDF (10 fitur teratas)
        feature_sums = tfidf_df.drop(columns=['Sentimen']).sum(axis=0).sort_values(ascending=False)[:10]

        # Visualisasi distribusi fitur
        feature_dist_path = os.path.join('static', 'img', 'tweet_8_tfidf_feature_distribution.png')
        plt.figure(figsize=(16, 9))
        feature_sums.plot(kind='barh', color='skyblue', edgecolor='black')
        plt.xlabel('Total Bobot')
        plt.ylabel('Fitur')
        plt.title('10 Fitur Teratas Berdasarkan TF-IDF')
        plt.gca().invert_yaxis()  # Balikkan urutan agar fitur dengan bobot terbesar di atas
        plt.savefig(feature_dist_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return render_template(
            '18_tfidf_dataset.html',
            title="Representasi Fitur",
            data_shape=data_shape,
            file_details=processed_files_list,
            selected_file=selected_file,
            download_link=url_for('download_file', filename=output_file),
            feature_dist_path=feature_dist_path
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "error")
        return render_template('18_tfidf_dataset.html', file_details=processed_files_list, title="Representasi Fitur")

@app.route('/split-dataset', methods=['GET', 'POST'])
def split_dataset():
    try:
        from sklearn.model_selection import train_test_split

        # Ambil daftar file di direktori processed
        processed_files_list = os.listdir(os.path.join(os.getcwd(), 'data', 'processed'))
        processed_files_list = [f for f in processed_files_list if allowed_file(f)]

        selected_file = 'dataset_8_tfidf_features.csv'
        input_path = os.path.join(os.getcwd(), 'data', 'processed', selected_file)
        train_file = "dataset_9_train.csv"
        test_file = "dataset_9_test.csv"
        train_path = os.path.join(os.getcwd(), 'data', 'processed', train_file)
        test_path = os.path.join(os.getcwd(), 'data', 'processed', test_file)

        # Pastikan direktori processed ada
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)

        # Membaca dataset
        data = pd.read_csv(input_path)

        # Pastikan kolom 'Sentimen' ada
        if 'Sentimen' not in data.columns:
            flash("Kolom 'Sentimen' tidak ditemukan dalam dataset.", "error")
            return render_template('19_split_dataset.html', file_details=processed_files_list, title="Pemisahan Data")

        # Pisahkan fitur dan label
        X = data.drop(columns=['Sentimen'])
        y = data['Sentimen']

        # Pemisahan data dengan rasio 70:30
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Gabungkan kembali fitur dan label untuk data latih dan data uji
        train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
        test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

        # Simpan data latih dan data uji ke file
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        # Informasi dataset
        data_shape = data.shape
        train_shape = train_data.shape
        test_shape = test_data.shape

        # Visualisasi distribusi label dalam data latih dan data uji
        train_label_dist = y_train.value_counts()
        test_label_dist = y_test.value_counts()
        train_sentiment_counts = train_data['Sentimen'].value_counts()
        test_sentiment_counts = test_data['Sentimen'].value_counts()

        train_dist_path = os.path.join('static', 'img', 'train_label_distribution.png')
        test_dist_path = os.path.join('static', 'img', 'test_label_distribution.png')

        plt.figure(figsize=(16, 9))
        plt.bar(
            ['Negatif (-1)', 'Netral (0)', 'Positif (1)'],
            [train_label_dist.get(-1, 0), train_label_dist.get(0, 0), train_label_dist.get(1, 0)],
            color=['red', 'gray', 'green']
        )
        plt.title('Distribusi Label dalam Data Latih')
        plt.xlabel('Label')
        plt.ylabel('Jumlah')
        plt.savefig(train_dist_path, bbox_inches='tight', facecolor='white')
        plt.close()

        plt.figure(figsize=(16, 9))
        plt.bar(
            ['Negatif (-1)', 'Netral (0)', 'Positif (1)'],
            [test_label_dist.get(-1, 0), test_label_dist.get(0, 0), test_label_dist.get(1, 0)],
            color=['red', 'gray', 'green']
        )
        plt.title('Distribusi Label dalam Data Uji')
        plt.xlabel('Label')
        plt.ylabel('Jumlah')
        plt.savefig(test_dist_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return render_template(
            '19_split_dataset.html',
            title="Pemisahan Data",
            data_shape=data_shape,
            train_shape=train_shape,
            test_shape=test_shape,
            train_dist_path=train_dist_path,
            test_dist_path=test_dist_path,
            file_details=processed_files_list,
            selected_file=selected_file,
            train_download_link=url_for('download_file', filename=train_file),
            test_download_link=url_for('download_file', filename=test_file),
            train_sentiment_counts=train_sentiment_counts,
            test_sentiment_counts=test_sentiment_counts
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "error")
        return render_template('19_split_dataset.html', file_details=processed_files_list, title="Pemisahan Data")

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