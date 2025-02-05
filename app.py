import os
import io
import pandas as pd
import re
import secrets
import csv
import nltk
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import matplotlib
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    jsonify,
)
from datetime import datetime
from werkzeug.utils import secure_filename
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

matplotlib.use("Agg")

app = Flask(__name__)

# Atur secret key untuk keamanan
app.secret_key = secrets.token_hex(16)  # Gunakan string acak yang kuat dalam produksi

# Konfigurasi path folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), "data", "uploaded")
PROCESSED_FOLDER = os.path.join(os.getcwd(), "data", "processed")
STATIC_FOLDER = os.path.join(os.getcwd(), "static", "img")
MODELED_FOLDER = os.path.join(os.getcwd(), "data", "modeled")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER
app.config["MODELED_FOLDER"] = MODELED_FOLDER

# Pastikan direktori tersedia
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(MODELED_FOLDER, exist_ok=True)

# 🔹 Fungsi Cek Ekstensi File
ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx"}


# Fungsi pengecekan ekstensi file
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Fungsu mengahapus mentions
def remove_mentions(text):
    return re.sub(r"@\w+", "", text)


# Fungsi menghapus hashtag
def remove_hashtags(text):
    return re.sub(r"#\w+", "", text)


# Fungsi untuk menghapus karakter khusus
def remove_uniques(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


# Fungsi untuk menghapus emoji
def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


# Fungsi untuk menghapus tautan
def remove_links(text):
    return re.sub(r"http\S+|www\.\S+", "", text)


# Fungsi untuk menghapus tag HTML dan entitas
def remove_html_tags_and_entities(text):
    if isinstance(text, str):
        text = re.sub(r"<.*?>", "", text)  # Hapus tag HTML
        text = re.sub(r"&[a-z]+;", "", text)  # Hapus entitas HTML
    return text


# Fungsi untuk menghapus simbol atau karakter khusus
def remove_special_characters(text):
    return re.sub(r"[^\w\s]", "", text)


# Fungsi untuk menghapus underscore
def remove_underscores(text):
    if isinstance(text, str):
        return text.replace("_", "")  # Mengganti underscore dengan string kosong
    return text


# Route untuk halaman index
@app.route("/")
def index():
    return render_template("index.html", title="Dashboard")


# Route untuk halaman tentang
@app.route("/about")
def about():
    return render_template("about.html", title="Tentang Aplikasi")


# Route untuk mengunduh file yang sudah dibuat
@app.route("/download/<filename>")
def download_file(filename):
    # Tentukan direktori berdasarkan nama file
    if filename.startswith("dataset_1"):
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
    elif filename.startswith("model_1"):
        download_dir = os.path.join(os.getcwd(), "data", "modeled")
    elif filename.startswith("model_2"):
        download_dir = os.path.join(os.getcwd(), "data", "modeled")
    else:
        flash("File tidak valid untuk diunduh.", "danger")
        return redirect(url_for("/"))

    try:
        return send_from_directory(download_dir, filename, as_attachment=True)
    except Exception as e:
        flash("Gagal mengunduh file.", "danger")
        return redirect(request.referrer)


# @app.route('/feature-representation', methods=['GET', 'POST'])
# def feature_representation():

# === BERHENTI DI SINI ===


@app.route("/feature-representation-bow", methods=["GET", "POST"])
def feature_representation_bow():
    try:
        # Terapkan Bag of Words pada kolom Stemmed_Tweet
        from sklearn.feature_extraction.text import CountVectorizer

        # Ambil daftar file di direktori processed
        processed_files_list = os.listdir(
            os.path.join(os.getcwd(), "data", "processed")
        )
        processed_files_list = [f for f in processed_files_list if allowed_file(f)]

        selected_file = "dataset_7_encoded.csv"
        input_path = os.path.join(os.getcwd(), "data", "processed", selected_file)
        output_file = "dataset_8_tfidf_features.csv"
        output_path = os.path.join(os.getcwd(), "data", "processed", output_file)

        # Pastikan direktori processed ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Membaca dataset
        data = pd.read_csv(input_path)

        # Pastikan kolom 'Stemmed_Tweet' ada
        if "Stemmed_Tweet" not in data.columns:
            flash("Kolom 'Stemmed_Tweet' tidak ditemukan dalam dataset.", "danger")
            return render_template(
                "18_tfidf_dataset.html",
                file_details=processed_files_list,
                title="Representasi Fitur",
            )

        # Konversi kolom 'Stemmed_Tweet' ke string
        data["Stemmed_Tweet"] = data["Stemmed_Tweet"].astype(str).fillna("")

        # # Terapkan TF-IDF pada kolom Stemmed_Tweet
        # tfidf = TfidfVectorizer(
        #     analyzer='word',
        #     token_pattern=r'\b\w+\b',
        #     lowercase=True,
        #     max_features=1000
        # )
        # tfidf_matrix = tfidf.fit_transform(data['Stemmed_Tweet'].apply(lambda x: ' '.join(eval(x))))
        # feature_names = tfidf.get_feature_names_out()

        # # Konversi hasil TF-IDF ke DataFrame
        # tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

        # # Mapping untuk Label Encoding
        # sentiment_mapping = {'positif': 1, 'netral': 0, 'negatif': -1}
        # tfidf_df['Sentimen'] = data['Sentimen'].map(sentiment_mapping)

        # # Simpan dataset dengan fitur TF-IDF
        # tfidf_df.to_csv(output_path, index=False)

        # # Informasi dataset
        # data_shape = tfidf_df.shape

        # # Distribusi fitur TF-IDF (10 fitur teratas)
        # feature_sums = tfidf_df.drop(columns=['Sentimen']).sum(axis=0).sort_values(ascending=False)[:10]

        # # Visualisasi distribusi fitur
        # feature_dist_path = os.path.join('static', 'img', 'tweet_8_tfidf_feature_distribution.png')
        # plt.figure(figsize=(16, 9))
        # feature_sums.plot(kind='barh', color='skyblue', edgecolor='black')
        # plt.xlabel('Total Bobot')
        # plt.ylabel('Fitur')
        # plt.title('10 Fitur Teratas Berdasarkan TF-IDF')
        # plt.gca().invert_yaxis()  # Balikkan urutan agar fitur dengan bobot terbesar di atas
        # plt.savefig(feature_dist_path, bbox_inches='tight', facecolor='white')
        # plt.close()

        bow = CountVectorizer(
            analyzer="word",
            token_pattern=r"\b\w+\b",
            lowercase=True,
            max_features=999,  # Batasi hingga 1000 fitur (opsional)
        )
        bow_matrix = bow.fit_transform(
            data["Stemmed_Tweet"].apply(lambda x: " ".join(eval(x)))
        )
        feature_names = bow.get_feature_names_out()

        # Konversi hasil BoW ke DataFrame
        bow_df = pd.DataFrame(bow_matrix.toarray(), columns=feature_names)

        # Mapping untuk Label Encoding
        sentiment_mapping = {"positif": 1, "netral": 0, "negatif": -1}
        bow_df["Sentimen"] = data["Sentimen"].map(sentiment_mapping)

        # Simpan dataset dengan fitur BoW
        bow_df.to_csv(output_path, index=False)

        # Informasi dataset
        data_shape = bow_df.shape

        # Distribusi fitur BoW (10 fitur teratas)
        feature_sums = (
            bow_df.drop(columns=["Sentimen"])
            .sum(axis=0)
            .sort_values(ascending=False)[:10]
        )

        # Visualisasi distribusi fitur BoW (10 fitur teratas)
        feature_dist_path = os.path.join(
            "static", "img", "tweet_8_bow_feature_distribution.png"
        )
        plt.figure(figsize=(16, 9))
        feature_sums.plot(kind="barh", color="skyblue", edgecolor="black")
        plt.xlabel("Total Frekuensi")
        plt.ylabel("Fitur")
        plt.title("10 Fitur Teratas Berdasarkan Bag of Words")
        plt.gca().invert_yaxis()  # Balikkan urutan agar fitur dengan frekuensi terbesar di atas
        plt.savefig(feature_dist_path, bbox_inches="tight", facecolor="white")
        plt.close()

        return render_template(
            "18_tfidf_dataset.html",
            title="Representasi Fitur",
            data_shape=data_shape,
            file_details=processed_files_list,
            selected_file=selected_file,
            download_link=url_for("download_file", filename=output_file),
            feature_dist_path=feature_dist_path,
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "danger")
        return render_template(
            "18_tfidf_dataset.html",
            file_details=processed_files_list,
            title="Representasi Fitur",
        )


@app.route("/naive-bayes-model", methods=["GET", "POST"])
def naive_bayes_model():
    try:
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import (
            classification_report,
            accuracy_score,
            confusion_matrix,
            ConfusionMatrixDisplay,
        )
        from sklearn.impute import SimpleImputer
        import joblib

        # Ambil daftar file di direktori processed
        processed_files_list = os.listdir(
            os.path.join(os.getcwd(), "data", "processed")
        )
        processed_files_list = [f for f in processed_files_list if allowed_file(f)]

        train_file = "dataset_9_train.csv"
        test_file = "dataset_9_test.csv"
        train_path = os.path.join(os.getcwd(), "data", "processed", train_file)
        test_path = os.path.join(os.getcwd(), "data", "processed", test_file)

        # Pastikan file tersedia
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            flash(
                "Data latih atau data uji tidak ditemukan. Silakan lakukan pemisahan data terlebih dahulu.",
                "danger",
            )
            return render_template(
                "21_model_naive_bayes.html",
                file_details=processed_files_list,
                title="Model Naive Bayes",
            )

        # Membaca dataset
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        # Pastikan kolom 'Sentimen' ada
        if "Sentimen" not in train_data.columns or "Sentimen" not in test_data.columns:
            flash("Kolom 'Sentimen' tidak ditemukan dalam dataset.", "danger")
            return render_template(
                "21_model_naive_bayes.html",
                file_details=processed_files_list,
                title="Model Naive Bayes",
            )

        # Pisahkan fitur dan label
        X_train = train_data.drop(columns=["Sentimen"])
        y_train = train_data["Sentimen"]
        X_test = test_data.drop(columns=["Sentimen"])
        y_test = test_data["Sentimen"]

        # Periksa dan tangani nilai NaN di data latih dan uji
        if y_train.isnull().any() or y_test.isnull().any():
            y_train = y_train.fillna(0)  # Mengisi nilai NaN dengan 0
            y_test = y_test.fillna(0)

        if X_train.isnull().any().any() or X_test.isnull().any().any():
            imputer = SimpleImputer(strategy="constant", fill_value=0)
            X_train = pd.DataFrame(
                imputer.fit_transform(X_train), columns=X_train.columns
            )
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        # Inisialisasi dan latih model Naive Bayes
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Prediksi pada data uji
        y_pred = model.predict(X_test)

        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Negatif", "Netral", "Positif"],
            output_dict=True,
        )

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
        cm_path = os.path.join(
            "static", "img", "model_1_naive_bayes_confusion_matrix.png"
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Negatif", "Netral", "Positif"]
        )
        fig, ax = plt.subplots(figsize=(16, 9))  # Atur ukuran gambar
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        plt.title("Confusion Matrix - Naive Bayes")
        plt.savefig(cm_path, bbox_inches="tight", facecolor="white")
        plt.close()

        # Simpan model
        model_path = os.path.join(
            os.getcwd(), "data", "modeled", "model_1_naive_bayes.pkl"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        # Simpan hasil evaluasi ke dalam file CSV
        eval_file = os.path.join(
            os.getcwd(), "data", "processed", "model_0_calculated.csv"
        )
        os.makedirs(os.path.dirname(eval_file), exist_ok=True)

        eval_data = []
        for label, metrics in report.items():
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                eval_data.append(
                    {
                        "Model": "Naive Bayes",
                        "Kelas": label,
                        "Akurasi": accuracy,
                        "Presisi": metrics["precision"],
                        "Recall": metrics["recall"],
                        "F1-Score": metrics["f1-score"],
                        "Support": metrics["support"],
                    }
                )

        # Simpan ke dalam CSV
        if not os.path.exists(eval_file):
            pd.DataFrame(eval_data).to_csv(eval_file, index=False)
        else:
            pd.DataFrame(eval_data).to_csv(
                eval_file, mode="a", header=False, index=False
            )

        return render_template(
            "21_model_naive_bayes.html",
            title="Model Naive Bayes",
            accuracy=accuracy,
            cm_path=cm_path,
            file_details=processed_files_list,
            train_file=train_file,
            test_file=test_file,
            model_download_link=url_for(
                "download_file", filename="model_1_naive_bayes.pkl"
            ),
            report=report,
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "danger")
        return render_template(
            "21_model_naive_bayes.html",
            file_details=processed_files_list,
            title="Model Naive Bayes",
        )


@app.route("/svm-model", methods=["GET", "POST"])
def svm_model():
    try:
        from sklearn.svm import SVC
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            accuracy_score,
            ConfusionMatrixDisplay,
        )
        import seaborn as sns
        import joblib

        # Ambil daftar file di direktori processed
        processed_files_list = os.listdir(
            os.path.join(os.getcwd(), "data", "processed")
        )
        processed_files_list = [f for f in processed_files_list if allowed_file(f)]

        train_file = "dataset_9_train.csv"
        test_file = "dataset_9_test.csv"
        train_path = os.path.join(os.getcwd(), "data", "processed", train_file)
        test_path = os.path.join(os.getcwd(), "data", "processed", test_file)

        # Membaca dataset latih dan uji
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        # Pisahkan fitur dan label
        X_train = train_data.drop(columns=["Sentimen"])
        y_train = train_data["Sentimen"]
        X_test = test_data.drop(columns=["Sentimen"])
        y_test = test_data["Sentimen"]

        # Tangani nilai NaN pada fitur
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_test.mean())

        # Tangani nilai NaN pada label
        if y_train.isnull().values.any() or y_test.isnull().values.any():
            # flash("Label latih atau uji mengandung nilai NaN. Baris dengan NaN akan dihapus.", "warning")
            valid_train_indices = ~y_train.isnull()
            X_train = X_train[valid_train_indices]
            y_train = y_train[valid_train_indices]

            valid_test_indices = ~y_test.isnull()
            X_test = X_test[valid_test_indices]
            y_test = y_test[valid_test_indices]

        # Validasi setelah imputasi
        assert not X_train.isnull().values.any(), "Masih ada NaN pada X_train"
        assert not X_test.isnull().values.any(), "Masih ada NaN pada X_test"
        assert not y_train.isnull().values.any(), "Masih ada NaN pada y_train"
        assert not y_test.isnull().values.any(), "Masih ada NaN pada y_test"

        # Inisialisasi dan pelatihan model SVM
        model = SVC(kernel="linear", random_state=42)
        model.fit(X_train, y_train)

        # Prediksi pada data uji
        y_pred = model.predict(X_test)

        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Negatif", "Netral", "Positif"],
            output_dict=True,
        )
        confusion = confusion_matrix(y_test, y_pred)

        # Simpan confusion matrix sebagai gambar
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
        cm_path = os.path.join("static", "img", "model_2_svm_confusion_matrix.png")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Negatif", "Netral", "Positif"]
        )
        fig, ax = plt.subplots(figsize=(16, 9))  # Atur ukuran gambar
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        plt.title("Confusion Matrix - Support Vector Machine")
        plt.savefig(cm_path, bbox_inches="tight", facecolor="white")
        plt.close()

        # Simpan model
        model_path = os.path.join(os.getcwd(), "data", "modeled", "model_2_svm.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        # Tambahkan hasil evaluasi ke file model_0_calculated.csv
        csv_path = os.path.join(
            os.getcwd(), "data", "processed", "model_0_calculated.csv"
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        results = []
        for label, metrics in report.items():
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                results.append(
                    {
                        "Model": "SVM",
                        "Kelas": label,
                        "Akurasi": accuracy,
                        "Presisi": metrics["precision"],
                        "Recall": metrics["recall"],
                        "F1-Score": metrics["f1-score"],
                        "Support": metrics["support"],
                    }
                )

        # Simpan ke file CSV
        if not os.path.exists(csv_path):
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
        else:
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
            df.to_csv(csv_path, index=False)

        return render_template(
            "22_model_svm.html",
            title="Model SVM",
            accuracy=accuracy,
            cm_path=cm_path,
            file_details=processed_files_list,
            train_file=train_file,
            test_file=test_file,
            model_download_link=url_for("download_file", filename="model_2_svm.pkl"),
            report=report,
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "danger")
        return render_template(
            "22_model_svm.html", file_details=processed_files_list, title="Model SVM"
        )


@app.route("/evaluation-model", methods=["GET", "POST"])
def evaluation_model():
    try:
        # Membaca file CSV dengan hasil evaluasi
        file_path = os.path.join(
            os.getcwd(), "data", "processed", "model_0_calculated.csv"
        )
        if not os.path.exists(file_path):
            flash(
                "File evaluasi tidak ditemukan. Pastikan file model_0_calculated.csv sudah dibuat.",
                "danger",
            )
            return render_template("23_model_evaluation.html", title="Evaluasi Model")

        # Membaca file model_0_calculated.csv
        data = pd.read_csv(file_path)

        # Menambahkan visualisasi
        # Ringkasan Akurasi Per Model
        accuracy_summary = data.groupby("Model")["Akurasi"].mean().reset_index()

        # Bar Chart Perbandingan Akurasi
        accuracy_chart_path = os.path.join(
            "static", "img", "evaluation_1_accuracy_comparison.png"
        )
        plt.figure(figsize=(16, 9))
        sns.barplot(x="Model", y="Akurasi", data=accuracy_summary, palette="viridis")
        plt.title("Perbandingan Akurasi Antar Model")
        plt.xlabel("Model")
        plt.ylabel("Akurasi")
        plt.ylim(0, 1)
        plt.savefig(accuracy_chart_path, bbox_inches="tight", facecolor="white")
        plt.close()

        # Bar Chart Performa Per Kelas untuk Naive Bayes
        naive_bayes_data = data[data["Model"] == "Naive Bayes"]
        nb_performance_path = os.path.join(
            "static", "img", "evaluation_1_naive_bayes_performance.png"
        )
        naive_bayes_data.set_index("Kelas")[["Presisi", "Recall", "F1-Score"]].plot(
            kind="bar", figsize=(16, 9), color=["skyblue", "orange", "green"]
        )
        plt.title("Performa Naive Bayes Per Kelas")
        plt.ylabel("Nilai")
        plt.ylim(0, 1)
        plt.savefig(nb_performance_path, bbox_inches="tight", facecolor="white")
        plt.close()

        # Bar Chart Performa Per Kelas untuk SVM
        svm_data = data[data["Model"] == "SVM"]
        svm_performance_path = os.path.join(
            "static", "img", "evaluation_1_svm_performance.png"
        )
        svm_data.set_index("Kelas")[["Presisi", "Recall", "F1-Score"]].plot(
            kind="bar", figsize=(16, 9), color=["skyblue", "orange", "green"]
        )
        plt.title("Performa SVM Per Kelas")
        plt.ylabel("Nilai")
        plt.ylim(0, 1)
        plt.savefig(svm_performance_path, bbox_inches="tight", facecolor="white")
        plt.close()

        return render_template(
            "23_model_evaluation.html",
            title="Evaluasi Model",
            accuracy_chart_path=accuracy_chart_path,
            nb_performance_path=nb_performance_path,
            svm_performance_path=svm_performance_path,
            naive_bayes_data=naive_bayes_data.to_dict("records"),
            svm_data=svm_data.to_dict("records"),
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "danger")
        return render_template("23_model_evaluation.html", title="Evaluasi Model")


@app.route("/compare-model", methods=["GET", "POST"])
def compare_model():
    try:
        # Path ke file hasil evaluasi model
        evaluation_file_path = os.path.join(
            os.getcwd(), "data", "processed", "model_0_calculated.csv"
        )

        # Pastikan file evaluasi tersedia
        if not os.path.exists(evaluation_file_path):
            flash(
                "File evaluasi model tidak ditemukan. Pastikan model sudah dievaluasi.",
                "danger",
            )
            return render_template("24_model_compare.html", title="Perbandingan Model")

        # Membaca file evaluasi
        evaluation_data = pd.read_csv(evaluation_file_path)

        # Rata-rata metrik untuk setiap model
        avg_metrics = (
            evaluation_data.groupby("Model")
            .agg(
                {
                    "Akurasi": "mean",
                    "Presisi": "mean",
                    "Recall": "mean",
                    "F1-Score": "mean",
                }
            )
            .reset_index()
        )

        # Simpan grafik perbandingan akurasi antar model
        accuracy_plot_path = os.path.join(
            "static", "img", "compare_1_models_accuracy.png"
        )
        plt.figure(figsize=(16, 9))
        sns.barplot(x="Model", y="Akurasi", data=avg_metrics, palette="viridis")
        plt.title("Perbandingan Akurasi Antar Model")
        plt.ylabel("Akurasi")
        plt.xlabel("Model")
        plt.ylim(0, 1)
        plt.savefig(accuracy_plot_path, bbox_inches="tight", facecolor="white")
        plt.close()

        # Simpan grafik perbandingan rata-rata presisi, recall, dan F1-Score
        metrics_plot_path = os.path.join(
            "static", "img", "compare_1_models_metrics.png"
        )
        melted_metrics = avg_metrics.melt(
            id_vars="Model",
            value_vars=["Presisi", "Recall", "F1-Score"],
            var_name="Metrik",
            value_name="Nilai",
        )
        plt.figure(figsize=(16, 9))
        sns.barplot(
            x="Model", y="Nilai", hue="Metrik", data=melted_metrics, palette="muted"
        )
        plt.title("Perbandingan Rata-rata Presisi, Recall, dan F1-Score")
        plt.ylabel("Nilai")
        plt.xlabel("Model")
        plt.ylim(0, 1)
        plt.legend(title="Metrik")
        plt.savefig(metrics_plot_path, bbox_inches="tight", facecolor="white")
        plt.close()

        # Menyiapkan data untuk tabel perbandingan
        comparison_table = avg_metrics.to_dict(orient="records")

        return render_template(
            "24_model_compare.html",
            title="Perbandingan Model",
            accuracy_plot_path=accuracy_plot_path,
            metrics_plot_path=metrics_plot_path,
            comparison_table=comparison_table,
        )

    except Exception as e:
        flash(f"Terjadi kesalahan: {e}", "danger")
        return render_template("24_model_compare.html", title="Perbandingan Model")


@app.route("/evaluasi")
def evaluasi():
    return render_template("evaluasi.html", title="Tentang Aplikasi")


@app.route("/analisis-hasil")
def analisis_hasil():
    return render_template("analisis_hasil.html", title="Tentang Aplikasi")


# ! Baru


# 🔹 Fungsi untuk Halaman Data Eksplorasi
@app.route("/data-exploration")
def data_exploration():
    uploaded_filename = "dataset_0_raw.csv"
    dataset_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_filename)

    # Pastikan variabel ini hanya True atau False
    dataset_uploaded = os.path.exists(dataset_path)
    sentiment_counts = {"positif": 0, "negatif": 0, "netral": 0}  # Default

    if dataset_uploaded:
        try:
            data = pd.read_csv(dataset_path)

            # Cek apakah kolom Sentimen ada
            if "Sentimen" in data.columns:
                sentiment_counts = data["Sentimen"].value_counts().to_dict()

            # Informasi dataset
            data_shape = data.shape
            duplicate_count = data.duplicated().sum()
            null_count = data.isnull().sum().sum()

            # Hitung jumlah unik
            data_unique = data.nunique().to_frame(name="Unique Values").reset_index()
            data_unique.rename(columns={"index": "Column"}, inplace=True)
            data_unique_html = data_unique.to_html(
                classes="table table-striped", index=False
            )

            # Ringkasan dataset
            data_head = data.head().to_html(classes="table table-striped", index=False)

            # Statistik Deskriptif
            data_description = (
                data.describe().round(2).to_html(classes="table table-striped")
            )

            # Deteksi elemen yang perlu dibersihkan
            empty_tweets = data["Tweet"].str.strip().eq("").sum()
            emoji_tweets = (
                data["Tweet"].apply(lambda x: bool(re.search(r"[^\w\s]", str(x)))).sum()
            )
            links = data["Tweet"].str.contains("http|www", na=False).sum()
            symbols = data["Tweet"].str.contains(r"[^\w\s]", na=False).sum()
            only_numbers = data["Tweet"].str.match(r"^\d+$", na=False).sum()
            tweets_with_numbers = data["Tweet"].str.contains(r"\d", na=False).sum()
            short_tweets = (
                data["Tweet"].apply(lambda x: len(str(x).split())) < 3
            ).sum()

        except Exception as e:
            flash(f"Terjadi kesalahan dalam membaca dataset: {e}", "danger")
            return redirect(url_for("data_exploration"))

    return render_template(
        "data_exploration.html",
        title="Data Eksplorasi",
        uploaded_filename=uploaded_filename,
        dataset_uploaded=dataset_uploaded,
        data_shape=data_shape if dataset_uploaded else None,
        duplicate_count=duplicate_count if dataset_uploaded else None,
        null_count=null_count if dataset_uploaded else None,
        data_unique=data_unique_html if dataset_uploaded else None,
        data_head=data_head if dataset_uploaded else None,
        data_description=data_description if dataset_uploaded else None,
        empty_tweets=empty_tweets if dataset_uploaded else None,
        emoji_tweets=emoji_tweets if dataset_uploaded else None,
        links=links if dataset_uploaded else None,
        symbols=symbols if dataset_uploaded else None,
        only_numbers=only_numbers if dataset_uploaded else None,
        tweets_with_numbers=tweets_with_numbers if dataset_uploaded else None,
        short_tweets=short_tweets if dataset_uploaded else None,
        sentiment_counts=sentiment_counts if dataset_uploaded else None,
        sentiment_chart_path=url_for("static", filename="img/tweet_0_sentiment_distribution.png") if dataset_uploaded else None,
        wordcloud_path=url_for("static", filename="img/tweet_0_wordcloud.png") if dataset_uploaded else None,
        chart_path=url_for("static", filename="img/tweet_0_length_distribution.png") if dataset_uploaded else None,
    )


# 🔹 Fungsi untuk Mengunggah Dataset
@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        flash("Tidak ada file yang diunggah.", "danger")
        return redirect(url_for("data_exploration"))

    file = request.files["file"]
    if file.filename == "":
        flash("Pilih file terlebih dahulu.", "danger")
        return redirect(url_for("data_exploration"))

    if file and allowed_file(file.filename):
        filename = "dataset_0_raw.csv"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            # Jika file bukan CSV, konversi ke CSV
            if file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
                try:
                    data = pd.read_excel(filepath, engine="openpyxl")
                except Exception as e1:
                    try:
                        data = pd.read_excel(filepath, engine="xlrd")
                    except Exception as e2:
                        flash(f"Error membaca file Excel: {str(e2)}", "danger")
                        return redirect(url_for("data_exploration"))

                data.to_csv(filepath, index=False, sep=",")
            else:
                # Validasi delimiter CSV
                with open(filepath, "r") as f:
                    sample = f.read(1024)
                    try:
                        detected_delimiter = csv.Sniffer().sniff(sample).delimiter
                    except csv.Error:
                        detected_delimiter = ","  # Default fallback delimiter

                data = pd.read_csv(filepath, delimiter=detected_delimiter)
                if detected_delimiter != ",":
                    data.to_csv(filepath, index=False, sep=",")

        except Exception as e:
            flash(f"Error saat membaca atau mengonversi file: {str(e)}", "danger")
            return redirect(url_for("data_exploration"))

        # Normalisasi Nama Kolom
        if "full_text" in data.columns:
            data.rename(columns={"full_text": "Tweet"}, inplace=True)

        if "Tweet" not in data.columns:
            flash("Kolom 'Tweet' tidak ditemukan dalam dataset!", "danger")
            return redirect(url_for("data_exploration"))

        # Tambahkan Kolom Panjang Tweet
        data["Tweet Length"] = data["Tweet"].apply(lambda x: len(str(x).split()))
        data.to_csv(filepath, index=False)

        # 🔹 **Buat dan Simpan Visualisasi Gambar**
        try:
            # **Distribusi Sentimen**
            sentiment_chart_path = os.path.join(app.config["STATIC_FOLDER"], "tweet_0_sentiment_distribution.png"
            )
            if os.path.exists(sentiment_chart_path):
                os.remove(sentiment_chart_path)

            plt.figure(figsize=(16, 9))
            sentiment_counts = data["Sentimen"].value_counts().to_dict()
            plt.bar(
                sentiment_counts.keys(),
                sentiment_counts.values(),
                color=["green", "red", "blue"],
            )
            plt.xlabel("Sentimen")
            plt.ylabel("Jumlah")
            plt.title("Distribusi Sentimen")
            plt.savefig(sentiment_chart_path, bbox_inches="tight", facecolor="white")
            plt.close()

            # **Distribusi Panjang Tweet**
            chart_path = os.path.join(app.config["STATIC_FOLDER"], "tweet_0_length_distribution.png"
            )
            if os.path.exists(chart_path):
                os.remove(chart_path)

            plt.figure(figsize=(16, 9))
            data["Tweet Length"].hist(bins=30, color="blue", edgecolor="black")
            plt.xlabel("Jumlah Kata")
            plt.ylabel("Jumlah Tweet")
            plt.title("Distribusi Panjang Tweet")
            plt.savefig(chart_path, bbox_inches="tight", facecolor="white")
            plt.close()

            # **WordCloud**
            wordcloud_path = os.path.join(app.config["STATIC_FOLDER"], "tweet_0_wordcloud.png")
            if os.path.exists(wordcloud_path):
                os.remove(wordcloud_path)

            text = " ".join(data["Tweet"].dropna())
            wordcloud = WordCloud(
                width=1280, height=720, background_color="white"
            ).generate(text)
            plt.figure(figsize=(16, 9))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(wordcloud_path, bbox_inches="tight", facecolor="white")
            plt.close()

        except Exception as e:
            flash(f"Error saat membuat visualisasi: {str(e)}", "error")
            return redirect(url_for("data_exploration"))

        flash("Dataset berhasil diunggah!", "success")
        return redirect(url_for("data_exploration"))

    flash("Format file tidak didukung! Hanya CSV, XLSX, dan XLS.", "danger")
    return redirect(url_for("data_exploration"))


# 🔹 Route untuk Menampilkan Status Preprocessing
@app.route("/pre-processing")
def preprocessing():
    try:
        raw_file = "dataset_0_raw.csv"
        raw_path = os.path.join(app.config["UPLOAD_FOLDER"], raw_file)

        # Daftar file hasil preprocessing
        processed_files = {
            "Pembersihan": "dataset_1_cleaned.csv",
            "Normalisasi": "dataset_2_normalized.csv",
            "Tokenisasi": "dataset_3_tokenized.csv",
            "No Stopwords": "dataset_4_no_stopwords.csv",
            "Stemming": "dataset_5_stemmed.csv",
            "Label Encoding": "dataset_6_encoded.csv",
            "Pembagian": "dataset_7_train.csv",
        }

        # * Tampilkan ini jika perlu
        # Cek apakah semua file hasil preprocessing ada dan tidak kosong
        # missing_files = []
        # for step, filename in processed_files.items():
        #     file_path = os.path.join(PROCESSED_FOLDER, filename)

        #     # Cek apakah file ada dan tidak kosong
        #     if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        #         missing_files.append(f"{step} ({filename})")

        # if missing_files:
        #     flash(f"Belum melakukan proses pra-pemrosesan data.", "danger")

        # Cek apakah setiap file hasil preprocessing ada
        preprocessing_status = {}
        try:
            preprocessing_status = {
                step: os.path.exists(os.path.join(PROCESSED_FOLDER, filename))
                for step, filename in processed_files.items()
            }
        except Exception as e:
            flash(f"❌ Kesalahan dalam mengecek status preprocessing: {e}", "danger")
            preprocessing_status = {
                step: False for step in processed_files.keys()
            }  # Inisialisasi dengan False

        # Ambil nama file yang tersedia atau tandai sebagai "Belum tersedia"
        preprocessing_files = {
            step: filename if preprocessing_status[step] else "Belum tersedia"
            for step, filename in processed_files.items()
        }

        # 🛠 Inisialisasi variabel default (mencegah "referenced before assignment")
        data_shape_raw = None
        data_shape_cleaned = None
        data_head_cleaned = None
        data_description_cleaned = None
        duplicate_count_cleaned = None
        null_count_cleaned = None
        cleaning_methods = None
        comparison_table_cleaned = None
        comparison_samples_cleaned = []
        chart_path_cleaned = None
        wordcloud_path_cleaned = None
        download_link_cleaned = None
        data_count_cleaned = None
        data_count_normalized = None
        comparison_samples_normalized = []
        data_shape_normalized = None
        data_head_normalized = None
        data_description_normalized = None
        chart_path_normalized = None
        wordcloud_path_normalized = None
        download_link_normalized = None
        data_count_tokenized = None
        comparison_samples_tokenized = []
        data_shape_tokenized = None
        data_head_tokenized = None
        data_description_tokenized = None
        chart_path_tokenized = None
        wordcloud_path_tokenized = None
        download_link_tokenized = None
        data_count_no_stopwords = None
        total_no_stopwords = None
        comparison_samples_no_stopwords = []
        data_shape_no_stopwords = None
        data_head_no_stopwords = None
        data_description_no_stopwords = None
        chart_path_no_stopwords = None
        wordcloud_path_no_stopwords = None
        download_link_no_stopwords = None
        data_count_stemmed = None
        total_stemmed = None
        comparison_samples_stemmed = []
        data_shape_stemmed = None
        data_head_stemmed = None
        data_description_stemmed = None
        chart_path_stemmed = None
        wordcloud_path_stemmed = None
        download_link_stemmed = None
        sentiment_encoded = None
        sentiment_stemmed = None
        comparison_samples_encoded = []
        sentiment_count_encoded = []
        data_shape_encoded = None
        data_head_encoded = None
        data_description_encoded = None
        chart_path_encoded = None
        download_link_encoded = None
        train_file = None
        test_file = None
        train_label_split = []
        test_label_split = []
        comparison_split = None
        chart_train_split = None
        chart_test_split = None
        data_shape_train = None
        data_shape_test = None
        data_head_train = None
        data_head_test = None
        data_description_train = None
        data_description_test = None
        data_distribution_split = []
        chart_path_split = None
        download_link_train = None
        download_link_test = None

        if os.path.exists(raw_path):
            data_raw = pd.read_csv(raw_path)
            data_shape_raw = data_raw.shape

        # **📌 1️⃣ Pembersihan Data**
        if preprocessing_status["Pembersihan"]:
            try:
                cleaned_file = "dataset_1_cleaned.csv"
                cleaned_path = os.path.join(
                    app.config["PROCESSED_FOLDER"], cleaned_file
                )

                if (
                    not os.path.exists(cleaned_path)
                    and not os.stat(cleaned_path).st_size > 0
                ):
                    flash("File hasil perbersihan data belum tersedia.", "danger")
                elif not os.path.exists(raw_path) and not os.stat(raw_path).st_size > 0:
                    flash("File mentah belum tersedia.", "danger")
                else:
                    data_cleaned = pd.read_csv(cleaned_path)

                    # **📊 Informasi Dataset**
                    data_shape_cleaned = data_cleaned.shape
                    duplicate_count_cleaned = data_cleaned.duplicated().sum()
                    null_count_cleaned = data_cleaned.isnull().sum().sum()
                    data_head_cleaned = data_cleaned.head().to_html(
                        classes="table table-striped", index=False
                    )
                    data_description_cleaned = (
                        data_cleaned.describe()
                        .round(2)
                        .to_html(classes="table table-striped")
                    )

                    # Metode pembersihan
                    cleaning_methods = [
                        "Menghapus mention (@username)",
                        "Menghapus hashtag (#hashtag)",
                        "Menghapus karakter unik selain huruf, angka, dan spasi",
                        "Menghapus emoji",
                        "Menghapus tautan (URL)",
                        "Menghapus tag HTML dan entitas",
                        "Menghapus simbol atau karakter khusus",
                        "Menghapus underscore (_) dalam teks",
                        "Menghapus duplikat pada kolom 'Tweet'",
                        "Menghapus nilai kosong pada kolom 'Tweet'",
                    ]

                    if cleaning_methods is None:
                        cleaning_methods = []

                    # Perbandingan sebelum dan sesudah pembersihan
                    comparison_table_cleaned = f"""
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Langkah</th>
                                <th>Jumlah Data</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Sebelum Pembersihan</td>
                                <td>{data_shape_raw[0] if data_shape_raw else "Tidak tersedia"}</td>
                            </tr>
                            <tr>
                                <td>Setelah Pembersihan</td>
                                <td>{data_shape_cleaned[0]}</td>
                            </tr>
                        </tbody>
                    </table>
                    """

                    # Ambil beberapa contoh sebelum dan sesudah pembersihan
                    comparison_samples_cleaned = []
                    for i in range(min(5, len(data_cleaned))):  # Ambil 5 contoh
                        comparison_samples_cleaned.append(
                            {
                                "Sebelum": (
                                    data_raw.iloc[i]["Tweet"]
                                    if i < len(data_raw)
                                    else "-"
                                ),
                                "Sesudah": data_cleaned.iloc[i]["Tweet"],
                            }
                        )

                    # **Download File**
                    download_link_cleaned = url_for(
                        "download_file", filename=cleaned_file
                    )

            except Exception as e:
                flash(f"❌ Kesalahan pada Pembersihan Data: {e}", "danger")

        # **📌 2️⃣ Normalisasi Data**
        if preprocessing_status["Normalisasi"]:
            try:
                normalized_file = "dataset_2_normalized.csv"
                cleaned_file = "dataset_1_cleaned.csv"
                normalized_path = os.path.join(
                    app.config["PROCESSED_FOLDER"], normalized_file
                )
                cleaned_path = os.path.join(
                    app.config["PROCESSED_FOLDER"], cleaned_file
                )

                if not os.path.exists(normalized_path):
                    flash("File hasil normalisasi belum tersedia.", "danger")
                elif not os.path.exists(cleaned_path):
                    flash("File hasil normalisasi belum tersedia.", "danger")
                else:
                    data_normalized = pd.read_csv(normalized_path)
                    data_cleaned = pd.read_csv(cleaned_path)

                    # Jumlah data sebelum dan sesudah normalisasi
                    data_count_cleaned = len(data_cleaned)
                    data_count_normalized = len(data_normalized)

                    # Ambil contoh sebelum dan sesudah normalisasi
                    comparison_samples_normalized = []
                    for i in range(min(5, len(data_normalized))):  # Ambil 5 contoh
                        comparison_samples_normalized.append(
                            {
                                "Sebelum": (
                                    data_cleaned.iloc[i]["Tweet"]
                                    if i < len(data_cleaned)
                                    else "-"
                                ),
                                "Sesudah": data_normalized.iloc[i]["Tweet"],
                            }
                        )

                    # **📊 Informasi Dataset**
                    data_shape_normalized = data_normalized.shape
                    data_head_normalized = data_normalized.head().to_html(
                        classes="table table-striped", index=False
                    )
                    data_description_normalized = (
                        data_normalized.describe()
                        .round(2)
                        .to_html(classes="table table-striped")
                    )

                    # **Download File**
                    download_link_normalized = url_for(
                        "download_file", filename=normalized_file
                    )

            except Exception as e:
                flash(f"❌ Kesalahan pada Normalisasi Data: {e}", "danger")

        # **📌 3️⃣ Tokenisasi Data**
        if preprocessing_status["Tokenisasi"]:
            try:
                tokenized_file = "dataset_3_tokenized.csv"
                normalized_file = "dataset_2_normalized.csv"
                tokenized_path = os.path.join(
                    app.config["PROCESSED_FOLDER"], tokenized_file
                )
                normalized_path = os.path.join(
                    app.config["PROCESSED_FOLDER"], normalized_file
                )

                if not os.path.exists(tokenized_path):
                    flash("File hasil tokenisasi data belum tersedia.", "danger")
                elif not os.path.exists(normalized_path):
                    flash("File hasil normalisasi data belum tersedia.", "danger")
                else:
                    # Baca dataset hasil tokenisasi
                    data_tokenized = pd.read_csv(tokenized_path)
                    data_normalized = pd.read_csv(normalized_path)

                    # Perbandingan jumlah data sebelum & sesudah tokenisasi
                    data_count_normalized = len(data_normalized)
                    data_count_tokenized = len(data_tokenized)

                    # Contoh sebelum & sesudah tokenisasi
                    comparison_samples_tokenized = []
                    for i in range(min(5, len(data_tokenized))):
                        comparison_samples_tokenized.append(
                            {
                                "Sebelum": (
                                    data_normalized.iloc[i]["Tweet"]
                                    if i < len(data_normalized)
                                    else "-"
                                ),
                                "Sesudah": data_tokenized.iloc[i]["Tokenized"],
                            }
                        )

                    # **📊 Informasi Dataset**
                    data_shape_tokenized = data_tokenized.shape
                    data_head_tokenized = data_tokenized.head().to_html(
                        classes="table table-striped", index=False
                    )
                    data_description_tokenized = (
                        data_tokenized.describe()
                        .round(2)
                        .to_html(classes="table table-striped")
                    )

                    # **Download File**
                    download_link_tokenized = url_for(
                        "download_file", filename=tokenized_file
                    )

            except Exception as e:
                flash(f"❌ Kesalahan pada Tokenisasi Data: {e}", "danger")

        # **📌 4️⃣ Penghapusan Stopwords**
        if preprocessing_status["No Stopwords"]:
            try:
                stopwords_file = "dataset_4_no_stopwords.csv"
                tokenized_file = "dataset_3_tokenized.csv"
                stopwords_path = os.path.join(
                    app.config["PROCESSED_FOLDER"], stopwords_file
                )
                tokenized_path = os.path.join(
                    app.config["PROCESSED_FOLDER"], tokenized_file
                )

                if not os.path.exists(stopwords_path):
                    flash("File hasil penghapusan stopwords belum tersedia.", "danger")
                elif not os.path.exists(tokenized_path):
                    flash("File hasil tokenisasi belum tersedia.", "danger")
                else:
                    # Baca dataset hasil penghapusan stopwords
                    data_no_stopwords = pd.read_csv(stopwords_path)
                    data_tokenized = pd.read_csv(tokenized_path)

                    # Hitung total kata sebelum dan sesudah penghapusan stopwords
                    data_count_tokenized = (
                        data_tokenized["Tokenized"].apply(lambda x: len(eval(x))).sum()
                    )
                    data_count_no_stopwords = (
                        data_no_stopwords["Tokenized"]
                        .apply(lambda x: len(eval(x)))
                        .sum()
                    )
                    total_no_stopwords = data_count_tokenized - data_count_no_stopwords

                    # Contoh sebelum & sesudah penghapusan stopwords
                    comparison_samples_no_stopwords = []
                    for i in range(min(5, len(data_no_stopwords))):
                        comparison_samples_no_stopwords.append(
                            {
                                "Sebelum": (
                                    data_tokenized.iloc[i]["Tokenized"]
                                    if i < len(data_tokenized)
                                    else "-"
                                ),
                                "Sesudah": data_no_stopwords.iloc[i]["Tokenized"],
                            }
                        )

                    # **📊 Informasi Dataset**
                    data_shape_no_stopwords = data_no_stopwords.shape
                    data_head_no_stopwords = data_no_stopwords.head().to_html(
                        classes="table table-striped", index=False
                    )
                    data_description_no_stopwords = (
                        data_no_stopwords.describe()
                        .round(2)
                        .to_html(classes="table table-striped")
                    )

                    # **Download File**
                    download_link_no_stopwords = url_for(
                        "download_file", filename=stopwords_file
                    )

            except Exception as e:
                flash(f"❌ Kesalahan pada Penghapusan Stopwords: {e}", "danger")

        # **📌 5️⃣ Stemming Data**
        if preprocessing_status["Stemming"]:
            try:
                stemmed_file = "dataset_5_stemmed.csv"
                no_stopwords_file = "dataset_4_no_stopwords.csv"
                stemmed_path = os.path.join(
                    app.config["PROCESSED_FOLDER"], stemmed_file
                )
                no_stopwords_path = os.path.join(
                    app.config["PROCESSED_FOLDER"], no_stopwords_file
                )

                if not os.path.exists(stemmed_path):
                    flash("File hasil stemming belum tersedia.", "danger")
                elif not os.path.exists(no_stopwords_path):
                    flash("File hasil penghapusan stopwords belum tersedia.", "danger")
                else:
                    # Baca dataset hasil stemming
                    data_stemmed = pd.read_csv(stemmed_path)
                    data_no_stopwords = pd.read_csv(no_stopwords_path)

                    # Jumlah kata sebelum dan sesudah stemming
                    data_count_no_stopwords = (
                        data_no_stopwords["Tokenized"].str.split().apply(len).sum()
                    )
                    data_count_stemmed = (
                        data_stemmed["Tokenized"].str.split().apply(len).sum()
                    )

                    # Hitung jumlah kata yang berubah setelah stemming
                    total_stemmed = data_count_no_stopwords - data_count_stemmed

                    # Contoh sebelum & sesudah stemming
                    comparison_samples_stemmed = []
                    for i in range(min(5, len(data_stemmed))):
                        comparison_samples_stemmed.append(
                            {
                                "Sebelum": (
                                    data_no_stopwords.iloc[i]["Tokenized"]
                                    if i < len(data_no_stopwords)
                                    else "-"
                                ),
                                "Sesudah": data_stemmed.iloc[i]["Tokenized"],
                            }
                        )

                    # **📊 Informasi Dataset**
                    data_shape_stemmed = data_stemmed.shape
                    data_head_stemmed = data_stemmed.head().to_html(
                        classes="table table-striped", index=False
                    )
                    data_description_stemmed = (
                        data_stemmed.describe()
                        .round(2)
                        .to_html(classes="table table-striped")
                    )

                    # **Download File**
                    download_link_stemmed = url_for(
                        "download_file", filename=stemmed_file
                    )

            except Exception as e:
                flash(f"❌ Kesalahan pada Stemming Data: {e}", "danger")

        # **📌 6️⃣ Label Encoding**
        if preprocessing_status["Label Encoding"]:
            try:
                encoded_file = "dataset_6_encoded.csv"
                stemmed_file = "dataset_5_stemmed.csv"
                encoded_path = os.path.join(
                    app.config["PROCESSED_FOLDER"], encoded_file
                )
                stemmed_path = os.path.join(
                    app.config["PROCESSED_FOLDER"], stemmed_file
                )

                # Pastikan file tersedia dan tidak kosong sebelum membaca
                if (
                    not os.path.exists(encoded_path)
                    or os.stat(encoded_path).st_size == 0
                ):
                    flash("File hasil Label Encoding belum tersedia.", "danger")
                    data_encoded = pd.DataFrame()  # Set sebagai DataFrame kosong
                else:
                    data_encoded = pd.read_csv(encoded_path)

                if (
                    not os.path.exists(stemmed_path)
                    or os.stat(stemmed_path).st_size == 0
                ):
                    flash("File hasil Stemming belum tersedia.", "danger")
                    data_stemmed = pd.DataFrame()  # Set sebagai DataFrame kosong
                else:
                    data_stemmed = pd.read_csv(stemmed_path)

                # **🔹 Cek apakah data terbaca dengan benar**
                if not isinstance(data_encoded, pd.DataFrame):
                    flash("❌ Data hasil Label Encoding bukan DataFrame!", "danger")
                    data_encoded = pd.DataFrame()

                if not isinstance(data_stemmed, pd.DataFrame):
                    flash("❌ Data hasil Stemming bukan DataFrame!", "danger")
                    data_stemmed = pd.DataFrame()

                # **🔹 Pastikan kolom yang dibutuhkan ada**
                required_columns = ["Tweet", "Sentimen", "Label_Encoded"]
                for col in required_columns:
                    if col not in data_encoded.columns:
                        flash(
                            f"❌ Kolom '{col}' tidak ditemukan dalam dataset Label Encoding!",
                            "danger",
                        )
                        data_encoded = pd.DataFrame()
                        break  # Hentikan eksekusi jika ada kolom yang hilang

                # **📊 Distribusi Label Sentimen**
                if not data_encoded.empty:
                    sentiment_count_encoded = (
                        data_encoded["Label_Encoded"].value_counts().to_dict()
                    )
                    sentiment_encoded = {
                        -1: sentiment_count_encoded.get(-1, 0),
                        0: sentiment_count_encoded.get(0, 0),
                        1: sentiment_count_encoded.get(1, 0),
                    }
                    sentiment_stemmed = [
                        f"Positif : {sentiment_count_encoded.get(1, 0)} sampel",
                        f"Netral  : {sentiment_count_encoded.get(0, 0)} sampel",
                        f"Negatif : {sentiment_count_encoded.get(-1, 0)} sampel",
                    ]
                else:
                    sentiment_encoded = {}
                    sentiment_stemmed = ["Data belum tersedia"]

                # **📌 Contoh sebelum & sesudah encoding**
                comparison_samples_encoded = []
                if not data_encoded.empty and not data_stemmed.empty:
                    for i in range(min(5, len(data_encoded))):
                        comparison_samples_encoded.append(
                            {
                                "Tweet": data_stemmed.iloc[i]["Tweet"],
                                "Sentimen": data_stemmed.iloc[i]["Sentimen"],
                                "Encoded": data_encoded.iloc[i]["Label_Encoded"],
                            }
                        )

                # **📊 Informasi Dataset**
                data_shape_encoded = (
                    data_encoded.shape if not data_encoded.empty else (0, 0)
                )
                data_head_encoded = (
                    data_encoded.head().to_html(
                        classes="table table-striped", index=False
                    )
                    if not data_encoded.empty
                    else "<p>Tidak ada data</p>"
                )
                data_description_encoded = (
                    data_encoded.describe()
                    .round(2)
                    .to_html(classes="table table-striped")
                    if not data_encoded.empty
                    else "<p>Tidak ada data statistik</p>"
                )

                # **Download File**
                download_link_encoded = (
                    url_for("download_file", filename=encoded_file)
                    if not data_encoded.empty
                    else None
                )

            except Exception as e:
                flash(f"❌ Kesalahan pada Label Encoding: {e}", "danger")

        # **📌 7️⃣ Pembagian Data**
        if preprocessing_status.get("Pembagian"):
            try:
                train_file = "dataset_7_train.csv"
                test_file = "dataset_7_test.csv"
                encoded_file = "dataset_6_encoded.csv"
                train_path = os.path.join(PROCESSED_FOLDER, train_file)
                test_path = os.path.join(PROCESSED_FOLDER, test_file)
                encoded_path = os.path.join(PROCESSED_FOLDER, encoded_file)

                if not os.path.exists(train_path):
                    flash("File hasil pembagian data train belum tersedia.", "danger")
                elif not os.path.exists(test_path):
                    flash("File hasil pembagian data test belum tersedia.", "danger")
                elif (not os.path.exists(encoded_path) or os.stat(encoded_path).st_size == 0):
                    flash("File hasil label encoding belum tersedia.", "danger")
                else:
                    data_train = pd.read_csv(train_path)
                    data_test = pd.read_csv(test_path)
                    data_encoded = pd.read_csv(encoded_path)

                    if not isinstance(data_encoded, pd.DataFrame):
                        raise ValueError(
                            "❌ data_encoded bukan DataFrame! Cek format file CSV."
                        )

                    # Hitung distribusi label dalam Training & Testing
                    train_label_dist = (
                        data_train["Label_Encoded"].value_counts().to_dict()
                    )
                    test_label_dist = (
                        data_test["Label_Encoded"].value_counts().to_dict()
                    )

                    # Tambahkan validasi untuk memastikan formatnya dictionary
                    if not isinstance(train_label_dist, dict):
                        raise ValueError(
                            "❌ train_label_dist bukan dictionary! Periksa data."
                        )

                    if not isinstance(test_label_dist, dict):
                        raise ValueError(
                            "❌ test_label_dist bukan dictionary! Periksa data."
                        )

                    train_label_split = [
                        f"-1 (Negatif): {train_label_dist.get(-1, 0)} sampel",
                        f"0 (Netral): {train_label_dist.get(0, 0)} sampel",
                        f"1 (Positif): {train_label_dist.get(1, 0)} sampel",
                    ]

                    test_label_split = [
                        f"-1 (Negatif): {test_label_dist.get(-1, 0)} sampel",
                        f"0 (Netral): {test_label_dist.get(0, 0)} sampel",
                        f"1 (Positif): {test_label_dist.get(1, 0)} sampel",
                    ]

                    # Buat dataframe untuk perbandingan jumlah data sebelum & sesudah pembagian
                    comparison_df = pd.DataFrame(
                        {
                            "Langkah": [
                                "Sebelum Pembagian",
                                "Data Training",
                                "Data Testing",
                            ],
                            "Jumlah Data": [
                                len(data_encoded),
                                len(data_train),
                                len(data_test),
                            ],
                            "Negatif": [
                                sentiment_count_encoded.get(-1, 0),
                                train_label_dist.get(-1, 0),
                                test_label_dist.get(-1, 0),
                            ],
                            "Netral": [
                                sentiment_count_encoded.get(0, 0),
                                train_label_dist.get(0, 0),
                                test_label_dist.get(0, 0),
                            ],
                            "Positif": [
                                sentiment_count_encoded.get(1, 0),
                                train_label_dist.get(1, 0),
                                test_label_dist.get(1, 0),
                            ],
                        }
                    )

                    # Tampilkan dalam bentuk HTML table
                    comparison_split = comparison_df.to_html(
                        classes="table table-striped", index=False
                    )

                    # Informasi dataset
                    data_shape_train = data_train.shape
                    data_shape_test = data_test.shape

                    data_head_train = data_train.head().to_html(
                        classes="table table-striped", index=False
                    )
                    data_head_test = data_test.head().to_html(
                        classes="table table-striped", index=False
                    )

                    data_description_train = (
                        data_train.describe()
                        .round(2)
                        .to_html(classes="table table-striped")
                    )
                    data_description_test = (
                        data_test.describe()
                        .round(2)
                        .to_html(classes="table table-striped")
                    )

                    # **📊 Distribusi Data Train dan Test**
                    train_size = len(data_train)
                    test_size = len(data_test)
                    total_size = train_size + test_size

                    train_percentage = round((train_size / total_size) * 100, 2)
                    test_percentage = round((test_size / total_size) * 100, 2)

                    # **📌 Buat List untuk Distribusi Data**
                    data_distribution_split = [
                        f"Train Data: {train_size} sampel ({train_percentage}%)",
                        f"Test Data: {test_size} sampel ({test_percentage}%)",
                    ]

                    download_link_train = url_for("download_file", filename=train_file)
                    download_link_test = url_for("download_file", filename=test_file)

            except Exception as e:
                flash(f"❌ Kesalahan pada Pembagian Data: {e}", "danger")

        return render_template(
            "pre_processing.html",
            title="Pra-Pemrosesan Data",
            data_shape_raw=data_shape_raw,
            # Pembersihan Data
            dataset_uploaded=os.path.exists(raw_path),
            preprocessing_status=preprocessing_status,
            preprocessing_files=preprocessing_files,
            data_shape_cleaned=data_shape_cleaned,
            data_head_cleaned=data_head_cleaned,
            data_description_cleaned=data_description_cleaned,
            duplicate_count_cleaned=duplicate_count_cleaned,
            null_count_cleaned=null_count_cleaned,
            cleaning_methods=cleaning_methods,
            comparison_table_cleaned=comparison_table_cleaned,
            comparison_samples_cleaned=comparison_samples_cleaned or [],
            chart_path_cleaned=url_for("static", filename="img/tweet_1_length_distribution_cleaned.png"),
            wordcloud_path_cleaned=url_for("static", filename="img/tweet_1_wordcloud_cleaned.png"),
            download_link_cleaned=download_link_cleaned,
            # Normalisasi Data
            data_count_cleaned=data_count_cleaned,
            data_count_normalized=data_count_normalized,
            comparison_samples_normalized=comparison_samples_normalized,
            data_shape_normalized=data_shape_normalized,
            data_head_normalized=data_head_normalized,
            data_description_normalized=data_description_normalized,
            chart_path_normalized=url_for("static", filename="img/tweet_2_length_distribution_normalized.png"),
            wordcloud_path_normalized=url_for("static", filename="img/tweet_2_wordcloud_normalized.png"),
            download_link_normalized=download_link_normalized,
            # Tokenisasi Data
            data_count_tokenized=data_count_tokenized,
            comparison_samples_tokenized=comparison_samples_tokenized,
            data_shape_tokenized=data_shape_tokenized,
            data_head_tokenized=data_head_tokenized,
            data_description_tokenized=data_description_tokenized,
            chart_path_tokenized=url_for("static", filename="img/tweet_3_length_distribution_tokenized.png"),
            wordcloud_path_tokenized=url_for("static", filename="img/tweet_3_wordcloud_tokenized.png"),
            download_link_tokenized=download_link_tokenized,
            # No Stopwords Data
            data_count_no_stopwords=data_count_no_stopwords,
            total_no_stopwords=total_no_stopwords,
            comparison_samples_no_stopwords=comparison_samples_no_stopwords,
            data_shape_no_stopwords=data_shape_no_stopwords,
            data_head_no_stopwords=data_head_no_stopwords,
            data_description_no_stopwords=data_description_no_stopwords,
            chart_path_no_stopwords=url_for("static", filename="img/tweet_4_length_distribution_no_stopwords.png"),
            wordcloud_path_no_stopwords=url_for("static", filename="img/tweet_4_wordcloud_no_stopwords.png"),
            download_link_no_stopwords=download_link_no_stopwords,
            # Stemming Data
            data_count_stemmed=data_count_stemmed,
            total_stemmed=total_stemmed,
            comparison_samples_stemmed=comparison_samples_stemmed,
            data_shape_stemmed=data_shape_stemmed,
            data_head_stemmed=data_head_stemmed,
            data_description_stemmed=data_description_stemmed,
            chart_path_stemmed=url_for("static", filename="img/tweet_5_length_distribution_stemmed.png"),
            wordcloud_path_stemmed=url_for("static", filename="img/tweet_5_wordcloud_stemmed.png"),
            download_link_stemmed=download_link_stemmed,
            # Label Encoding Data
            sentiment_encoded=sentiment_encoded,
            sentiment_stemmed=sentiment_stemmed,
            comparison_samples_encoded=comparison_samples_encoded,
            data_shape_encoded=data_shape_encoded,
            data_head_encoded=data_head_encoded,
            data_description_encoded=data_description_encoded,
            chart_path_encoded=url_for("static", filename="img/tweet_6_label_distribution_encoded.png"),
            download_link_encoded=download_link_encoded,
            # Pembagian Data
            train_file=train_file,
            test_file=test_file,
            train_label_split=train_label_split or [],
            test_label_split=test_label_split or [],
            comparison_split=comparison_split,
            chart_train_split=url_for("static", filename="img/tweet_7_train_split_distribution.png"),
            chart_test_split=url_for("static", filename="img/tweet_7_test_split_distribution.png"),
            data_shape_train=data_shape_train,
            data_shape_test=data_shape_test,
            data_head_train=data_head_train,
            data_head_test=data_head_test,
            data_description_train=data_description_train,
            data_description_test=data_description_test,
            data_distribution_split=data_distribution_split,
            chart_path_split=url_for("static", filename="img/tweet_7_split_data_distribution.png"),
            download_link_train=download_link_train,
            download_link_test=download_link_test,
            all=all,
        )

    except Exception as e:
        flash(f"Terjadi kesalahan dalam pra-pemrosesan: {e}", "danger")
        return render_template(
            "pre_processing.html",
            preprocessing_status={},
            preprocessing_files={},
            data_shape_raw=None,
            data_shape_cleaned=None,
            data_head_cleaned=None,
            data_description_cleaned=None,
            duplicate_count_cleaned=None,
            null_count_cleaned=None,
            cleaning_methods=None,
            comparison_table_cleaned=None,
            comparison_samples_cleaned=None,
            chart_path_cleaned=None,
            wordcloud_path_cleaned=None,
            download_link_cleaned=None,
            data_count_cleaned=None,
            data_count_normalized=None,
            comparison_samples_normalized=None,
            data_shape_normalized=None,
            data_head_normalized=None,
            data_description_normalized=None,
            chart_path_normalized=None,
            wordcloud_path_normalized=None,
            download_link_normalized=None,
            data_count_tokenized=None,
            comparison_samples_tokenized=None,
            data_shape_tokenized=None,
            data_head_tokenized=None,
            data_description_tokenized=None,
            chart_path_tokenized=None,
            wordcloud_path_tokenized=None,
            download_link_tokenized=None,
            data_count_no_stopwords=None,
            total_no_stopwords=None,
            comparison_samples_no_stopwords=None,
            data_shape_no_stopwords=None,
            data_head_no_stopwords=None,
            data_description_no_stopwords=None,
            chart_path_no_stopwords=None,
            wordcloud_path_no_stopwords=None,
            download_link_no_stopwords=None,
            data_count_stemmed=None,
            total_stemmed=None,
            comparison_samples_stemmed=None,
            data_shape_stemmed=None,
            data_head_stemmed=None,
            data_description_stemmed=None,
            chart_path_stemmed=None,
            wordcloud_path_stemmed=None,
            download_link_stemmed=None,
            sentiment_encoded=None,
            sentiment_stemmed=None,
            comparison_samples_encoded=None,
            data_shape_encoded=None,
            data_head_encoded=None,
            data_description_encoded=None,
            chart_path_encoded=None,
            download_link_encoded=None,
            train_file=None,
            test_file=None,
            train_label_split=[],
            test_label_split=[],
            comparison_split=None,
            chart_train_split=None,
            chart_test_split=None,
            data_shape_train=None,
            data_shape_test=None,
            data_head_train=None,
            data_head_test=None,
            data_description_train=None,
            data_description_test=None,
            data_distribution_split=None,
            chart_path_split=None,
            download_link_train=None,
            download_link_test=None,
            all=all,
        )


# 🔹 Fungsi utama untuk melakukan pra-pemrosesan
@app.route("/start-preprocessing", methods=["POST"])
def start_preprocessing():
    raw_file = "dataset_0_raw.csv"
    raw_path = os.path.join(app.config["UPLOAD_FOLDER"], raw_file)

    if not os.path.exists(raw_path):
        return jsonify({"success": False, "message": "Dataset belum diunggah."})

    try:
        # 📌 1️⃣ Pembersihan Data
        try:
            print("🚀 Memulai Pembersihan Data...")
            cleaned_path = os.path.join(app.config["PROCESSED_FOLDER"], "dataset_1_cleaned.csv")
            
            # **Hapus file lama jika ada**
            if os.path.exists(cleaned_path):
                os.remove(cleaned_path)
                
            data = pd.read_csv(raw_path)
            if "Tweet" not in data.columns:
                return jsonify(
                    {
                        "success": False,
                        "message": "Kolom 'Tweet' tidak ditemukan dalam dataset!",
                    }
                )

            data["Tweet"] = data["Tweet"].astype(str)
            data["Tweet"] = data["Tweet"].apply(lambda x: remove_mentions(x))
            data["Tweet"] = data["Tweet"].apply(lambda x: remove_hashtags(x))
            data["Tweet"] = data["Tweet"].apply(lambda x: remove_uniques(x))
            data["Tweet"] = data["Tweet"].apply(lambda x: remove_emoji(x))
            data["Tweet"] = data["Tweet"].apply(lambda x: remove_links(x))
            data["Tweet"] = data["Tweet"].apply(lambda x: remove_html_tags_and_entities(x))
            data["Tweet"] = data["Tweet"].apply(lambda x: remove_special_characters(x))
            data["Tweet"] = data["Tweet"].apply(lambda x: remove_underscores(x))
            data = data.drop_duplicates(subset=["Tweet"])
            data = data.dropna(subset=["Tweet"])
            
            # **Simpan hasil pembersihan**
            data.to_csv(cleaned_path, index=False)
            print("✅ Pembersihan Data Selesai.")
            
            # **🔹 Visualisasi: Distribusi Panjang Tweet**
            chart_path_cleaned = os.path.join(app.config["STATIC_FOLDER"], "tweet_1_length_distribution_cleaned.png")
            plt.figure(figsize=(16, 9))
            data["Tweet"].str.split().apply(len).plot(
                kind="hist",
                bins=30,
                color="blue",
                edgecolor="black",
                title="Distribusi Panjang Tweet Setelah Pembersihan"
            )
            plt.xlabel("Jumlah Kata")
            plt.ylabel("Frekuensi")
            plt.savefig(chart_path_cleaned, bbox_inches="tight", facecolor="white")
            plt.close()

            # **☁️ WordCloud**
            wordcloud_path_cleaned = os.path.join(app.config["STATIC_FOLDER"], "tweet_1_wordcloud_cleaned.png")
            text = " ".join(data["Tweet"].dropna())
            wordcloud = WordCloud(width=1280, height=720, background_color="white").generate(text)
            plt.figure(figsize=(16, 9))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(wordcloud_path_cleaned, bbox_inches="tight", facecolor="white")
            plt.close()

        except Exception as e:
            print(f"❌ Kesalahan pada Pembersihan Data: {e}")

        # 📌 2️⃣ Normalisasi Data
        try:
            print("🚀 Memulai Normalisasi Data...")
            normalized_path = os.path.join(app.config["PROCESSED_FOLDER"], "dataset_2_normalized.csv")

            # **Hapus file lama jika ada**
            if os.path.exists(normalized_path):
                os.remove(normalized_path)

            # **Load Data Pembersihan**
            cleaned_path = os.path.join(app.config["PROCESSED_FOLDER"], "dataset_1_cleaned.csv")
            data = pd.read_csv(cleaned_path)
            
            # (Proses normalisasi)
            data["Tweet"] = data["Tweet"].str.lower()
            data["Tweet"] = data["Tweet"].str.replace(r"[^a-z\s]+", " ", regex=True)
            data["Tweet"] = data["Tweet"].str.replace(r"\s+", " ", regex=True)

            # **Simpan hasil normalisasi**
            data.to_csv(normalized_path, index=False)
            print("✅ Normalisasi Data Selesai.")

            # **🔹 Visualisasi: Distribusi Panjang Tweet Setelah Normalisasi**
            chart_path_normalized = os.path.join(app.config["STATIC_FOLDER"], "tweet_2_length_distribution_normalized.png")
            plt.figure(figsize=(16, 9))
            data["Tweet"].str.split().apply(len).plot(
                kind="hist",
                bins=30,
                color="green",
                edgecolor="black",
                title="Distribusi Panjang Tweet Setelah Normalisasi"
            )
            plt.xlabel("Jumlah Kata")
            plt.ylabel("Frekuensi")
            plt.savefig(chart_path_normalized, bbox_inches="tight", facecolor="white")
            plt.close()

            # **☁️ WordCloud Setelah Normalisasi**
            wordcloud_path_normalized = os.path.join(app.config["STATIC_FOLDER"], "tweet_2_wordcloud_normalized.png")
            text = " ".join(data["Tweet"].dropna())
            wordcloud = WordCloud(width=1280, height=720, background_color="white").generate(text)
            plt.figure(figsize=(16, 9))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(wordcloud_path_normalized, bbox_inches="tight", facecolor="white")
            plt.close()

        except Exception as e:
            print(f"❌ Kesalahan pada Normalisasi Data: {e}")

        # 📌 3️⃣ Tokenisasi Data
        try:
            print("🚀 Memulai Tokenisasi Data...")
            # **Download resource NLTK jika belum ada**
            try:
                nltk.download("punkt", quiet=True)
            except Exception as e:
                print(f"❌ Gagal mengunduh 'punkt' dari NLTK: {e}")

            tokenized_path = os.path.join(app.config["PROCESSED_FOLDER"], "dataset_3_tokenized.csv")
            normalized_path = os.path.join(app.config["PROCESSED_FOLDER"], "dataset_2_normalized.csv")

            # **Hapus file lama jika ada**
            if os.path.exists(tokenized_path):
                os.remove(tokenized_path)
                
            # **Validasi keberadaan dataset normalisasi**
            if not os.path.exists(normalized_path):
                print("❌ File normalisasi tidak ditemukan!")
                return jsonify({"success": False, "message": "File normalisasi tidak ditemukan!"})

            # **Load Dataset Normalisasi**
            data = pd.read_csv(normalized_path)
            if "Tweet" not in data.columns:
                return jsonify({"success": False, "message": "Kolom 'Tweet' tidak ditemukan dalam dataset!"})

            # (Proses tokenisasi)
            data["Tokenized"] = data["Tweet"].apply(word_tokenize)
            
            # **Simpan hasil tokenisasi**
            data.to_csv(tokenized_path, index=False)
            print("✅ Tokenisasi Data Selesai.")

            # **📊 Visualisasi: Distribusi Panjang Tokenized Tweet**
            chart_path_tokenized = os.path.join(app.config["STATIC_FOLDER"], "tweet_3_length_distribution_tokenized.png")
            plt.figure(figsize=(16, 9))
            data["Tokenized"].apply(len).plot(
                kind="hist",
                bins=30,
                color="purple",
                edgecolor="black",
                title="Distribusi Panjang Tokenized Tweet"
            )
            plt.xlabel("Jumlah Token")
            plt.ylabel("Frekuensi")
            plt.savefig(chart_path_tokenized, bbox_inches="tight", facecolor="white")
            plt.close()

            # **☁️ WordCloud Setelah Tokenisasi**
            wordcloud_path_tokenized = os.path.join(app.config["STATIC_FOLDER"], "tweet_3_wordcloud_tokenized.png")
            text_tokenized = " ".join([" ".join(tokens) for tokens in data["Tokenized"].dropna()])
            wordcloud_tokenized = WordCloud(width=1280, height=720, background_color="white").generate(text_tokenized)
            plt.figure(figsize=(16, 9))
            plt.imshow(wordcloud_tokenized, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(wordcloud_path_tokenized, bbox_inches="tight", facecolor="white")
            plt.close()
            
        except Exception as e:
            print(f"❌ Kesalahan pada Tokenisasi Data: {e}")

        # 📌 4️⃣ Penghapusan Stopwords
        try:
            print("🚀 Memulai Penghapusan Stopwords...")
            
            # **Download resource NLTK jika belum ada**
            try:
                nltk.download("stopwords", quiet=True)
            except Exception as e:
                print(f"❌ Gagal mengunduh 'punkt' dari NLTK: {e}")
            
            stopwords_path = os.path.join(PROCESSED_FOLDER, "dataset_4_no_stopwords.csv")
            tokenized_path = os.path.join(PROCESSED_FOLDER, "dataset_3_tokenized.csv")

            # **Hapus file lama jika ada**
            if os.path.exists(stopwords_path):
                os.remove(stopwords_path)

            # **Validasi keberadaan dataset tokenisasi**
            if not os.path.exists(tokenized_path):
                print("❌ File tokenisasi tidak ditemukan!")
                return jsonify({"success": False, "message": "File tokenisasi tidak ditemukan!"})

            # **Load Dataset Tokenisasi**
            data = pd.read_csv(tokenized_path, converters={"Tokenized": eval})  # Evaluasi string list ke bentuk list Python

            # (Proses stemming)
            stop_words = set(stopwords.words("indonesian"))

            # Stopwords kustom
            manual_stopwords = [
                'gelo',
                'mentri2',
                'yg',
                'ga',
                'udh',
                'aja',
                'kaga',
                'bgt',
                'spt',
                'sdh',
                'dr',
                'utan',
                'tuh',
                'budi',
                'bodi',
                'psi_id',
                'fufufafa',
                'pln',
                'lu',
                'krn',
                'dah',
                'jd',
                'tdk',
                'dll',
                'golkar_id',
                'dlm',
                'ri',
                'jg',
                'ni',
                'sbg',
                'tp',
                'nih',
                'gini',
                'jkw',
                'nggak',
                'bs',
                'pk',
                'ya',
                'gk',
                'gw',
                'gua',
                'klo',
                'msh',
                'blm',
                'gue',
                'sih',
                'pa',
                'dgn',
                'skrg',
                'pake',
                'si',
                'dg',
                'utk',
                'deh',
                'tu',
                'hrt',
                'ala',
                'mdy',
                'moga',
                'tau',
                'liat',
                'orang2',
                'jadi',
            ]
            stop_words.update(manual_stopwords)

            # Stopwords dari file CSV
            stopword_file = os.path.join(PROCESSED_FOLDER, "stopwordbahasa.csv")
            if os.path.exists(stopword_file):
                stopword_df = pd.read_csv(stopword_file, header=None)
                stop_words.update(stopword_df[0].tolist())

            def remove_stopwords_from_tokens(tokens):
                try:
                    tokens = eval(tokens)  # Ubah string token menjadi list Python
                    return [word for word in tokens if word.lower() not in stop_words]
                except Exception as e:
                    return []

            data["Tokenized"] = data["Tokenized"].apply(
                lambda x: remove_stopwords_from_tokens(str(x))
            )
            
            # **Simpan hasil penghapusan stopwords**
            data.to_csv(stopwords_path, index=False,)
            print("✅ Penghapusan Stopwords Selesai.")
            
            # **📊 Visualisasi: Distribusi Panjang Tweet Setelah Penghapusan Stopwords**
            chart_path_no_stopwords = os.path.join(STATIC_FOLDER, "tweet_4_length_distribution_no_stopwords.png")
            plt.figure(figsize=(16, 9))
            data["Tokenized"].apply(len).plot(
                kind="hist",
                bins=30,
                color="purple",
                edgecolor="black",
                title="Distribusi Panjang No Stopwords Tweet"
            )
            plt.xlabel("Jumlah Kata")
            plt.ylabel("Frekuensi")
            plt.savefig(chart_path_no_stopwords, bbox_inches="tight", facecolor="white")
            plt.close()

            # **☁️ WordCloud Setelah Penghapusan Stopwords**
            wordcloud_path_no_stopwords = os.path.join(STATIC_FOLDER, "tweet_4_wordcloud_no_stopwords.png")
            text_no_stopwords = " ".join([" ".join(tokens) for tokens in data["Tokenized"].dropna()])
            wordcloud_no_stopwords = WordCloud(width=1280, height=720, background_color="white").generate(text_no_stopwords)
            plt.figure(figsize=(16, 9))
            plt.imshow(wordcloud_no_stopwords, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(wordcloud_path_no_stopwords, bbox_inches="tight", facecolor="white")
            plt.close()

        except Exception as e:
            print(f"❌ Kesalahan pada Penghapusan Stopwords: {e}")

        # 📌 5️⃣ Stemming Data
        try:
            # (Proses stemming)
            print("🚀 Memulai Stemming Data...")
            
            # **Pastikan pustaka Sastrawi tersedia**
            try:
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
            except Exception as e:
                print(f"❌ Gagal mengunduh 'Stemmer' dari Sastrawi: {e}")

            stemmed_path = os.path.join(PROCESSED_FOLDER, "dataset_5_stemmed.csv")
            no_stopwords_path = os.path.join(PROCESSED_FOLDER, "dataset_4_no_stopwords.csv")

            # **Hapus file lama jika ada**
            if os.path.exists(stemmed_path):
                os.remove(stemmed_path)

            # **Validasi keberadaan dataset sebelum stemming**
            if not os.path.exists(no_stopwords_path):
                print("❌ File hasil penghapusan stopwords tidak ditemukan!")
                return jsonify({"success": False, "message": "File hasil penghapusan stopwords tidak ditemukan!"})

            # **Load Dataset Hasil Penghapusan Stopwords**
            data = pd.read_csv(no_stopwords_path, converters={"Tokenized": eval})

            # **Lakukan Stemming pada setiap kata dalam Tokenized**
            def apply_stemming(tokens):
                return [stemmer.stem(word) for word in tokens]
            
            data["Tokenized"] = data["Tokenized"].apply(apply_stemming)
            
            # **Simpan hasil stemming ke CSV**
            data.to_csv(stemmed_path, index=False)

            # **📊 Visualisasi: Distribusi Panjang Tweet Setelah Stemming**
            chart_path_stemmed = os.path.join(STATIC_FOLDER, "tweet_5_length_distribution_stemmed.png")
            plt.figure(figsize=(16, 9))
            data["Tokenized"].apply(len).plot(
                kind="hist",
                bins=30,
                color="brown",
                edgecolor="black",
                title="Distribusi Panjang Tweet Setelah Stemming"
            )
            plt.xlabel("Jumlah Kata")
            plt.ylabel("Frekuensi")
            plt.savefig(chart_path_stemmed, bbox_inches="tight", facecolor="white")
            plt.close()

            # **☁️ WordCloud Setelah Stemming**
            wordcloud_path_stemmed = os.path.join(STATIC_FOLDER, "tweet_5_wordcloud_stemmed.png")
            text_stemmed = " ".join([" ".join(tokens) for tokens in data["Tokenized"].dropna()])
            wordcloud_stemmed = WordCloud(width=1280, height=720, background_color="white").generate(text_stemmed)
            plt.figure(figsize=(16, 9))
            plt.imshow(wordcloud_stemmed, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(wordcloud_path_stemmed, bbox_inches="tight", facecolor="white")
            plt.close()

        except Exception as e:
            print(f"❌ Kesalahan pada Stemming Data: {e}")

        # 📌 6️⃣ Label Encoding
        try:
            print("🚀 Memulai Label Encoding...")
            
            encoded_path = os.path.join(PROCESSED_FOLDER, "dataset_6_encoded.csv")
            stemmed_path = os.path.join(PROCESSED_FOLDER, "dataset_5_stemmed.csv")

            # **Hapus file lama jika ada**
            if os.path.exists(encoded_path):
                os.remove(encoded_path)

            # **Validasi keberadaan dataset sebelum Label Encoding**
            if not os.path.exists(stemmed_path):
                print("❌ File hasil Stemming tidak ditemukan!")
                return jsonify({"success": False, "message": "File hasil Stemming tidak ditemukan!"})

            # **Load Dataset Hasil Stemming**
            data = pd.read_csv(stemmed_path)
            
            # **Pastikan kolom 'Sentimen' tersedia sebelum proses Label Encoding**
            if "Sentimen" not in data.columns:
                print("❌ Kesalahan: Kolom 'Sentimen' tidak ditemukan dalam dataset!")
                return jsonify({"success": False, "message": "Kolom 'Sentimen' tidak ditemukan dalam dataset!"})

            # **Lakukan Mapping Label Encoding**
            sentiment_mapping = {"positif": 1, "netral": 0, "negatif": -1}
            data["Label_Encoded"] = data["Sentimen"].map(sentiment_mapping)

            # **Simpan hasil Label Encoding ke CSV**
            data.to_csv(encoded_path, index=False)
            print("✅ Label Encoding Selesai.")

            # **📊 Visualisasi: Distribusi Label Encoding**
            chart_path_encoded = os.path.join(STATIC_FOLDER, "tweet_6_label_distribution_encoded.png")
            plt.figure(figsize=(16, 9))
            label_counts_encoded = data["Label_Encoded"].value_counts().reindex([-1, 0, 1], fill_value=0)
            label_counts_encoded.plot(kind="bar", color="orange", edgecolor="black")
            plt.xlabel("Kategori Label")
            plt.ylabel("Frekuensi")
            plt.title("Distribusi Label Encoding")
            plt.xticks(ticks=[0, 1, 2], labels=["-1 (Negatif)", "0 (Netral)", "1 (Positif)"], rotation=0)
            plt.savefig(chart_path_encoded, bbox_inches="tight", facecolor="white")
            plt.close()
    
        except Exception as e:
            print(f"❌ Kesalahan pada Label Encoding: {e}")

        # 📌 7️⃣ Pembagian Data
        try:
            print("🚀 Memulai Pembagian Data...")

            train_path = os.path.join(PROCESSED_FOLDER, "dataset_7_train.csv")
            test_path = os.path.join(PROCESSED_FOLDER, "dataset_7_test.csv")
            encoded_path = os.path.join(PROCESSED_FOLDER, "dataset_6_encoded.csv")

            # **Hapus file lama jika ada**
            for path in [train_path, test_path]:
                if os.path.exists(path):
                    os.remove(path)

            # **Validasi keberadaan dataset sebelum pembagian**
            if not os.path.exists(encoded_path):
                print("❌ File hasil Label Encoding tidak ditemukan!")
                return jsonify({"success": False, "message": "File hasil Label Encoding tidak ditemukan!"})

            # **Load Dataset Hasil Encoding**
            data = pd.read_csv(encoded_path)

            # **Pastikan kolom yang dibutuhkan ada**
            if "Tokenized" not in data.columns or "Label_Encoded" not in data.columns:
                print("❌ Kolom 'Tokenized' atau 'Label_Encoded' tidak ditemukan dalam dataset!")
                return jsonify({"success": False, "message": "Kolom yang diperlukan tidak ditemukan dalam dataset!"})

            # (Proses stemming)
            X = data["Tokenized"]
            y = data["Label_Encoded"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            train_data = pd.DataFrame({"Tokenized": X_train, "Label_Encoded": y_train})
            test_data = pd.DataFrame({"Tokenized": X_test, "Label_Encoded": y_test})

            train_data.to_csv(
                os.path.join(PROCESSED_FOLDER, "dataset_7_train.csv"), index=False
            )
            test_data.to_csv(
                os.path.join(PROCESSED_FOLDER, "dataset_7_test.csv"), index=False
            )
            print("✅ Pembagian Data Selesai.")
            
            # **📊 Visualisasi Distribusi Label dalam Data Training **
            chart_train_split = os.path.join(STATIC_FOLDER, "tweet_7_train_split_distribution.png")
            plt.figure(figsize=(16, 9))
            y_train.value_counts().reindex([-1, 0, 1], fill_value=0).plot(kind="bar", color="blue", edgecolor="black")
            plt.title("Distribusi Sentimen dalam Data Training")
            plt.xlabel("Kategori Sentimen")
            plt.ylabel("Jumlah Sampel")
            plt.xticks(ticks=[0, 1, 2], labels=["-1 (Negatif)", "0 (Netral)", "1 (Positif)"], rotation=0)
            plt.savefig(chart_train_split, bbox_inches="tight", facecolor="white")
            plt.close()

            # **📊 Visualisasi Distribusi Label dalam Data Testing**
            chart_test_split = os.path.join(STATIC_FOLDER, "tweet_7_test_split_distribution.png")
            plt.figure(figsize=(16, 9))
            y_test.value_counts().reindex([-1, 0, 1], fill_value=0).plot(kind="bar", color="orange", edgecolor="black")
            plt.title("Distribusi Sentimen dalam Data Testing")
            plt.xlabel("Kategori Sentimen")
            plt.ylabel("Jumlah Sampel")
            plt.xticks(ticks=[0, 1, 2], labels=["-1 (Negatif)", "0 (Netral)", "1 (Positif)"], rotation=0)
            plt.savefig(chart_test_split, bbox_inches="tight", facecolor="white")
            plt.close()
            
            # **📊 Distribusi Data Train dan Test**
            train_size = len(train_data)
            test_size = len(test_data)
            total_size = train_size + test_size

            train_percentage = round((train_size / total_size) * 100, 2)
            test_percentage = round((test_size / total_size) * 100, 2)

            # **📌 Buat List untuk Distribusi Data**
            data_distribution_split = [
                f"Train Data: {train_size} sampel ({train_percentage}%)",
                f"Test Data: {test_size} sampel ({test_percentage}%)",
            ]

            # **📈 Visualisasi Distribusi Data Train vs Test**
            chart_path_split = os.path.join(
                STATIC_FOLDER, "tweet_7_split_data_distribution.png"
            )

            plt.figure(figsize=(16, 9))
            plt.bar(
                ["Train Data", "Test Data"],
                [train_size, test_size],
                color=["blue", "orange"],
            )
            plt.title("Distribusi Data Setelah Pembagian")
            plt.ylabel("Jumlah Sampel")
            plt.xlabel("Kategori Data")
            plt.savefig(chart_path_split, bbox_inches="tight", facecolor="white")
            plt.close()

        except Exception as e:
            print(f"❌ Kesalahan pada Pembagian Data: {e}")

        flash("Pra pemrosesan data berhasil dilakukan!", "success")
        return jsonify({"success": True})

    except Exception as e:
        print(f"❌ Terjadi Kesalahan: {str(e)}")
        return jsonify({"success": False, "message": str(e)})


# 🔹 Fungsi untuk Halaman Pemodelan Data
@app.route("/modeling")
def modeling():
    try:
        model_files = {
            "Naive Bayes": "model_1a_naive_bayes.pkl",
            "Count Vectorizer": "model_1b_count_vectorizer.pkl",
            "SVM": "model_2a_svm.pkl",
            "TF-IDF Vectorizer": "model_2b_tfidf_vectorizer.pkl",
        }

        model_trained = all(
            os.path.exists(os.path.join(app.config["MODELED_FOLDER"], model_files[key]))
            for key in model_files
        )
        
        report_path = os.path.join(
            app.config["PROCESSED_FOLDER"], "model_0_calculated.csv"
        )

        report_data = []
        if os.path.exists(report_path):
            report_df = pd.read_csv(report_path)
            report_data = report_df.to_dict(orient="records")
            for row in report_data:
                row["Kelas"] = str(row["Kelas"])  # Pastikan kelas dalam format string
                row["Akurasi"] = f"{row['Akurasi'] * 100:.2f}%"  # Ubah Akurasi ke format persen
                row["Presisi"] = f"{row['Presisi'] * 100:.2f}%"
                row["Recall"] = f"{row['Recall'] * 100:.2f}%"
                row["F1-Score"] = f"{row['F1-Score'] * 100:.2f}%"
            # 🔹 Pisahkan Laporan Klasifikasi untuk Naive Bayes dan SVM
            report_data_nb = [row for row in report_data if row["Model"] == "Naive Bayes"]
            report_data_svm = [row for row in report_data if row["Model"] == "SVM"]

        # 🔹 **Tambahkan Default `sentiment_counts`**
        sentiment_counts = {"positif": 0, "negatif": 0, "netral": 0}

        # 🔹 Ambil nama file jika model sudah tersedia
        model_filenames = {
            key: os.path.basename(value) if model_trained else "Nama File PKL"
            for key, value in model_files.items()
        }
        
        return render_template(
            "modeling.html",
            title="Pemodelan Data",
            model_trained=model_trained,
            model_filenames=model_filenames,
            report_data=report_data,
            sentiment_counts=sentiment_counts,
            report_data_nb=report_data_nb,
            report_data_svm=report_data_svm,
            cm_path_nb=url_for("static", filename="img/model_1_naive_bayes_confusion_matrix.png"),
            cm_path_svm=url_for("static", filename="img/model_2_svm_confusion_matrix.png"),
        )

    except Exception as e:
        print(f"❌ Error dalam halaman modeling: {str(e)}")
        return render_template(
            "modeling.html", title="Pemodelan Data", model_trained=False, report_data=[]
        )


# 🔹 Fungsi utama untuk melakukan modeling
@app.route("/start-modeling", methods=["POST"])
def start_modeling():
    try:
        train_path = os.path.join(app.config["PROCESSED_FOLDER"], "dataset_7_train.csv")
        test_path = os.path.join(app.config["PROCESSED_FOLDER"], "dataset_7_test.csv")

        # 🔹 Validasi apakah dataset tersedia
        if not os.path.exists(train_path):
            flash("❌ Data latih belum tersedia!", "danger")
        elif not os.path.exists(test_path):
            flash("❌ Data tes belum tersedia!", "danger")
        else:
        
            # 🔥 **Hapus File yang Sudah Ada Sebelum Modeling**
            files_to_remove = [
                os.path.join(app.config["PROCESSED_FOLDER"], "model_0_calculated.csv"),
                os.path.join(app.config["STATIC_FOLDER"], "model_1_naive_bayes_confusion_matrix.png"),
                os.path.join(app.config["STATIC_FOLDER"], "model_2_svm_confusion_matrix.png")
            ]

            for file in files_to_remove:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"🗑 File dihapus: {file}")

            # 🔹 Load Dataset
            print("📂 Memuat dataset...")
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            X_train, y_train = df_train["Tokenized"], df_train["Label_Encoded"]
            X_test, y_test = df_test["Tokenized"], df_test["Label_Encoded"]

            # 🔹 Vektorisasi
            count_vectorizer = CountVectorizer()
            tfidf_vectorizer = TfidfVectorizer()

            # 🔹 Transformasi Data untuk Naive Bayes & SVM
            print("🔠 Melakukan vektorisasi...")
            X_train_nb = count_vectorizer.fit_transform(X_train)
            X_test_nb = count_vectorizer.transform(X_test)

            X_train_svm = tfidf_vectorizer.fit_transform(X_train)
            X_test_svm = tfidf_vectorizer.transform(X_test)

            # 🔹 Model Naive Bayes
            print("🤖 Melatih model Naive Bayes...")
            nb_model = MultinomialNB()
            nb_model.fit(X_train_nb, y_train)
            y_pred_nb = nb_model.predict(X_test_nb)
            nb_accuracy = accuracy_score(y_test, y_pred_nb)

            # 🔹 Model SVM
            print("🤖 Melatih model SVM...")
            svm_model = SVC(kernel="linear", C=1)
            svm_model.fit(X_train_svm, y_train)
            y_pred_svm = svm_model.predict(X_test_svm)
            svm_accuracy = accuracy_score(y_test, y_pred_svm)

            # 🔹 Simpan Model ke File
            print("💾 Menyimpan model...")
            joblib.dump(
                nb_model,
                os.path.join(app.config["MODELED_FOLDER"], "model_1a_naive_bayes.pkl"),
            )
            joblib.dump(
                count_vectorizer,
                os.path.join(
                    app.config["MODELED_FOLDER"], "model_1b_count_vectorizer.pkl"
                ),
            )

            joblib.dump(
                svm_model, os.path.join(app.config["MODELED_FOLDER"], "model_2a_svm.pkl")
            )
            joblib.dump(
                tfidf_vectorizer,
                os.path.join(
                    app.config["MODELED_FOLDER"], "model_2b_tfidf_vectorizer.pkl"
                ),
            )

            # 🔹 Buat Laporan Klasifikasi
            print("📊 Membuat laporan klasifikasi...")
            report_nb = classification_report(y_test, y_pred_nb, output_dict=True)
            report_svm = classification_report(y_test, y_pred_svm, output_dict=True)

            # 🔹 Simpan Laporan ke CSV
            report_data = []

            for label in [
                "-1",
                "0",
                "1",
            ]:  # Kelas Negatif (-1), Netral (0), Positif (1)
                if label in report_nb:
                    report_data.append(
                        [
                            "Naive Bayes",
                            label,
                            round(nb_accuracy, 4),
                            round(report_nb[label]["precision"], 4),
                            round(report_nb[label]["recall"], 4),
                            round(report_nb[label]["f1-score"], 4),
                            int(report_nb[label]["support"]),
                        ]
                    )
                if label in report_svm:
                    report_data.append(
                        [
                            "SVM",
                            label,
                            round(svm_accuracy, 4),
                            round(report_svm[label]["precision"], 4),
                            round(report_svm[label]["recall"], 4),
                            round(report_svm[label]["f1-score"], 4),
                            int(report_svm[label]["support"]),
                        ]
                    )

            report_df = pd.DataFrame(
                report_data,
                columns=[
                    "Model",
                    "Kelas",
                    "Akurasi",
                    "Presisi",
                    "Recall",
                    "F1-Score",
                    "Support",
                ],
            )
            report_path = os.path.join(
                app.config["PROCESSED_FOLDER"], "model_0_calculated.csv"
            )
            report_df.to_csv(report_path, index=False)

            # 🔹 Confusion Matrix
            print("📊 Membuat confusion matrix...")
            cm_nb = confusion_matrix(y_test, y_pred_nb)
            cm_svm = confusion_matrix(y_test, y_pred_svm)

            # 🔹 Visualisasi Confusion Matrix
            cm_path_nb = os.path.join(
                app.config["STATIC_FOLDER"], "model_1_naive_bayes_confusion_matrix.png"
            )
            cm_path_svm = os.path.join(
                app.config["STATIC_FOLDER"], "model_2_svm_confusion_matrix.png"
            )

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm_nb,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Negatif", "Netral", "Positif"],
                yticklabels=["Negatif", "Netral", "Positif"],
            )
            plt.title("Confusion Matrix - Naive Bayes")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.savefig(cm_path_nb)
            plt.close()

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm_svm,
                annot=True,
                fmt="d",
                cmap="Oranges",
                xticklabels=["Negatif", "Netral", "Positif"],
                yticklabels=["Negatif", "Netral", "Positif"],
            )
            plt.title("Confusion Matrix - SVM")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.savefig(cm_path_svm)
            plt.close()

            flash("Pemodelan data berhasil dilakukan!", "success")
            print("✅ Pemodelan selesai!")
            return jsonify({"success": True})

    except Exception as e:
        print(f"❌ Terjadi kesalahan dalam pemodelan: {str(e)}")
        return jsonify({"success": False, "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
