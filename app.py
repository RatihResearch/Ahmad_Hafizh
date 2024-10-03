import streamlit as st
import numpy as np
import joblib
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
import nltk

# Preprocessing function
stop_factory = StopWordRemoverFactory()
more_stopword = [
    'sdt', 'sdm', 'sendok', 'makan', 'teh', 'spt', 'sejumput', 'kilogram', 'kg', 'gram', 'g', 'gr', 'ml', 'cm', 'ons',
    'kecil', 'sedang', 'besar', 'sedikit', 'secukupnya', 'lembar', 'lbr', 'siung', 'mangkok', 'ruas', 'piring', 'matang',
    'btg', 'bh', 'masak', 'potong', 'ptng', 'masak', 'cincang', 'iris', 'cuci', 'bersih', 'buah', 'buahnya', 'buahnya',
    'bungkus', 'bks', 'iris', 'serut', 'butir', 'biji', 'stgh', 'stgah', 'lt', 'liter', 'sdh', 'ikat', 'tbsps', 'genggam',
    'utuh', 'tipis', 'btr', 'kotak', 'jadi', 'parut', 'cina', 'rebus', 'celup', 'keriting', 'yg', 'buang', 'tumis', 'sachet',
    'kupas', 'pakai', 'rajang', 'bagi', 'utk', 'kukus', 'bakar', 'tumbuk', 'halus', 'kasar', 'sisir', 'saring', 'sisih',
    'fillet', 'dadu', 'halus', 'panas', 'celup', 'hancur', 'papan', 'segar', 'ukur', 'rebus', 'uk', 'buat', 'belah', 'ceplok',
    'celup', 'isi', 'aja', 'serong', 'goreng', 'sedap', 'tangkai', 'siap', 'hangat', 'keping', 'batang', 'ekor', 'memar', 'larut',
    'suka', 'instan', 'sesuai', 'kocok'
]
data_stopword = stop_factory.get_stop_words() + more_stopword
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


def preprocess(s):
    # Remove Punctuation
    s = re.sub(r'[^\w\s\d\n]', ' ', s)
    # Remove Digits
    s = re.sub(r'\d+', ' ', s)

    hasil = []
    word_token = word_tokenize(s)  # Tokenisasi
    unique_words = set()

    for word in word_token:
        word = word.strip().lower()  # Case folding
        if word not in data_stopword:  # Stopword removal
            word = stemmer.stem(word)  # Stemming
            if word not in unique_words:  # Cek apakah kata sudah ada sebelumnya
                hasil.append(word)
                unique_words.add(word)
        else:
            continue
    # Penggabungan kata hasil pre-processing
    result_sentence = " ".join(hasil).strip()
    return result_sentence


# Load the trained models
tfidf_vectorizer = joblib.load('./model/tfidf_vectorizer.pkl')
knn_model = joblib.load('./model/model_knn-fix.pkl')
svm_model = joblib.load('./model/model_svm-fix.pkl')
mlp_model = joblib.load('./model/model_mlp-fix.pkl')

# Download necessary NLTK data
nltk.download('punkt')

# Title
st.title('Allergen Detection on Food Recipe')

# Input text
food_input = st.text_input('Input Komposisi Makanan:')

# Choosing model
model_choice = st.selectbox('Choose a model:', ['KNN', 'SVM', 'MLP'])

# Button to predict
if st.button('Predict'):
    if food_input:
        # Preprocess the input using the preprocessing function
        preprocessed_input = preprocess(food_input)

        # Transform the preprocessed input using the TF-IDF vectorizer
        input_features = tfidf_vectorizer.transform([preprocessed_input])

        # Predict using the chosen model
        if model_choice == 'KNN':
            prediction = knn_model.predict(input_features)
        elif model_choice == 'SVM':
            prediction = svm_model.predict(input_features)
        elif model_choice == 'MLP':
            prediction = mlp_model.predict(input_features)

        # Assuming prediction is a multi-label binary array
        allergens = ['Susu', 'Kacang', 'Telur', 'Makanan Laut', 'Gandum']
        detected_allergens = [allergens[i]
                              for i in range(len(allergens)) if prediction[0][i] == 1]

        if detected_allergens:
            st.write('Detected Allergens:', ', '.join(detected_allergens))
        else:
            st.write('No Allergen Detected.')
    else:
        st.write('Please enter the food description.')