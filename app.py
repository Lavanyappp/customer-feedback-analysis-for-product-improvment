import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import contractions
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import joblib
import json

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

from streamlit_lottie import st_lottie
st.set_page_config(layout='wide')

# Function to load animation files
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load animation file
logo = load_lottiefile('Animation - 1696469171082.json')

# Load trained model and vectorizer
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Centered title
st.markdown("<h1 style='text-align: center;'>Sentiment Analysis on Product Reviews</h1>", unsafe_allow_html=True)
st_lottie(logo, speed=1, reverse=False, quality="low", loop=True, height=250)

# Sidebar Information
st.sidebar.header("About This App")
st.sidebar.markdown("""
This application analyzes the sentiment of product reviews.  
- **Input**: A product review  
- **Processing**: Text is cleaned, vectorized, and analyzed using an **SVM model**  
- **Output**: Positive or Negative sentiment  
""")

st.sidebar.info("💡 Try entering different types of reviews to see the sentiment analysis in action!")

# Centered text input
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    user_input = st.text_area('Enter a review:')

    if st.button('Analyze Sentiment'):
        if user_input:
            # Preprocess the input text
            user_input = re.sub(r'[\W_]+', ' ', contractions.fix(re.sub(r'\d+', '', user_input.replace(' s ', ' ')))).lower()
            tokens = word_tokenize(user_input)
            user_input = " ".join([token for token in tokens if token not in stopwords.words('english')])
            user_input = ' '.join([SnowballStemmer("english").stem(word) for word in user_input.split()])

            # Vectorize input
            user_input_vec = vectorizer.transform([user_input])

            # Make a prediction
            prediction = model.predict(user_input_vec)

            # Display the result
            if prediction == 0:
                st.warning('⚠️ Warning: This review seems to have a negative sentiment.')
            else:
                st.success('🎉 Great news! This review appears to have a positive sentiment.')
        else:
            st.warning('Please enter a review for analysis.')
