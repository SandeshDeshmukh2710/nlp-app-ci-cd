from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import joblib
from bs4 import BeautifulSoup
import string
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize

# import sys
# sys.path.insert(1, "C:/Users/Admin/Desktop/Data Science/MLOps/CI-CD-Practice/src")

#import sentiment_analysis as SA

app = FastAPI()

# Load trained model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

class SentimentRequest(BaseModel):
    review: str

def preprocess_text(text):
    text = text.lower()

    # Removing HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Handling contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)

    # Removing punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Joining tokens back into a string
    processed_text = ' '.join(tokens)

    return processed_text


@app.post("/predict")
async def predict_sentiment(review: str = Form(...)):
    # Preprocess the input text
    processed_text = preprocess_text(review)
    
    # Vectorize the processed text
    processed_vec = vectorizer.transform([processed_text])
    
    # Predict sentiment
    prediction = model.predict(processed_vec)
    
    # Return prediction result
    return {"review": review, "sentiment": int(prediction[0])}
