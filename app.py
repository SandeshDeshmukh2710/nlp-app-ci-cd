from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import joblib

import sys
sys.path.insert(1, "C:/Users/Admin/Desktop/Data Science/MLOps/CI-CD-Practice/src")

import sentiment_analysis as SA

app = FastAPI()

# Load trained model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

class SentimentRequest(BaseModel):
    review: str

@app.post("/predict")
async def predict_sentiment(review: str = Form(...)):
    # Preprocess the input text
    processed_text = SA.preprocess_text(review)
    
    # Vectorize the processed text
    processed_vec = vectorizer.transform([processed_text])
    
    # Predict sentiment
    prediction = model.predict(processed_vec)
    
    # Return prediction result
    return {"review": review, "sentiment": int(prediction[0])}
