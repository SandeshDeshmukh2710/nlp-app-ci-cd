import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import string

nltk.download('stopwords')
nltk.download('punkt')

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

def preprocess_data(df):
    # Applying preprocessing to the 'review' column
    df['review'] = df['review'].apply(preprocess_text)
    
    # Convert 'sentiment' to binary labels (0 for negative, 1 for positive)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    return df

def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate the model
    X_test_vec = vectorizer.transform(X_test)
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)

    print(f'Model Accuracy: {accuracy}')

    return model, vectorizer

def save_model_and_vectorizer(model, vectorizer, model_path='sentiment_model.joblib', vectorizer_path='vectorizer.joblib'):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

if __name__ == "__main__":
    # Example usage with the provided sample data
    data = {'review': ["One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked...",
                       "A wonderful little production. The filming technique is very unassuming...",
                       "I thought this was a wonderful way to spend time on a too hot summer weekend...",
                       "Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet...",
                       "Petter Mattei's 'Love in the Time of Money' is a visually stunning film to watch...",
                       "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication...",
                       "I sure would like to see a resurrection of an up-to-date Sea Hunt series...",
                       "This show was an amazing, fresh & innovative idea in the 70's when it first aired...",
                       "Encouraged by the positive comments about this film on here I was looking forward to watching this film. Bad mistake."],
            'sentiment': ['positive', 'positive', 'positive', 'negative', 'positive', 'positive', 'positive', 'negative', 'negative']}
    

    df = pd.DataFrame(data)
    
    # Preprocess the sample data
    df = preprocess_data(df)
    
    trained_model, trained_vectorizer = train_model(df)
    save_model_and_vectorizer(trained_model, trained_vectorizer)
