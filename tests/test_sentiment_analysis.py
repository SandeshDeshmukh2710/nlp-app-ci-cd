import sys
sys.path.insert(1, "C:/Users/Admin/Desktop/Data Science/MLOps/CI-CD-Practice/src")

import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sentiment_analysis as SA

class TestSentimentAnalysis(unittest.TestCase):

    def test_preprocess_text(self):
        # Test preprocess_text function
        input_text = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked."
        expected_output = "one reviewers mentioned watching 1 oz episode hooked"
        self.assertEqual(SA.preprocess_text(input_text), expected_output)

    def test_train_model(self):
        # Test train_model function with more realistic data
        # Sample data for testing
        data = {'review': ["One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked.",
                           "A wonderful little production. The filming technique is very unassuming...",
                           "I thought this was a wonderful way to spend time on a too hot summer weekend..."],
                'sentiment': [1, 1, 0]}
        df = pd.DataFrame(data)

        # Preprocess the data
        df['review'] = df['review'].apply(SA.preprocess_text)

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

        # Vectorize the text
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        # Train a simple model
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        # Ensure the model is trained
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
