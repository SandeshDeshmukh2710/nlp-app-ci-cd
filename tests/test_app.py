import unittest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

class TestApp(unittest.TestCase):

    def test_predict_sentiment(self):
        # Test /predict endpoint
        response = client.post("/predict", data={"review": "This is a positive review."})
        data = response.json()
        self.assertIn("review", data)
        self.assertIn("sentiment", data)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
