import requests

API_URL = "http://127.0.0.1:5000/api/predict"

sample_patients = [
    {
        "age": 45,
        "trestbps": 135,
        "chol": 220,
        "thalach": 160,
        "oldpeak": 0.5,
        "ca": 1,
        "sex": "male",
        "cp": "atypical angina",
        "fbs": "no",
        "restecg": "left ventricular hypertrophy",
        "exang": "no",
        "slope": "flat",
        "thal": "fixed defect"
    },
    {
        "age": 50,
        "trestbps": 150,
        "chol": 240,
        "thalach": 155,
        "oldpeak": 1.2,
        "ca": 0,
        "sex": "female",
        "cp": "asymptomatic",
        "fbs": "yes",
        "restecg": "normal",
        "exang": "yes",
        "slope": "downsloping",
        "thal": "reversible defect"
    },
    {
        "age": 65,
        "trestbps": 145,
        "chol": 270,
        "thalach": 130,
        "oldpeak": 2.0,
        "ca": 3,
        "sex": "male",
        "cp": "typical angina",
        "fbs": "no",
        "restecg": "ST-T wave abnormality",
        "exang": "yes",
        "slope": "upsloping",
        "thal": "normal"
    }
]

def print_probabilities(prob_dict):
    """Print each class probability on its own line."""
    for cls, prob in prob_dict.items():
        print(f"{cls}: {prob*100:.2f}%")

def test_api():
    for i, patient in enumerate(sample_patients, 1):
        response = requests.post(API_URL, json=patient)
        print(f"Sample {i}:")
        print("="*25)
        if response.status_code == 200:
            data = response.json()
            print(f"Prediction: {data['prediction']}, Confidence: {data['confidence']*100:.2f}%\n")
            print("Class Probabilities: ")
            print_probabilities(data['probabilities'])
        else:
            print(f"Error {response.status_code} - {response.json()}")
        print("\n")

if __name__ == "__main__":
    test_api()
