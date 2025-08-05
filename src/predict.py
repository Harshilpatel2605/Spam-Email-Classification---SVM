import pickle
from src.linear_svm import LinearSVM  # Make sure this import matches how it was saved

# Load model and vectorizer
model = pickle.load(open('models/svm_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

def predict_spam(text: str) -> int:
    """
    Predict whether a message is spam (1) or not spam (0).
    """
    # Transform text to TF-IDF vector
    X = vectorizer.transform([text]).toarray()  # must be dense for LinearSVM
    pred = model.predict(X)[0]                  # {-1, +1}
    return 0 if pred == -1 else 1
