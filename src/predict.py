
import os
import sys
import pickle

# Ensure both package-style and legacy module paths resolve for unpickling
try:
    from src.linear_svm import LinearSVM  # preferred path
    # Alias to support models pickled with module name 'linear_svm'
    import src.linear_svm as _svm_mod
    sys.modules.setdefault('linear_svm', _svm_mod)
except Exception:
    # Fallback if running without package structure
    from linear_svm import LinearSVM  # type: ignore

# Get the base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and vectorizer using absolute paths
model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'svm_model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'models', 'vectorizer.pkl'), 'rb'))

def predict_spam(text: str) -> int:
    """
    Predict whether a message is spam (1) or not spam (0).
    """
    # Transform text to TF-IDF vector
    X = vectorizer.transform([text]).toarray()  # must be dense for LinearSVM
    pred = model.predict(X)[0]                  # {-1, +1}
    return 0 if pred == -1 else 1
