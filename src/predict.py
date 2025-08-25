
import os
import sys
import pickle

try:
    from src.linear_svm import LinearSVM 
    import src.linear_svm as _svm_mod
    sys.modules.setdefault('linear_svm', _svm_mod)
except Exception:

    from linear_svm import LinearSVM  
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'svm_model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'models', 'vectorizer.pkl'), 'rb'))


#Predict whether a message is spam (1) or not spam (0).
def predict_spam(text: str) -> int:
    
    # Transform text to TF-IDF vector and then predict.
    X = vectorizer.transform([text]).toarray() 
    pred = model.predict(X)[0]     # {-1, +1}             

    return 0 if pred == -1 else 1
