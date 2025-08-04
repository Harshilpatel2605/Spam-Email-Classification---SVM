import pickle

# Load model and vectorizer
model = pickle.load(open('svm_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def predict_spam(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]
