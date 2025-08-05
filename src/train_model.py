import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
from src.linear_svm import LinearSVM

# Load data
df = pd.read_csv('Dataset/data.csv', encoding='latin-1', sep='\t', names=['label', 'message'])    
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train_text, X_test_text, y_train_orig, y_test_orig = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train = vectorizer.fit_transform(X_train_text).toarray()
X_test = vectorizer.transform(X_test_text).toarray()

# Convert labels to {-1, +1}
y_train = np.where(y_train_orig.values == 0, -1, 1)
y_test = np.where(y_test_orig.values == 0, -1, 1)

# Train SVM
svm = LinearSVM(C=1.0)
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
y_pred_class = np.where(y_pred == -1, 0, 1)
y_test_class = np.where(y_test == -1, 0, 1)
acc = accuracy_score(y_test_class, y_pred_class)
print(f"Accuracy: {acc*100:.2f}%")

# Save
pickle.dump(svm, open('models/svm_model.pkl', 'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
