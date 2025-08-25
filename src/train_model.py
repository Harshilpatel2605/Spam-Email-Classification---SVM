import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
from src.linear_svm import LinearSVM

import pandas as pd

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv(
    '../Dataset/combined_data.csv',
    encoding='latin-1',
    sep=',',
    names=['label', 'text'],
    header=0
)
print("Dataset loaded successfully.\n\n")
df.info()
print("\n\n")
print(df.head())
print("taking only 10,000 samples from the dataset, due to memory limitations on CPU.")
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# Train-test split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
print("Train and test split completed.")

# Then split train into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)
print("Training Data : 60%, Validation Data : 20%, Test Data : 20%")

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,2))
# Fit on training set only
X_train = vectorizer.fit_transform(X_train).toarray()
X_val   = vectorizer.transform(X_val).toarray()
X_test  = vectorizer.transform(X_test).toarray()

# Convert labels to {-1, +1}
y_train = np.where(y_train == 0, -1, 1)
y_val   = np.where(y_val == 0, -1, 1)
y_test  = np.where(y_test == 0, -1, 1)

# Train SVM on our custom model.
# hypertuning c (regularization constant), train on various values of c
C_values = [0.001, 0.01, 0.1, 1, 10, 50, 100]

best_val_acc = 0
best_C = None
best_model = None

for C in C_values:
    print(f"Training with C={C}...")
    svm = LinearSVM(C=C)
    svm.fit(X_train, y_train)
    
    val_acc = np.mean(svm.predict(X_val) == y_val)
    print(f"C={C}, Validation Accuracy={val_acc:.4f}\n")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_C = C
        best_model = svm # store the best model for later use.
print("..training completed")
print(f"Best C={best_C} with Validation Accuracy={best_val_acc:.4f}\n")

# Evaluate best model on test set
print("Evaluating best model on test set...")
y_pred = best_model.predict(X_test)
y_pred_class = np.where(y_pred == -1, 0, 1)
y_test_class = np.where(y_test == -1, 0, 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test_class, y_pred_class)
print(f"Model Accuracy on Test Dataset: {acc*100:.2f}%")

# Save best model and vectorizer
print("Saving model and vectorizer...")
pickle.dump(best_model, open('../models/svm_model.pkl', 'wb'))
pickle.dump(vectorizer, open('../models/vectorizer.pkl', 'wb'))

