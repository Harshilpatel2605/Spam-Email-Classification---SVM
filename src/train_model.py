import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import pickle
from linear_svm import LinearSVM

# -------------------------------
# 1. Load dataset
# -------------------------------
print("Loading the dataset...")
df = pd.read_csv(
    '../Dataset/combined_data.csv',
    encoding='latin-1',
    sep=',',
    names=['label', 'text'],
    header=0
)
print("Dataset loaded successfully.\n")
print(df['label'].value_counts())

# -------------------------------
# 2. Preprocess text
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    return text.strip()

df['text'] = df['text'].apply(clean_text)

# -------------------------------
# 3. Balance dataset (equal ham/spam)
# -------------------------------
ham = df[df['label'] == 0]
spam = df[df['label'] == 1]

if len(spam) > len(ham):
    spam = resample(spam, replace=False, n_samples=len(ham), random_state=42)
else:
    ham = resample(ham, replace=False, n_samples=len(spam), random_state=42)

df = pd.concat([ham, spam]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Balanced dataset size:", df['label'].value_counts())

# Optional: limit size if memory is an issue
print("Taking only 10,000 samples (due to memory limits)...")
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# -------------------------------
# 4. Train-test split
# -------------------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print("Training Data : 60%, Validation Data : 20%, Test Data : 20%")

# -------------------------------
# 5. TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1,2),
    min_df=3
)

X_train = vectorizer.fit_transform(X_train).toarray()
X_val   = vectorizer.transform(X_val).toarray()
X_test  = vectorizer.transform(X_test).toarray()

# Convert labels to {-1, +1}
y_train = np.where(y_train == 0, -1, 1)
y_val   = np.where(y_val == 0, -1, 1)
y_test  = np.where(y_test == 0, -1, 1)

# -------------------------------
# 6. Train SVM with hyperparameter tuning
# -------------------------------
C_values = [1]

best_val_acc = 0
best_C = None
best_model = None

for C in C_values:
    print(f"\nTraining with C={C}...")
    svm = LinearSVM(C=C)
    svm.fit(X_train, y_train)
    
    val_acc = np.mean(svm.predict(X_val) == y_val)
    print(f"C={C}, Validation Accuracy={val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_C = C
        best_model = svm

print("\nTraining completed.")
print(f"Best C={best_C} with Validation Accuracy={best_val_acc:.4f}")

# -------------------------------
# 7. Evaluate best model
# -------------------------------
print("\nEvaluating best model on test set...")
y_pred = best_model.predict(X_test)
y_pred_class = np.where(y_pred == -1, 0, 1)
y_test_class = np.where(y_test == -1, 0, 1)

acc = accuracy_score(y_test_class, y_pred_class)
print(f"Test Accuracy: {acc*100:.2f}%\n")
print(classification_report(y_test_class, y_pred_class, target_names=['ham', 'spam']))

# -------------------------------
# 8. Save model & vectorizer
# -------------------------------
print("Saving model and vectorizer...")
pickle.dump(best_model, open('../models/svm_model.pkl', 'wb'))
pickle.dump(vectorizer, open('../models/vectorizer.pkl', 'wb'))
print("Model saved successfully.")
