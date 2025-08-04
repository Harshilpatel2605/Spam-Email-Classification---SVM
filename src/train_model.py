import pandas as pd
from sklearn.model_selection import train_test_split #To split your dataset into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer #Converts text into TF-IDF vectors
from sklearn.svm import SVC #Support Vector Classifier from scikit-learn
from sklearn.metrics import classification_report #Evaluates the model’s performance
import pickle #Used to save your model and vectorizer for later use

# Load & Pre-process data
df = pd.read_csv('Dataset/data.csv', encoding='latin-1', sep='\t', names=['label', 'message'])
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1}) # map the string labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)


# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000) # Limits vocabulary to the top 3000 most important words (to reduce noise & computation).
X_train_tfidf = vectorizer.fit_transform(X_train) # Learns vocabulary from training data and transforms it.
X_test_tfidf = vectorizer.transform(X_test) # Transforms the test data using the learned vocabulary.

# SVM
clf = SVC(kernel='linear')  # Creates a linear Support Vector Machine Classifier.
clf.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = clf.predict(X_test_tfidf)
# Evaluate
accuracy = clf.score(X_test_tfidf, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model + vectorizer
pickle.dump(clf, open('svm_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
