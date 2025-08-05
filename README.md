# 📩 Spam Email Classifier – SVM

A machine learning project to classify SMS messages as **Spam** or **Ham** using a **Soft-Margin Linear SVM** implemented from scratch (Dual formulation with Quadratic Programming), combined with **TF-IDF** text feature extraction.  
The project also includes a **Streamlit web app** for real-time message classification.

---

## 🚀 Live Demo
🔗 [Spam Detector Web App](https://your-deployed-streamlit-link.com)

---

## 📌 Features
- **Custom SVM Implementation** – Dual form with slack variables, solved using `cvxopt` (no scikit-learn SVM).
- **Soft Margin SVM** – Tunable `C` parameter to balance margin size and classification errors.
- **TF-IDF Vectorization** – Converts raw text into numerical vectors while reducing noise with stop-word removal.
- **Interactive Web App** – Built with **Streamlit** for live spam detection.
- **Model Persistence** – Uses `pickle` to save and load the model and vectorizer.
- **High Accuracy** – Achieved ~99% test accuracy on SMS Spam dataset.

---

## 📂 Project Structure
Spam Email Classifier/
├── app/
│ └── streamlit_app.py # Streamlit frontend
├── src/
│ ├── linear_svm.py # Custom SVM implementation
│ ├── train_model.py # Train & save model
│ └── predict.py # Load model & make predictions
├── models/
│ ├── svm_model.pkl # Saved trained model
│ └── vectorizer.pkl # Saved TF-IDF vectorizer
├── Dataset/
│ └── data.csv # SMS spam dataset
└── requirements.txt # Python dependencies


---

## 🛠 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/spam-email-classifier.git
cd spam-email-classifier

### 2️⃣ Install dependencies
'''
pip install -r requirements.txt
'''

### 3️⃣ Train the Model
'''
python -m src.train_model
'''

### 4️⃣ Run the Streamlit App
'''
streamlit run app/streamlit_app.py
'''

## 📊 Model Details

    Algorithm: Soft Margin Linear SVM (Dual form)

    Solver: cvxopt

    Kernel: Linear

    Feature Extraction: TF-IDF (max_features=3000)

    Evaluation Metric: Accuracy

## Results
- Accuracy : 99.01%

