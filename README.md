# Spam Email Classifier – SVM

A machine learning project to classify SMS messages as **Spam** or **Legitimate** using a **Soft-Margin Linear SVM** implemented from scratch (Dual formulation with Quadratic Programming), combined with **TF-IDF** text feature extraction.  
The project also includes a **Streamlit web app** for real-time message classification.

---

##  Live Demo
🔗 [Spam Detector Web App](https://spam-detector-svm.streamlit.app/)

---

##  Features
- **Custom SVM Implementation** – Dual form with slack variables, solved using `cvxopt` (no scikit-learn SVM).
- **Soft Margin SVM** – Tunable `C` parameter to balance margin size and classification errors.
- **TF-IDF Vectorization** – Converts raw text into numerical vectors while reducing noise with stop-word removal.
- **Interactive Web App** – Built with **Streamlit** for live spam detection.
- **Model Persistence** – Uses `pickle` to save and load the model and vectorizer.
- **High Accuracy** – Achieved ~99% test accuracy on SMS Spam dataset.


