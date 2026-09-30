# Emotion Detector - Sentiment Analysis with Machine Learning

## Project Description
A web app that detects the emotion behind a piece of text. Built with Python, scikit-learn and Streamlit. Text is cleaned, converted with TF-IDF, and classified by a tuned Logistic Regression model into six emotions: sadness, anger, love, surprise, fear and joy. The app shows the predicted emotion, a confidence score, and a probability chart for every emotion.

## Model Overview
- **Dataset:** `train.txt` (`text;emotion`, 16,000 samples, 80/20 train-test split)
- **Preprocessing:** lowercase, remove punctuation, numbers, URLs, emojis and stopwords
- **Features:** TF-IDF: To convert text into number since ML model can't process raw data.
- **Models tried:** Multinomial Naive Bayes (~66%), Logistic Regression (~86%)
- **Final model:** Logistic Regression tuned with GridSearchCV (~89% test accuracy)

## Prerequisites
- Python 3.8+
- pip
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, nltk, joblib, streamlit

## Installation Instructions
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install numpy pandas matplotlib seaborn scikit-learn nltk joblib streamlit
python -m nltk.downloader punkt punkt_tab stopwords
```

## Usage Guide
1. Make sure `best_model.pkl` and `vectorizer.pkl` are in the same folder as `app.py`.
   (To retrain them, run all cells in `sentimental_analysis_model.ipynb`.)
2. Launch the app:
   ```bash
   streamlit run app.py
   ```
3. Open the local URL shown in the terminal (usually http://localhost:8501).
4. Type a sentence, or pick an example, then click **Analyze emotion**.

Example: `I am so happy today, everything went great!` → **Joy** 

## Project Structure
```
├── app.py                             # Streamlit UI
├── sentimental_analysis_model.ipynb   # Model training notebook
├── train.txt                          # Dataset
├── best_model.pkl                     # Trained model
└── vectorizer.pkl                     # TF-IDF vectorizer
```

## Tech Stack
Python, scikit-learn, NLTK, pandas, Streamlit

## 🚀 Live demo
[Try the Emotion Detector App] (https://ml-sentiment-analysis-classifier.streamlit.app) 
