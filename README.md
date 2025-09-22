# 📰 Fake News Prediction

## Overview
This project implements a machine learning-based system to classify news articles as **real** or **fake**. Using Natural Language Processing (NLP) techniques like text preprocessing, feature extraction, and supervised learning models, the system can identify misleading or false information.

## Features
- Preprocesses textual data (stopword removal, tokenization, stemming/lemmatization).
- Extracts features using **TF-IDF** and **CountVectorizer**.
- Trains and evaluates multiple ML models (Logistic Regression, Naive Bayes, Random Forest, etc.).
- Provides evaluation metrics: Accuracy, Precision, Recall, F1-Score.
- Runs entirely within a Jupyter Notebook.

## File Structure
| File                          | Purpose                                                   |
|-------------------------------|-----------------------------------------------------------|
| `Fake_news_prediction.ipynb`  | Main notebook containing data preprocessing, training, and evaluation. |
| `README.md`                   | Project documentation.                                    |
| `requirements.txt`            | Python dependencies.                                      |
| `data/`                       | Dataset folder (not included in repo).                   |

## Getting Started
### Prerequisites
- Python 3.x  
- Jupyter Notebook  
- Install dependencies: `pip install -r requirements.txt`

### Usage
1. Place dataset inside `data/` folder  
2. Open the notebook: `jupyter notebook Fake_news_prediction.ipynb`  
3. Run all cells to preprocess data, train models, and evaluate  

## Example
- Classification report and confusion matrix are generated in the notebook  
- Best-performing model highlighted (e.g., Logistic Regression with X% accuracy)  

## How It Works
1. Input: Loads dataset of labeled news articles  
2. Preprocessing: Cleans text by removing punctuation, stopwords, stemming/lemmatization  
3. Feature Extraction: Converts text into numeric form using TF-IDF or CountVectorizer  
4. Model Training: Trains classifiers such as Logistic Regression or Naive Bayes  
5. Evaluation: Measures model performance using accuracy, precision, recall, and F1-score  
6. Prediction: Classifies unseen articles as real or fake  

## Customization
- Adjust preprocessing steps (custom stopword list, lemmatization, etc.)  
- Modify feature extraction (TF-IDF, word embeddings)  
- Experiment with different ML/DL models (SVM, Random Forest, LSTM, BERT)  

## Limitations
- Model performance depends on dataset quality and size  
- May not generalize well to new domains or languages  
- Current version is educational/demo purpose, not production ready  
