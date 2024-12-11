import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.cluster import KMeans

os.environ['LOKY_MAX_CPU_COUNT'] = '4' #setting the number of cores manually

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def extract_features(X_train, X_test):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 3)
    )
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized, vectorizer


class MetaphorDetector:
    def __init__(self):
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 3)
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=4
            ))
        ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return classification_report(y_test, y_pred)


def cluster_metaphors(X, n_clusters=2):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 3)
    )
    X_vectorized = vectorizer.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_vectorized)

    return clusters, kmeans, vectorizer


def main():
    df = pd.read_csv('train-1.csv')

    df['processed_text'] = df['text'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'],
        df['label'],
        test_size=0.2,
        random_state=42
    )

    detector = MetaphorDetector()
    detector.train(X_train, y_train)

    print("Model Performance:")
    print(detector.evaluate(X_test, y_test))

    clusters, kmeans, vectorizer = cluster_metaphors(df['processed_text'])

    def predict_metaphor(text):
        processed_text = preprocess_text(text)
        prediction = detector.predict([processed_text])[0]
        return {
            'text': text,
            'contains_metaphor': bool(prediction),
            'cluster': kmeans.predict(vectorizer.transform([processed_text]))[0]
        }

    example_text = "Life is a long road with many twists and turns"
    print("\nExample Prediction:")
    result = predict_metaphor(example_text)
    print(f"Text: {result['text']}")
    print(f"Contains metaphor: {result['contains_metaphor']}")
    #print(f"Cluster: {result['cluster']}")


if __name__ == "__main__":
    main()