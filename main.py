import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


class MetaphorDetector:
    def __init__(self):
        # Dictionary mapping metaphorID to target words
        self.metaphor_words = {
            0: 'road',
            1: 'candle',
            2: 'light',
            3: 'spice',
            4: 'ride',
            5: 'train',
            6: 'boat'
        }

        # Initialize the pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])

    def preprocess_text(self, text):
        """Preprocess the input text."""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Join tokens back into text
        return ' '.join(tokens)

    def add_metaphor_context(self, text, metaphor_id):
        """Add metaphor-specific features to the text."""
        target_word = self.metaphor_words[metaphor_id]

        # Add the target word as a special feature
        return f"{text} TARGET_WORD_{target_word}"

    def prepare_features(self, df):
        """Prepare features from the input dataframe."""
        # Preprocess texts
        processed_texts = df['text'].apply(self.preprocess_text)

        # Add metaphor context
        processed_texts = [
            self.add_metaphor_context(text, mid)
            for text, mid in zip(processed_texts, df['metaphorID'])
        ]

        return processed_texts

    def fit(self, X, y):
        """Train the model."""
        # Prepare features
        processed_X = self.prepare_features(X)

        # Fit the pipeline
        self.pipeline.fit(processed_X, y)

    def predict(self, X):
        """Make predictions on new data."""
        # Prepare features
        processed_X = self.prepare_features(X)

        # Make predictions
        return self.pipeline.predict(processed_X)

    def evaluate(self, X_test, y_test):
        """Evaluate the model performance."""
        predictions = self.predict(X_test)
        return classification_report(y_test, predictions)


# Example usage
def main():
    # Load and prepare data
    df = pd.read_csv('train-1.csv')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df[['metaphorID', 'text']],
        df['label'],
        test_size=0.2,
        random_state=42
    )

    # Initialize and train the model
    detector = MetaphorDetector()
    detector.fit(X_train, y_train)

    # Evaluate the model
    print("Model Performance:")
    print(detector.evaluate(X_test, y_test))

    # Example prediction
    example_text = pd.DataFrame({
        'metaphorID': [0],
        'text': ["Life is a long road with many twists and turns"]
    })
    prediction = detector.predict(example_text)
    print(f"\nExample Prediction:")
    print(f"Text: {example_text['text'].iloc[0]}")
    print(f"Contains metaphor: {prediction[0]}")


if __name__ == "__main__":
    main()