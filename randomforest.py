import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')


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
        report = classification_report(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        return report, accuracy

    def grid_search(self, X, y):
        """Perform grid search for hyperparameter tuning."""
        # Define the parameter grid
        param_grid = {
            'tfidf__max_features': [500, 1000, 2000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )

        # Prepare features
        processed_X = self.prepare_features(X)

        # Perform grid search
        grid_search.fit(processed_X, y)

        # Update pipeline with best parameters
        self.pipeline = grid_search.best_estimator_

        print("Best Parameters:", grid_search.best_params_)
        return grid_search.best_params_


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

    # Initialize the detector
    detector = MetaphorDetector()

    # Perform grid search
    print("Performing Grid Search...")
    best_params = detector.grid_search(X_train, y_train)
    print("Best parameters found:", best_params)

    # Train the model with best parameters
    detector.fit(X_train, y_train)

    # Evaluate the model
    print("\nModel Performance on Test Data:")
    report, test_accuracy = detector.evaluate(X_test, y_test)
    print(report)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Compute training accuracy
    print("\nModel Performance on Training Data:")
    train_predictions = detector.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f"Training Accuracy: {train_accuracy:.4f}")

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
