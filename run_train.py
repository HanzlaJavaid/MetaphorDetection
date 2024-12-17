from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.model_selection import GridSearchCV
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
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import joblib
from datetime import datetime
import os


class Attention(nn.Module):
    """Attention mechanism to weigh embedding features."""
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        Returns:
        weighted_output: [batch_size, input_dim]
        attention_scores: [batch_size, seq_len]
        """
        # Compute attention scores
        scores = self.attention_weights(x).squeeze(-1)  # [batch_size, seq_len]
        scores = torch.softmax(scores, dim=1)  # Normalize scores

        # Weighted sum of features
        weighted_output = torch.bmm(scores.unsqueeze(1), x).squeeze(1)  # [batch_size, input_dim]
        return weighted_output, scores

class SentenceTransformerAttentionNN:
    def __init__(self, model_name='all-mpnet-base-v2', input_dim=768, hidden_dim=128, num_classes=2):
        # Initialize Sentence Transformer
        self.st_model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Attention + Fully Connected Neural Network
        self.attention = Attention(input_dim).to(self.device)
        # We'll output_value='token_embeddings', so we have [batch, seq_len, input_dim].
        # After attention, we get [batch, input_dim].

        # FC layers (after we combine attended vector with a residual vector)
        # If we choose concatenation, our input dimension might double (input_dim * 2).
        # Otherwise, if we choose addition, it stays the same. Let's choose concatenation for demonstration.
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            list(self.attention.parameters()) + list(self.fc_layers.parameters()),
            lr=1e-3,
            weight_decay=1e-5
        )

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

    def prepare_text(self, text, metaphor_id):
        """Prepare text by adding metaphor context."""
        target_word = self.metaphor_words[metaphor_id]
        return f"{text} [SEP] The target word is {target_word}"

    def get_token_embeddings_batch(self, texts):
        """Get token-level embeddings using Sentence Transformers."""
        # This returns a list of Tensors, each shape [seq_len, embedding_dim].
        embeddings_list = self.st_model.encode(
            texts,
            batch_size=32,
            convert_to_tensor=True,
            output_value='token_embeddings'
        )

        # embeddings_list is a list of tensors of shape [seq_len, embedding_dim]
        # Now we pad them to create a uniform tensor [batch_size, max_seq_len, embedding_dim].
        embeddings_padded = torch.nn.utils.rnn.pad_sequence(embeddings_list, batch_first=True)
        # Move to the correct device
        embeddings_padded = embeddings_padded.to(self.device)
        return embeddings_padded

    def prepare_features(self, df):
        """Prepare embeddings for the dataset."""
        texts = [
            self.prepare_text(str(text), mid)
            for text, mid in zip(df['text'], df['metaphorID'])
        ]
        print("Generating token-level SentenceTransformer embeddings...")
        embeddings = self.get_token_embeddings_batch(texts)
        # embeddings: [batch_size, seq_len, input_dim]
        return embeddings

    def fit(self, X, y, epochs=10, batch_size=32):
        """Train the Attention-based NN model."""
        self.fc_layers.train()
        self.attention.train()

        # Prepare data
        X_embeddings = self.prepare_features(X)
        y = torch.tensor(y.values, dtype=torch.long).to(self.device)

        dataset = TensorDataset(X_embeddings, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()

                # Apply attention mechanism
                attended_output, _ = self.attention(batch_X)  # [batch, input_dim]

                # Residual connection:
                # For example, take the first token embedding (often analogous to [CLS]).
                cls_embedding = batch_X[:, 0, :]  # [batch, input_dim]

                # Combine attended_output and cls_embedding
                combined_output = torch.cat([attended_output, cls_embedding], dim=1)  # [batch, 2*input_dim]

                # Fully connected layers
                outputs = self.fc_layers(combined_output)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    def predict(self, X):
        self.fc_layers.eval()
        self.attention.eval()
        X_embeddings = self.prepare_features(X)

        with torch.no_grad():
            attended_output, _ = self.attention(X_embeddings)
            cls_embedding = X_embeddings[:, 0, :]
            combined_output = torch.cat([attended_output, cls_embedding], dim=1)
            outputs = self.fc_layers(combined_output)
            _, predictions = torch.max(outputs, dim=1)

        return predictions.cpu().numpy()

    def predict_proba(self, X):
        self.fc_layers.eval()
        self.attention.eval()
        X_embeddings = self.prepare_features(X)

        with torch.no_grad():
            attended_output, _ = self.attention(X_embeddings)
            cls_embedding = X_embeddings[:, 0, :]
            combined_output = torch.cat([attended_output, cls_embedding], dim=1)
            probabilities = self.fc_layers(combined_output)

        return probabilities.cpu().numpy()

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return classification_report(y_test, predictions)


class RandomForestMetaphorDetector:
    def __init__(self):
        # Dictionary mapping metaphorID to target words
        self.metaphor_words = {
            0: 'road', 1: 'candle', 2: 'light', 3: 'spice',
            4: 'ride', 5: 'train', 6: 'boat'
        }

        # Initialize the pipeline with same parameters as original
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
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    def add_metaphor_context(self, text, metaphor_id):
        target_word = self.metaphor_words[metaphor_id]
        return f"{text} TARGET_WORD_{target_word}"

    def prepare_features(self, df):
        processed_texts = df['text'].apply(self.preprocess_text)
        processed_texts = [
            self.add_metaphor_context(text, mid)
            for text, mid in zip(processed_texts, df['metaphorID'])
        ]
        return processed_texts

    def fit(self, X, y):
        processed_X = self.prepare_features(X)
        self.pipeline.fit(processed_X, y)

    def predict(self, X):
        processed_X = self.prepare_features(X)
        return self.pipeline.predict(processed_X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return classification_report(y_test, predictions)


# Second model - Naive Bayes
class NaiveBayesMetaphorDetector:
    def __init__(self):
        self.metaphor_words = {
            0: 'road', 1: 'candle', 2: 'light', 3: 'spice',
            4: 'ride', 5: 'train', 6: 'boat'
        }

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('classifier', MultinomialNB())
        ])

        self.param_grid = {
            'tfidf__max_features': [500, 1000, 1500],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'classifier__alpha': [0.1, 0.5, 1.0]
        }

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    def add_metaphor_context(self, text, metaphor_id):
        target_word = self.metaphor_words[metaphor_id]
        return f"{text}"

    def prepare_features(self, df):
        processed_texts = df['text'].apply(self.preprocess_text)
        processed_texts = [
            self.add_metaphor_context(text, mid)
            for text, mid in zip(processed_texts, df['metaphorID'])
        ]
        return processed_texts

    def fit(self, X, y):
        processed_X = self.prepare_features(X)
        grid_search = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(processed_X, y)
        self.pipeline = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)

    def predict(self, X):
        processed_X = self.prepare_features(X)
        return self.pipeline.predict(processed_X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return classification_report(y_test, predictions)


# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')


def train_and_evaluate_models(train_path, output_dir='models'):
    """
    Train and evaluate all three models sequentially
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\nLoading data from:", train_path)
    df = pd.read_csv(train_path)

    # Split the data
    X = df[['metaphorID', 'text']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Random Forest Model
    print("\n" + "=" * 50)
    print("Training Random Forest Model...")
    rf_model = RandomForestMetaphorDetector()
    rf_model.fit(X_train, y_train)
    print("\nRandom Forest Model Performance:")
    print(rf_model.evaluate(X_test, y_test))

    # 2. Naive Bayes Model
    print("\n" + "=" * 50)
    print("Training Naive Bayes Model...")
    nb_model = NaiveBayesMetaphorDetector()
    nb_model.fit(X_train, y_train)
    print("\nNaive Bayes Model Performance:")
    print(nb_model.evaluate(X_test, y_test))

    # 3. Deep Learning Model
    print("\n" + "=" * 50)
    print("Training Deep Learning Model...")
    dl_model = SentenceTransformerAttentionNN()
    dl_model.fit(X_train, y_train, epochs=5)  # Reduced epochs for demonstration
    print("\nDeep Learning Model Performance:")
    print(dl_model.evaluate(X_test, y_test))

    # Save the deep learning model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f'dl_model_{timestamp}.pt')

    # Save just the model state dictionaries
    torch.save({
        'attention_state_dict': dl_model.attention.state_dict(),
        'fc_layers_state_dict': dl_model.fc_layers.state_dict()
    }, model_path)

    print(f"\nDeep Learning model saved to: {model_path}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train metaphor detection models')
    parser.add_argument('train_path', type=str, help='Path to the training CSV file')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save the trained models')

    # Parse arguments
    args = parser.parse_args()

    # Train and evaluate models
    train_and_evaluate_models(args.train_path, args.output_dir)


if __name__ == "__main__":
    print("This script will first run: Random Forrest Approach (Baseline)")
    print("This script will then run: Naive Bayes Approach")
    print("This script will finally run: Attention Based Residual Network (Best)")
    main()