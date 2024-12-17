import argparse
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from sklearn.metrics import classification_report


class Attention(nn.Module):
    """Attention mechanism to weigh embedding features."""

    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        scores = self.attention_weights(x).squeeze(-1)
        scores = torch.softmax(scores, dim=1)
        weighted_output = torch.bmm(scores.unsqueeze(1), x).squeeze(1)
        return weighted_output, scores


class SentenceTransformerAttentionNN:
    def __init__(self, model_name='all-mpnet-base-v2', input_dim=768, hidden_dim=128, num_classes=2):
        self.st_model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.attention = Attention(input_dim).to(self.device)
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        ).to(self.device)

        self.metaphor_words = {
            0: 'road', 1: 'candle', 2: 'light', 3: 'spice',
            4: 'ride', 5: 'train', 6: 'boat'
        }

    def prepare_text(self, text, metaphor_id):
        target_word = self.metaphor_words[metaphor_id]
        return f"{text} [SEP] The target word is {target_word}"

    def get_token_embeddings_batch(self, texts):
        embeddings_list = self.st_model.encode(
            texts,
            batch_size=32,
            convert_to_tensor=True,
            output_value='token_embeddings'
        )
        embeddings_padded = torch.nn.utils.rnn.pad_sequence(embeddings_list, batch_first=True)
        embeddings_padded = embeddings_padded.to(self.device)
        return embeddings_padded

    def prepare_features(self, df):
        texts = [
            self.prepare_text(str(text), mid)
            for text, mid in zip(df['text'], df['metaphorID'])
        ]
        print("Generating embeddings for test data...")
        embeddings = self.get_token_embeddings_batch(texts)
        return embeddings

    def load_model(self, model_path):
        """Load the saved model state"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.attention.load_state_dict(checkpoint['attention_state_dict'])
        self.fc_layers.load_state_dict(checkpoint['fc_layers_state_dict'])
        print(f"Model loaded from {model_path}")

    def predict(self, X):
        """Make predictions on new data"""
        self.attention.eval()
        self.fc_layers.eval()

        X_embeddings = self.prepare_features(X)

        with torch.no_grad():
            attended_output, _ = self.attention(X_embeddings)
            cls_embedding = X_embeddings[:, 0, :]
            combined_output = torch.cat([attended_output, cls_embedding], dim=1)
            outputs = self.fc_layers(combined_output)
            _, predictions = torch.max(outputs, dim=1)

        return predictions.cpu().numpy()


def run_test(model_path, test_path):
    """Run the test data through the model and print results"""
    # Load test data
    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)

    # Initialize model
    model = SentenceTransformerAttentionNN()

    # Load saved model state
    model.load_model(model_path)

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(test_df[['metaphorID', 'text']])

    # If the test data has labels, compute and show metrics
    if 'label' in test_df.columns:
        print("\nTest Set Performance:")
        print(classification_report(test_df['label'], predictions))

    # Add predictions to dataframe
    test_df['predicted_label'] = predictions

    # Save results
    output_path = test_path.rsplit('.', 1)[0] + '_predictions.csv'
    test_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test metaphor detection model')
    parser.add_argument('model_path', type=str,
                        help='Path to the saved model file (.pt)')
    parser.add_argument('test_path', type=str,
                        help='Path to the test CSV file')

    args = parser.parse_args()
    run_test(args.model_path, args.test_path)


if __name__ == "__main__":
    print("This script will use Attention Based Residual Network to predict labels of test data")
    main()