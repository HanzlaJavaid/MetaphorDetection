# Metaphor Detection Model Guide

This guide explains how to use the training and testing scripts for metaphor detection.

## Prerequisites

Ensure you have the following Python packages installed:
```bash
pip install pandas numpy torch scikit-learn sentence-transformers nltk
```
or 
```bash
pip install -r requirements.csv
```


## Data Format

Both training and test data should be CSV files with the following columns:
- `metaphorID` (int): ID of the metaphor type (0-6)
- `text` (string): The text to analyze
- `label` (int): 1 for metaphorical usage, 0 for literal usage

Example CSV format:
```csv
metaphorID,text,label
0,"Life is a long road with many twists and turns",1
0,"The road to the store is under construction",0
1,"Hope is a candle in the darkness",1
1,"The candle on the table is melting",0
```

## Running the Training Script

The training script (`run_train.py`) will train three models sequentially:
1. Random Forest
2. Naive Bayes
3. Deep Learning model with attention

### Usage:
```bash
python run_train.py path/to/training_data.csv --output_dir models
```

### Example:
```bash
python run_train.py data/train.csv --output_dir trained_models
```

### Expected Output:
```
Loading data from: data/train.csv

==================================================
Training Random Forest Model...
Random Forest Model Performance:


==================================================
Training Naive Bayes Model...
Best Parameters: {'tfidf__max_features': 1000, 'tfidf__ngram_range': (1, 2), 'classifier__alpha': 0.5}
Naive Bayes Model Performance:

==================================================
Training Deep Learning Model...
[Training progress...]
Deep Learning Model Performance:


Deep Learning model saved to: trained_models/dl_model_20241216_123456.pt
```

## Running the Test Script

The test script (`run_test.py`) loads a trained deep learning model and makes predictions on new data.

### Usage:
```bash
python run_test.py path/to/model.pt path/to/test_data.csv
```

### Example:
```bash
python run_test.py trained_models/dl_model_20241216_123456.pt data/test.csv
```

### Expected Output:
```
Loading test data from data/test.csv
Model loaded from trained_models/dl_model_20241216_123456.pt
Generating embeddings for test data...
Making predictions...

Test Set Performance:

Predictions saved to: data/test_predictions.csv
```

The output file `test_predictions.csv` will contain all original columns plus a new `predicted_label` column with the model's predictions.


## Model Files

After training, you'll find these files in your output directory:
- `dl_model_[timestamp].pt`: The saved deep learning model

The timestamp format is YYYYMMDD_HHMMSS, making each model file unique.
