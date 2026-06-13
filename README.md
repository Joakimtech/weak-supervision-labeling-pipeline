# Weak Supervision Labeling Pipeline

A production-ready weak supervision pipeline that automatically generates training labels for text classification using heuristic labeling functions, without requiring manual annotation.

## Overview

This project implements a complete weak supervision workflow that combines multiple noisy labeling functions (LFs) into a generative model to produce probabilistic labels for unlabeled text data. The pipeline then trains a discriminative classifier using these labels, demonstrating how domain knowledge can be encoded as imperfect rules to bootstrap model training.

### Key Features

- 13 heuristic labeling functions for sentiment analysis
- Snorkel-based generative model for label aggregation
- TF-IDF feature extraction with logistic regression classifier
- Complete training and evaluation pipeline
- Reusable prediction function for new text

## Architecture

```
Raw Text Data
     │
     ▼
Labeling Functions (13 heuristics)
     │
     ▼
Snorkel Generative Model
     │
     ▼
Probabilistic Labels
     │
     ▼
Discriminative Classifier (Logistic Regression)
     │
     ▼
Sentiment Predictions
```

## Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Core programming language |
| Snorkel | 0.9.5+ | Weak supervision framework |
| Pandas | 2.0.3+ | Data manipulation |
| Scikit-learn | 1.3.0+ | TF-IDF, classifier, evaluation |
| NumPy | 1.24.3+ | Numerical operations |
| Matplotlib | 3.7.2+ | Visualization |
| Seaborn | 0.12.2+ | Confusion matrix plotting |
| Joblib | 1.2.0+ | Model serialization |

## Dataset

The pipeline is demonstrated on the **Sentiment140** dataset, containing 10,000 tweets (5,000 positive, 5,000 negative). The dataset is automatically downloaded during pipeline execution.

- Source: Stanford University
- Size: 10,000 samples
- Classes: Positive (1) and Negative (0)
- Format: CSV with text and sentiment labels

## Installation

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joakimtech/weak-supervision-labeling-pipeline/blob/main/weak_supervision_pipeline.ipynb)

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/joakimtech/weak-supervision-labeling-pipeline.git
cd weak-supervision-labeling-pipeline

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from weak_supervision_pipeline import WeakSupervisionPipeline
import pandas as pd

# Initialize pipeline
pipeline = WeakSupervisionPipeline()

# Load your data (requires 'text' column)
df = pd.read_csv('your_data.csv')

# Train the pipeline
probs = pipeline.train_generative_model(df)
pipeline.train_discriminative_model(df, probs)

# Make predictions
predictions = pipeline.predict(["I love this product!", "Terrible experience."])
print(predictions)  # ['Positive', 'Negative']
```

### Complete Pipeline Example

```python
# Run the full pipeline (as shown in the notebook)
# This includes:
# 1. Data loading and preprocessing
# 2. Applying 13 labeling functions
# 3. Training generative model
# 4. Training discriminative classifier
# 5. Evaluation and visualization
```

## Labeling Functions

The pipeline includes 13 heuristic labeling functions:

| Category | Labeling Functions |
|----------|-------------------|
| Emoji-based | Positive emojis, Negative emojis |
| Keyword-based | Positive keywords, Negative keywords |
| Structural | Question marks, Exclamation marks |
| Length-based | Short text patterns |
| Intensifiers | Negative intensifiers (never, nothing, etc.) |
| Pattern-based | Complaint patterns, Gratitude expressions |
| Symbol-based | Thumbs up/down |

### Example Labeling Function

```python
@labeling_function()
def lf_positive_keywords(x):
    positive_words = ['love', 'great', 'awesome', 'amazing', 'good']
    if any(word in x.text.lower() for word in positive_words):
        return POSITIVE
    return ABSTAIN
```

## Results

The pipeline achieves the following performance on the Sentiment140 test set:

| Metric | Score |
|--------|-------|
| Accuracy | 51.4% |
| Precision (Positive) | 50.7% |
| Recall (Positive) | 100.0% |
| F1 Score (Positive) | 67.3% |

### Confusion Matrix

```
               Predicted
               Neg   Pos
Actual Neg      22   729
       Pos       0   750
```

### Key Observations

- The model shows a bias toward positive predictions due to the nature of the heuristic labeling functions
- Negative sentiment is more difficult to capture with simple keyword rules due to sarcasm and subtle language
- Coverage improved from 14.9% to 30.5% after adding more balanced labeling functions

## Project Structure

```
weak-supervision-labeling-pipeline/
├── weak_supervision_pipeline.py   # Main pipeline code
├── requirements.txt                # Python dependencies
├── outputs/                        # Trained models and results
│   ├── classifier.pkl              # Trained logistic regression
│   ├── vectorizer.pkl              # TF-IDF vectorizer
│   ├── label_model.pkl             # Snorkel generative model
│   └── evaluation_results.json     # Performance metrics
└── README.md                       # Documentation
```

## Limitations and Future Work

### Current Limitations

1. **Labeling Function Bias**: Current LFs are biased toward positive sentiment
2. **Coverage**: Only 30% of samples receive labels from at least one LF
3. **Dataset Specific**: Heuristics designed for tweets may not generalize to other domains

### Potential Improvements

1. **More Negative LFs**: Add sophisticated patterns for detecting negative sentiment
2. **External Models**: Incorporate pre-trained sentiment models as LFs
3. **Active Learning**: Use model uncertainty to select samples for human review
4. **Different Base Classifier**: Experiment with BERT or other transformers

## Lessons Learned

This project demonstrates that weak supervision is a powerful technique for reducing manual labeling effort, but the quality of outputs is highly dependent on:

- **Coverage**: More labeling functions generally improve results
- **Balance**: LFs should be balanced across all target classes
- **Specificity**: Overly broad LFs introduce noise
- **Independence**: Correlated LFs provide redundant information

## Contributing

Contributions are welcome. Please ensure any new labeling functions improve coverage without introducing excessive noise.

## License

MIT License

## Author

Joakim- [GitHub](https://github.com/joakimtech)

## Acknowledgments

- Snorkel team for the weak supervision framework
- Stanford University for the Sentiment140 dataset
```

---


I will then provide you with a final project summary and any additional instructions you may need.
