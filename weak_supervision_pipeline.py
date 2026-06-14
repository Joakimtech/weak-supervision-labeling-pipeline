"""
Weak Supervision Labeling Pipeline for Sentiment Analysis
==========================================================
This pipeline uses weak supervision to automatically label text data
using heuristic labeling functions, then trains a classifier.

Author: Joakimtech
Date: 2026
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
import joblib

# Constants
ABSTAIN = -1
NEGATIVE = 0
POSITIVE = 1

# ============================================
# LABELING FUNCTIONS
# ============================================

@labeling_function()
def lf_positive_emojis(x):
    positive_emojis = [':)', ':-)', ':D', ';-)', ';)', '😊', '😂', '❤️']
    if any(emoji in x.text for emoji in positive_emojis):
        return POSITIVE
    return ABSTAIN

@labeling_function()
def lf_negative_emojis(x):
    negative_emojis = [':(', ':-(', '😢', '😭', '😠', '👎']
    if any(emoji in x.text for emoji in negative_emojis):
        return NEGATIVE
    return ABSTAIN

@labeling_function()
def lf_positive_keywords(x):
    positive_words = ['love', 'great', 'awesome', 'amazing', 'good', 'happy', 'best', 'wonderful']
    if any(word in x.text.lower() for word in positive_words):
        return POSITIVE
    return ABSTAIN

@labeling_function()
def lf_negative_keywords(x):
    negative_words = ['hate', 'bad', 'terrible', 'awful', 'worst', 'sad', 'angry', 'disappointed']
    if any(word in x.text.lower() for word in negative_words):
        return NEGATIVE
    return ABSTAIN

@labeling_function()
def lf_question_negative(x):
    if x.text.endswith('?') and any(word in x.text.lower() for word in ['why', 'how', 'what']):
        return NEGATIVE
    return ABSTAIN

@labeling_function()
def lf_exclamation_positive(x):
    if x.text.endswith('!') and len(x.text) < 100:
        return POSITIVE
    return ABSTAIN

@labeling_function()
def lf_short_negative(x):
    if len(x.text.split()) < 5:
        negative_short = ['no', 'not', "don't", 'cant', 'bad']
        if any(word in x.text.lower() for word in negative_short):
            return NEGATIVE
    return ABSTAIN

@labeling_function()
def lf_negative_intensifiers(x):
    negative_intensifiers = ['never', 'nothing', 'nobody', 'nowhere', 'no one', 'cannot']
    if any(word in x.text.lower() for word in negative_intensifiers):
        return NEGATIVE
    return ABSTAIN

@labeling_function()
def lf_complaint_patterns(x):
    complaint_starts = ['i hate', 'i wish', 'why is', 'why does', 'cant believe']
    if any(x.text.lower().startswith(pattern) for pattern in complaint_starts):
        return NEGATIVE
    return ABSTAIN

@labeling_function()
def lf_thumbs_up(x):
    positive_symbols = ['👍', '💪', '🎉', '✨', '⭐']
    if any(symbol in x.text for symbol in positive_symbols):
        return POSITIVE
    return ABSTAIN

@labeling_function()
def lf_gratitude(x):
    gratitude = ['thank', 'thanks', 'appreciate', 'grateful', 'blessed']
    if any(word in x.text.lower() for word in gratitude):
        return POSITIVE
    return ABSTAIN

# ============================================
# MAIN PIPELINE CLASS
# ============================================

class WeakSupervisionPipeline:
    def __init__(self):
        self.labeling_functions = [
            lf_positive_emojis, lf_negative_emojis,
            lf_positive_keywords, lf_negative_keywords,
            lf_question_negative, lf_exclamation_positive, lf_short_negative,
            lf_negative_intensifiers, lf_complaint_patterns,
            lf_thumbs_up, lf_gratitude
        ]
        self.applier = PandasLFApplier(lfs=self.labeling_functions)
        self.label_model = None
        self.vectorizer = None
        self.classifier = None
    
    def train_generative_model(self, df_train, n_epochs=200):
        """Train the generative model on unlabeled data"""
        print("Applying labeling functions...")
        L_train = self.applier.apply(df=df_train)
        
        coverage = (L_train != ABSTAIN).any(axis=1).mean()
        print(f"Coverage: {coverage:.2%}")
        
        self.label_model = LabelModel(cardinality=2, verbose=False)
        self.label_model.fit(L_train, n_epochs=n_epochs, seed=42)
        
        probs = self.label_model.predict_proba(L_train)
        return probs
    
    def train_discriminative_model(self, df_train, probs):
        """Train the final classifier using probabilistic labels"""
        hard_labels = np.where(probs[:, 1] >= 0.5, 1, 0)
        
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train = self.vectorizer.fit_transform(df_train['text'].values)
        
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.classifier.fit(X_train, hard_labels)
    
    def predict(self, texts):
        """Predict sentiment for new texts"""
        if isinstance(texts, str):
            texts = [texts]
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        return ['Positive' if p == 1 else 'Negative' for p in predictions]
    
    def save(self, path_prefix='outputs/'):
        """Save trained models"""
        joblib.dump(self.classifier, f'{path_prefix}classifier.pkl')
        joblib.dump(self.vectorizer, f'{path_prefix}vectorizer.pkl')
        joblib.dump(self.label_model, f'{path_prefix}label_model.pkl')
    
    def load(self, path_prefix='outputs/'):
        """Load trained models"""
        self.classifier = joblib.load(f'{path_prefix}classifier.pkl')
        self.vectorizer = joblib.load(f'{path_prefix}vectorizer.pkl')
        self.label_model = joblib.load(f'{path_prefix}label_model.pkl')

print("Pipeline script created successfully")
