# src/core/model.py
import numpy as np
from sklearn.ensemble import IsolationForest
from ..config import TRAIN_MAX_ROWS

def fit_if(X):
    if len(X) > TRAIN_MAX_ROWS:
        X = X.sample(n=TRAIN_MAX_ROWS, random_state=42)
    return IsolationForest(
        n_estimators=150,        # a bit lighter
        max_samples='auto',
        contamination='auto',
        random_state=42,
        n_jobs=-1
    ).fit(X)

def score_if(model, X):
    s = -model.score_samples(X)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return s
