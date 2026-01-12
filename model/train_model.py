"""Train and save a cardiovacular risk model.

Usage:
  python model/train_model.py

This script trains a RandomForest pipeline on project_week/cardio_train.csv and
saves the pipeline to models/cardio_model.joblib
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, "project_week", "cardio_train.csv")
OUT_DIR = os.path.join(ROOT, "models")
OUT_MODEL = os.path.join(OUT_DIR, "cardio_model.joblib")

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, sep=';')

# Basic preprocessing
# Convert age (days) to years
if 'age' in df.columns:
    df['age_years'] = (df['age'] / 365).round(1)

features = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'smoke', 'alco', 'active']
X = df[features].fillna(0)
y = df['cardio']

print("Train/test split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ]
)

print("Training model...")
pipeline.fit(X_train, y_train)

print("Evaluating...")
proba = pipeline.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, proba))
print(classification_report(y_test, pipeline.predict(X_test)))

OUT_MODEL = os.path.join(OUT_DIR, "cardio_model.pkl")
print(f"Saving model to {OUT_MODEL}")
import pickle
with open(OUT_MODEL, "wb") as f:
    pickle.dump(pipeline, f)
print("Done.")