"""Étape 1 : modélisation et entraînement du classifieur texte (étape 1).

Charge ``textes.csv`` (dataset_sentiment_ecommerce), nettoie les textes
(voir ``sentiment_pipeline.clean_text``), vectorise avec ``TfidfVectorizer``,
entraîne un ``LogisticRegression`` sur les classes positif / négatif / neutre
et sauvegarde les artefacts dans ``models/`` (model.pkl, vectorizer.pkl,
metrics.json).

Exécution : python train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from sentiment_pipeline import (
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    VECTORIZER_PATH,
    clean_text,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset_sentiment_ecommerce"
TEXTES_CSV = DATASET_DIR / "textes.csv"

# Paramètres du modèle (étudiants : vous pouvez les retourner/compléter)
RANDOM_STATE = 42
MAX_ITER = 1000


def load_dataset() -> pd.DataFrame:
    """Charge le jeu de données textuel fourni."""
    df = pd.read_csv(TEXTES_CSV, encoding="utf-8")
    return df[["id", "texte", "sentiment_ref"]]


def build_training_corpus(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Nettoyage des textes et extraction des cibles.

    Le nettoyage est le même que celui appliqué à la prédiction
    (``sentiment_pipeline.clean_text``), garantissant la cohérence.
    """
    textes = [clean_text(texte) for texte in df["texte"]]
    cibles = df["sentiment_ref"].tolist()
    return textes, cibles


def train() -> tuple:
    """Entraîne le modèle et retourne (model, vectorizer, metrics)."""
    df = load_dataset()
    X_raw, y = build_training_corpus(df)

    # Vectorisation
    vectorizer = TfidfVectorizer()  # défaut : min_df=1, lowercase=True
    X = vectorizer.fit_transform(X_raw)

    # Entraînement
    model = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)
    model.fit(X, y)

    # Évaluation sur le jeu d'entraînement
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)

    metrics = {
        "model": "LogisticRegression (TfidfVectorizer)",
        "n_train_samples": len(df),
        "classes": [str(c) for c in model.classes_],
        "accuracy": round(float(report["accuracy"]), 4),
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }
    return model, vectorizer, metrics


def main() -> None:
    model, vectorizer, metrics = train()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Entraînement terminé ===")
    print(f"Classes : {metrics['classes']}")
    print(f"Exemples d'apprentissage : {metrics['n_train_samples']}")
    print(f"Accuracy (sur les données d'entraînement) : {metrics['accuracy']}")
    print(f"Taille du vocabulaire (tf-idf) : {metrics['vocabulary_size']}")
    print(classification_report_format(metrics))
    print("Modèles sauvegardés :")
    print(f"  - {VECTORIZER_PATH}")
    print(f"  - {MODEL_PATH}")
    print(f"  - {METRICS_PATH}")


def classification_report_format(metrics: dict) -> str:
    lines = ["Pécision / Rappel / F1 par classe :"]
    for classe, values in metrics["classification_report"].items():
        if isinstance(values, dict) and "precision" in values:
            lines.append(
                f"  - {classe}: precision={values['precision']:.3f} "
                f"recall={values['recall']:.3f} f1={values['f1-score']:.3f}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
