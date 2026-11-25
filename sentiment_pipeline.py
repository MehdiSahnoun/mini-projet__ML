"""Point d'entrée commun de la classification de sentiment (étape 1).

Charge l'artefact de classification (vectorizer + modèle) sauvegardé par
``train_model.py`` et expose :

- ``clean_text`` : nettoyage minimal d'un texte brut (minuscules, ponctuation
  enlevée, stopwords supprimés) — la même fonction est utilisée au
  vectorisation et à la prédiction, pour rester cohérent ;
- ``load_model`` : charge (en cache) le vectoriseur + le classifieur ;
- ``predict_text`` : renvoie le sentiment prédit, sa confiance et le texte
  extrait, identiques quelle que soit la source d'entrée.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import joblib

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Stopwords français (liste courte, suffisante pour un projet pédagogique).
STOP_WORDS = {
    "le", "la", "les", "l", "un", "une", "des", "du", "de", "au", "aux",
    "son", "sa", "ses", "leur", "leurs", "notre", "nos", "votre", "vos",
    "mon", "ma", "mes", "ce", "cet", "cette", "ces", "et", "ou", "mais",
    "donc", "or", "ni", "car", "que", "quoi", "où", "dont", "en", "dans",
    "pour", "par", "sur", "avec", "sans", "comme", "plus", "moins", "très",
    "trop", "assez", "alors", "aussi", "maintenant", "encore", "toujours",
    "jamais", "ne", "est", "être", "avoir", "ont", "sont", "a", "y",
    "je", "tu", "il", "elle", "ils", "elles", "nous", "vous", "se", "me",
    "te", "la", "il", "dès", "devant", "derrière", "sinon", "mais",
}

# Exécutables d'aide : tesseract (OCR) et ffmpeg (décodage audio) ne sont
# pas forcément sur le PATH système.
TOOL_DIRS = (
    PROJECT_DIR / "bin",
    Path(r"C:\Program Files\Tesseract-OCR"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR"),
)


def configure_external_tools_path() -> None:
    """Rend tesseract.exe et ffmpeg.exe accessibles via ``os.environ["PATH"]``."""
    current = os.environ.get("PATH", "")
    for directory in TOOL_DIRS:
        if directory.is_dir():
            entry = f"{directory};"
            if entry not in current:
                os.environ["PATH"] = f"{directory};{current}"


configure_external_tools_path()


def clean_text(text: str) -> str:
    """Nettoyage minimal du texte : minuscules, suppression des ponctuations
    et des stopwords, normalisation des espaces."""
    if not text:
        return ""
    lowered = text.lower()
    # Remplace tout ce qui n'est pas une lettre/chiffre/underscore/espace par un espace
    ascii_text = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    tokens = [token for token in ascii_text.split() if token and token not in STOP_WORDS]
    return " ".join(tokens)


@lru_cache(maxsize=1)
def load_model():
    """Charge le vectoriseur et le classifieur (caché en mémoire)."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable ({MODEL_PATH}). Veuillez d'abord lancer "
            f"la commande : python train_model.py"
        )
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return model, vectorizer


def predict_text(text: str, source: str = "text") -> dict:
    """Prédit le sentiment d'un texte (le texte peut venir de n'importe
    quelle source : texte brut, OCR ou transcription).

    Returns:
        dict contenant : ``source``, ``texte``, ``texte_extrait``,
        ``sentiment_predit``, ``confiance``, ``classes_possibles``.
    """
    if text is None or not str(text).strip():
        raise ValueError("Texte vide : rien à classifier")

    model, vectorizer = load_model()
    texte_extrait = clean_text(str(text))
    if not texte_extrait:
        texte_extrait = str(text)

    features = vectorizer.transform([texte_extrait])
    classes = [str(c) for c in model.classes_]
    predicted_label = str(model.predict(features)[0])
    try:
        predicted_index = classes.index(predicted_label)
    except ValueError:
        predicted_index = 0

    confiance = 1.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confiance = float(proba[predicted_index])

    return {
        "source": source,
        "texte": str(text),
        "texte_extrait": texte_extrait,
        "sentiment_predit": classes[predicted_index],
        "confiance": round(confiance, 4),
        "classes_possibles": classes,
    }
