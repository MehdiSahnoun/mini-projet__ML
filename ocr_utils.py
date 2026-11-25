"""Module de traitement OCR (étape 2 - images).

Prend en entrée une image (depuis le dossier ``images/`` ou téléversée via
l'API) et extrait le texte brut avec ``pytesseract`` + Pillow. Le texte
extrait est ensuite transmis au classifieur texte (``sentiment_pipeline``).

Prérequis : le binaire ``tesseract.exe`` doit être accessible via PATH
(``sentiment_pipeline.configure_external_tools_path()`` l'ajoute s'il est
installé dans ``C:\\Program Files\\Tesseract-OCR``).
"""

from __future__ import annotations

import pytesseract
from PIL import Image

from sentiment_pipeline import configure_external_tools_path

configure_external_tools_path()


def get_available_languages() -> list[str]:
    """Langues OCR disponibles (par ex. ['eng', 'fra'] via tessdata)."""
    try:
        return list(pytesseract.get_languages(config=""))
    except Exception:
        return ["eng"]


def extract_text_from_image(image_path) -> str:
    """Ouvre une image et extrait le texte brute via Tesseract.

    L'OCR français est utilisé si la langue ``fra`` est disponible dans
    ``tessdata``, sinon la langue par défaut (eng).
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        languages = get_available_languages()
        lang = "fra" if "fra" in languages else None
        text = pytesseract.image_to_string(img, lang=lang)
    return (text or "").strip()


if __name__ == "__main__":
    # Démo rapide : OCR des images fournies par le dataset
    from pathlib import Path

    base = Path(__file__).resolve().parent / "dataset_sentiment_ecommerce"
    for image_path in sorted((base / "images").glob("*.png"))[:3]:
        print(f"--- {image_path.name} ---")
        print(extract_text_from_image(image_path))
