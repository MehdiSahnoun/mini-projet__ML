"""API REST unifiée pour le classifieur de sentiment multi-source (étape 3).

Routes :
- ``POST /predict/text``   : texte brut (corps JSON ``{"texte": ...}``)
- ``POST /predict/image``  : fichier image (multipart) -> OCR -> classifieur
- ``POST /predict/audio``  : fichier audio (multipart) -> transcription ASR -> classifieur
- ``POST /predict``        : détection automatique du type (fichier ou JSON ``{"texte": ...}``)
- ``GET  /health``         : état du service (modèle chargé)
- ``GET  /``               : présentation

La réponse est un JSON unifié pour toutes les sources :
``{
    "source": "text" | "image" | "audio" | "auto",
    "texte": "...",               # texte fourni (si applicable)
    "texte_extrait": "...",       # texte effectivement classifié
    "sentiment_predit": "positif" | "negatif" | "neutre",
    "confiance": 0.87,
    "classes_possibles": ["negatif", "neutre", "positif"]
}``

Lancement : ``python -m uvicorn main:app --reload``
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from asr_utils import transcribe_audio
from ocr_utils import extract_text_from_image
from sentiment_pipeline import load_model, predict_text

app = FastAPI(
    title="Classifieur de Sentiment Multi-source",
    version="1.0.0",
    description=(
        "Unifie les entrées texte, image et audio. "
        "L'architecture est en cascade : OCR/ASR puis classifieur texte."
    ),
)

# --- Validation des types et tailles d'entrées ------------------------------
IMAGE_MIME_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/pjpeg", "image/gif",
    "image/webp", "image/bmp", "image/x-ms-bmp", "image/tiff", "image/x-tiff",
}
AUDIO_MIME_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mp4", "audio/aac", "audio/ogg", "audio/flac", "audio/x-m4a", "audio/m4a",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

MAX_TEXT_LENGTH = 10_000
MAX_IMAGE_BYTES = 5 * 1024 * 1024      # 5 Mo
MAX_AUDIO_BYTES = 20 * 1024 * 1024     # 20 Mo

TEMP_DIR = Path(tempfile.gettempdir())


class TextRequest(BaseModel):
    """Charge utile pour la prédiction sur texte brut."""

    texte: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH, description="Texte à classifier")


class PredictionResponse(BaseModel):
    """Réponse unifiée pour toutes les sources."""

    source: str = Field(description="Source d'entrée : text / image / audio / auto")
    texte: str = Field(description="Texte fourni (vide pour les sources OCR/ASR)")
    texte_extrait: str = Field(description="Texte effectivement classifié")
    sentiment_predit: str = Field(description="Classification finale : positif / negatif / neutre")
    confiance: float = Field(description="Probabilité de la classe prédite (0-1)")
    classes_possibles: list[str] = Field(description="Classes connues par le modèle")


# --- Fonctions utilitaires ----------------------------------------------------


async def read_upload_bytes(
    file: UploadFile,
    allowed_mimes: set[str],
    max_bytes: int,
    kind: str,
) -> bytes:
    """Lit les octets d'un fichier téléversé, en validant taille et type MIME."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail=f"{kind} : fichier vide")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{kind} : taille {len(data)} octets > limite de {max_bytes} octets",
        )
    mime = (file.content_type or "").strip().lower()
    if mime and mime not in allowed_mimes:
        raise HTTPException(
            status_code=422,
            detail=f"{kind} : type MIME non supporté : {mime}",
        )
    return data


def save_temp_file(data: bytes, suffix: str) -> Path:
    """Sauvegarde temporaire d'un fichier téléversé (effacé à la fin de la route)."""
    fd, path = tempfile.mkstemp(prefix="sentiment_", suffix=suffix, dir=TEMP_DIR)
    with os.fdopen(fd, "wb") as tmp:
        tmp.write(data)
    return Path(path)


def detect_source(filename: str, mime: str) -> Optional[str]:
    """Détecte la source à partir du type MIME ou de l'extension du fichier."""
    extension = Path(filename or "").suffix.lower()
    if mime.startswith("image/") or extension in IMAGE_EXTENSIONS:
        return "image"
    if mime.startswith("audio/") or extension in AUDIO_EXTENSIONS:
        return "audio"
    return None


async def predict_image_route(file: UploadFile) -> dict:
    """OCR (pytesseract) -> prédiction du sentiment."""
    data = await read_upload_bytes(file, IMAGE_MIME_TYPES, MAX_IMAGE_BYTES, "L'image")
    extension = Path(file.filename or "").suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        extension = ".png"
    image_path = save_temp_file(data, extension)
    try:
        texte_extrait = extract_text_from_image(image_path)
    finally:
        image_path.unlink(missing_ok=True)

    if not texte_extrait:
        raise HTTPException(status_code=422, detail="Aucun texte extrait de l'image par OCR")
    return predict_text(texte_extrait, source="image")


async def predict_audio_route(file: UploadFile) -> dict:
    """Transcription ASR (whisper) -> prédiction du sentiment."""
    data = await read_upload_bytes(file, AUDIO_MIME_TYPES, MAX_AUDIO_BYTES, "L'audio")
    extension = Path(file.filename or "").suffix.lower()
    if extension not in AUDIO_EXTENSIONS:
        extension = ".wav"
    audio_path = save_temp_file(data, extension)
    try:
        texte = transcribe_audio(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)

    if not texte:
        raise HTTPException(status_code=422, detail="Aucun texte transcrit de l'audio")
    return predict_text(texte, source="audio")


# --- Routes -------------------------------------------------------------------


@app.post("/predict/text", response_model=PredictionResponse)
def predict_text_route(payload: TextRequest) -> dict:
    """Prédit le sentiment d'un texte fourni en JSON."""
    return predict_text(payload.texte, source="text")


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image_endpoint(
    file: UploadFile = File(..., description="Fichier image (.png, .jpg, .jpeg, ...)"),
) -> dict:
    """Prédit le sentiment d'une image (extraction via OCR)."""
    return await predict_image_route(file)


@app.post("/predict/audio", response_model=PredictionResponse)
async def predict_audio_endpoint(
    file: UploadFile = File(..., description="Fichier audio (.wav, .mp3, .m4a, ...)"),
) -> dict:
    """Prédit le sentiment d'un fichier audio (transcription via ASR)."""
    return await predict_audio_route(file)


@app.post("/predict", response_model=PredictionResponse)
async def predict_auto_endpoint(request: Request) -> dict:
    """Détecte automatiquement le type d'entrée :

    - fichier image (multipart) -> ``/predict/image`` (OCR)
    - fichier audio (multipart) -> ``/predict/audio`` (ASR)
    - JSON ``{"texte": "..."}``  -> ``/predict/text``
    """
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()

    if content_type == "application/json":
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="Corps JSON invalide") from None
        if not isinstance(body, dict) or not body.get("texte"):
            raise HTTPException(
                status_code=422,
                detail="Corps JSON attendu : {\"texte\": \"votre texte ici\"}",
            )
        return predict_text(str(body["texte"]), source="auto")

    if content_type == "multipart/form-data":
        form = await request.form()
        file = form.get("file")
        if not file or not hasattr(file, "filename"):
            raise HTTPException(status_code=422, detail="Aucun fichier fourni (champ 'file')")
        filename = file.filename or ""
        source = detect_source(filename, file.content_type or "")
        if source is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Type de fichier non détecté. Envoyez un fichier image "
                    "(png/jpg/...) ou audio (wav/mp3/...) ou un JSON {\"texte\": ...}"
                ),
            )
        if source == "image":
            result = await predict_image_route(file)
        else:
            result = await predict_audio_route(file)
        result["source"] = "auto"
        return result

    raise HTTPException(
        status_code=422,
        detail="Format non supporté. Utilisez multipart/form-data avec un fichier ou application/json",
    )


@app.get("/health")
def health() -> JSONResponse:
    """Retourne l'état du service : les classes du modèle si chargé."""
    try:
        model, _ = load_model()
        classes = [str(c) for c in model.classes_]
        return {"status": "ok", "classes": classes}
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})


@app.get("/")
def root() -> dict:
    """Page d'accueil de l'API."""
    return {
        "message": "Classifieur de sentiment multi-source. "
                   "Utilisez POST /predict/text, /predict/image, /predict/audio ou /predict.",
        "docs": "/docs",
        "sources": ["text", "image", "audio", "auto"],
    }
