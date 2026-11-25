"""Module de transcription ASR (étape 2 - audio).

Transcrit des fichiers audio (.wav, .mp3, .m4a, .flac, .ogg, ...) en texte
avec openai-whisper (modèle "base"). Le texte transcrit est ensuite
transmis au classifieur texte via ``sentiment_pipeline``.

Prérequis : le binaire ``ffmpeg`` est fourni dans ``./bin/`` (voir
``sentiment_pipeline.configure_external_tools_path()``) pour le décodage
des fichiers audio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import whisper

from sentiment_pipeline import configure_external_tools_path

configure_external_tools_path()

_WHISPER_MODEL = None


def get_whisper_model(model_name: str = "base"):
    """Charge (et met en cache) le modèle whisper demandé."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model(model_name)
    return _WHISPER_MODEL


def transcribe_audio(audio_path, language: Optional[str] = "fr") -> str:
    """Transcrit un fichier audio en texte brut.

    Le décodage est effectué par whisper via ffmpeg (dans PATH).
    """
    model = get_whisper_model()
    result = model.transcribe(str(audio_path), language=language, verbose=False)
    return (result.get("text") or "").strip()


if __name__ == "__main__":
    # Démo rapide : transcription des fichiers fournis par le dataset
    from pathlib import Path

    base = Path(__file__).resolve().parent / "dataset_sentiment_ecommerce"
    for audio_path in sorted((base / "audios").glob("*.mp3"))[:3]:
        print(f"--- {audio_path.name} ---")
        print(transcribe_audio(audio_path))
