"""Tests unitaires et d'intégration de l'API (étape 5).

Exécution :
    python -m pytest test_api.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app, MAX_IMAGE_BYTES, MAX_AUDIO_BYTES
from sentiment_pipeline import MODEL_PATH, load_model

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset_sentiment_ecommerce"
IMAGES_DIR = DATASET_DIR / "images"
AUDIOS_DIR = DATASET_DIR / "audios"
IMAGE_EXAMPLE = IMAGES_DIR / "I01.png"
AUDIO_EXAMPLE = AUDIOS_DIR / "A01.mp3"

VALID_SENTIMENTS = {"positif", "negatif", "neutre"}

client = TestClient(app)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _assert_prediction_structure(body: dict, expected_source: str) -> None:
    assert body["source"] == expected_source
    assert body["sentiment_predit"] in VALID_SENTIMENTS
    assert isinstance(body["texte_extrait"], str) and len(body["texte_extrait"]) > 0
    assert isinstance(body["confiance"], float)
    assert 0.0 <= body["confiance"] <= 1.0
    assert isinstance(body["classes_possibles"], list)
    assert set(body["classes_possibles"]) >= VALID_SENTIMENTS


# --------------------------------------------------------------------------
# Tests : infrastructure
# --------------------------------------------------------------------------


def test_health_returns_status_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "classes" in body


def test_model_artifacts_exist():
    assert MODEL_PATH.exists()
    model, _ = load_model()
    assert [str(c) for c in model.classes_] == ["negatif", "neutre", "positif"]


def test_root_info():
    response = client.get("/")
    assert response.status_code == 200
    assert "predict" in response.json()["message"]


# --------------------------------------------------------------------------
# Tests : /predict/text
# --------------------------------------------------------------------------


def test_predict_text_happy_path():
    response = client.post(
        "/predict/text", json={"texte": "Le produit est excellent et la livraison rapide !"}
    )
    assert response.status_code == 200
    _assert_prediction_structure(response.json(), expected_source="text")


def test_predict_text_missing_field_422():
    response = client.post("/predict/text", json={})
    assert response.status_code == 422


def test_predict_text_empty_string_422():
    response = client.post("/predict/text", json={"texte": ""})
    assert response.status_code == 422


def test_predict_text_non_string_422():
    response = client.post("/predict/text", json={"texte": 12345})
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Tests : /predict/image
# --------------------------------------------------------------------------


@pytest.mark.skipif(not IMAGE_EXAMPLE.exists(), reason="Image d'exemple introuvable")
def test_predict_image_happy_path():
    response = client.post(
        "/predict/image",
        files={"file": (IMAGE_EXAMPLE.name, IMAGE_EXAMPLE.read_bytes(), "image/png")},
    )
    assert response.status_code == 200
    _assert_prediction_structure(response.json(), expected_source="image")


def test_predict_image_unsupported_mime_422():
    response = client.post(
        "/predict/image",
        files={"file": ("notes.txt", b"just some text", "text/plain")},
    )
    assert response.status_code == 422


def test_predict_image_empty_file_422():
    response = client.post(
        "/predict/image", files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 422


def test_predict_image_too_large_413():
    response = client.post(
        "/predict/image",
        files={"file": ("big.png", b"\x00" * (MAX_IMAGE_BYTES + 1), "image/png")},
    )
    assert response.status_code == 413


# --------------------------------------------------------------------------
# Tests : /predict/audio
# --------------------------------------------------------------------------


@pytest.mark.skipif(not AUDIO_EXAMPLE.exists(), reason="Audio d'exemple introuvable")
def test_predict_audio_happy_path():
    response = client.post(
        "/predict/audio",
        files={"file": (AUDIO_EXAMPLE.name, AUDIO_EXAMPLE.read_bytes(), "audio/mpeg")},
    )
    assert response.status_code == 200
    _assert_prediction_structure(response.json(), expected_source="audio")


def test_predict_audio_unsupported_mime_422():
    response = client.post(
        "/predict/audio",
        files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert response.status_code == 422


def test_predict_audio_too_large_413():
    response = client.post(
        "/predict/audio",
        files={"file": ("big.mp3", b"\x00" * (MAX_AUDIO_BYTES + 1), "audio/mpeg")},
    )
    assert response.status_code == 413


# --------------------------------------------------------------------------
# Tests : /predict (détection automatique)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not IMAGE_EXAMPLE.exists(), reason="Image d'exemple introuvable")
def test_predict_auto_with_image():
    response = client.post(
        "/predict",
        files={"file": (IMAGE_EXAMPLE.name, IMAGE_EXAMPLE.read_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "auto"
    _assert_prediction_structure(body, expected_source="auto")


def test_predict_auto_with_json_text():
    response = client.post("/predict", json={"texte": "Commande livrée ce matin, article conforme."})
    assert response.status_code == 200
    _assert_prediction_structure(response.json(), expected_source="auto")


def test_predict_auto_without_payload_422():
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_auto_json_without_text_422():
    response = client.post("/predict", json={"foo": "bar"})
    assert response.status_code == 422
