"""Mini-application web de test (étape 4 - Option B : Streamlit).

Lancez l'API FastAPI en arrière-plan :
    python -m uvicorn main:app --reload

Puis lancez cette interface :
    streamlit run webapp/streamlit_app.py
"""

from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://localhost:8000"
EMOJI = {"positif": "😊", "negatif": "☹️", "neutre": "😐"}

st.set_page_config(page_title="Classifieur de Sentiment", page_icon="📝", layout="centered")

st.title("Classifieur de Sentiment Multi-source")
st.caption("Mini-app de démonstration de l'API FastAPI : texte, image, audio")

# --- Vérification de la connectivité à l'API -------------------------------
api_url = st.sidebar.text_input("URL de l'API", value=API_URL)
try:
    health = requests.get(f"{api_url}/health", timeout=5)
    if health.status_code == 200:
        st.sidebar.success(f"API connectée — statut : {health.json()['status']}")
    else:
        st.sidebar.warning("L'API répond mais le statut est non ok")
except requests.RequestException:
    st.sidebar.warning(
        "API non jointe. Démarrez l'API d'abord : `python -m uvicorn main:app --reload`"
    )


def display_result(result: dict) -> None:
    sentiment = result.get("sentiment_predit", "")
    emoji = EMOJI.get(sentiment, "🤔")
    st.metric("Sentiment prédit", f"{sentiment} {emoji}")
    st.write(f"**Confiance :** {result['confiance'] * 100:.1f} %")
    st.progress(result["confiance"])
    st.caption(f"Source d'entrée : {result.get('source')}")
    if result.get("texte_extrait"):
        st.write("**Texte effectivement classifié :**")
        st.code(result["texte_extrait"])


def call_api(endpoint: str, method: str = "post", timeout: int = 120, **kwargs):
    response = getattr(requests, method)(f"{api_url}{endpoint}", timeout=timeout, **kwargs)
    if response.status_code == 200:
        display_result(response.json())
    else:
        st.error(f"Erreur HTTP {response.status_code} : {response.text}")


def handle_error(error: requests.RequestException) -> None:
    st.error(
        f"Erreur de connexion à l'API ({api_url}). "
        "Démarrez-la d'abord : `python -m uvicorn main:app --reload`"
    )


# --- Choix de la source d'entrée --------------------------------------------
mode = st.radio("Choisissez la source d'entrée", ["Texte", "Image", "Audio"], horizontal=True)

if mode == "Texte":
    texte = st.text_area(
        "Votre avis client",
        height=120,
        placeholder="Ex : Livraison rapide et produit parfait ! Je recommande.",
    )
    if st.button("Prédire le sentiment") and texte.strip():
        try:
            call_api("/predict/text", json={"texte": texte})
        except requests.RequestException:
            handle_error(None)
    elif not texte.strip():
        st.info("Veuillez saisir un texte non vide.")

elif mode == "Image":
    fichier = st.file_uploader(
        "Chargez une image (capture d'avis ou ticket scanné)",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
    )
    if fichier is not None and st.button("Prédire le sentiment"):
        try:
            call_api(
                "/predict/image",
                files={"file": (fichier.name, fichier.getvalue(), fichier.type or "image/png")},
            )
        except requests.RequestException:
            handle_error(None)

elif mode == "Audio":
    fichier = st.file_uploader(
        "Chargez un fichier audio (avis vocal : wav, mp3, ...)",
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
    )
    if fichier is not None and st.button("Prédire le sentiment"):
        try:
            call_api(
                "/predict/audio",
                files={"file": (fichier.name, fichier.getvalue(), fichier.type or "audio/mpeg")},
            )
        except requests.RequestException:
            handle_error(None)
