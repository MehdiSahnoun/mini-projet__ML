# Classifieur de Sentiment Multi-source (E-commerce Customer Voice)

**Mini-projet pédagogique** — Une plateforme e-commerce reçoit des milliers
d'avis clients sous trois formats hétérogènes : texte (avis web/chat), image
(captures d'écran ou tickets) et audio (avis vocaux). Le but : un système
unifié qui, quelle que soit la source, extrait l'information et prédit le
sentiment (positif / négatif / neutre), exposé via une API REST + une
interface de test.

**Architecture en cascade** : les entrées image (OCR via Tesseract) et audio
(transcription via Whisper) sont converties en texte brut, puis un
classifieur textuel unique (TF-IDF + LogisticRegression) prend la décision
finale pour les trois sources.

---

## 1. Architecture technique

```
┌───────────────────────────────────────────────────────────────────────┐
│                            API FastAPI                                │
│  POST /predict/text  │  POST /predict/image  │  POST /predict/audio  │
│  POST /predict (auto)│  GET /health │ GET /docs (Swagger)             │
└───────────────────────┬───────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Texte brut      OCR (pytesseract)  ASR (openai-whisper)
        └───────────────┼───────────────┘
                        ▼
            Classifieur texte (sentiment_pipeline.py)
                        ▼
        {"source", "texte", "texte_extrait", "sentiment_predit",
         "confiance", "classes_possibles"}
```

Modules du projet :

| Fichier | Rôle (étape pédagogique) |
| --- | --- |
| `sentiment_pipeline.py` | Classeur texte (`clean_text`, `load_model`, `predict_text`) |
| `train_model.py` | Étape 1 : prétraitement, vectorisation TF-IDF, entraînement LogisticRegression, évaluation, sauvegarde `models/` |
| `ocr_utils.py` | Étape 2 (images) : OCR via `pytesseract` + `Pillow` |
| `asr_utils.py` | Étape 2 (audio) : transcription via `openai-whisper` (+ `ffmpeg` pour le décodage) |
| `main.py` | Étape 3 : API REST FastAPI + schémas Pydantic |
| `webapp/streamlit_app.py` | Étape 4 : mini-app web de test (Option B) |
| `test_api.py` | Étape 5 : tests unitaires/d'intégration (pytest + httpx) |
| `requirements.txt` | Dépendances |

---

## 2. Prérequis

- Python 3.10+ (testé avec 3.15)
- **Tesseract OCR** : installez le binaire `tesseract.exe`
  (ex: [UB-Mannheim builds](https://github.com/UB-Mannheim/tesseract/releases),
  installez dans `C:\Program Files\Tesseract-OCR`). Le module
  `sentiment_pipeline.configure_external_tools_path()` l'ajoute à `PATH`
  s'il est dans `C:\Program Files\Tesseract-OCR`. Le dictionnaire français
  (`fra.traineddata`) est recherché dans `tessdata` (déjà présent).
- **ffmpeg** (requis par openai-whisper pour décoder les fichiers audio) :
  un binaire statique est fourni dans `bin/` (copié de `imageio-ffmpeg`,
  p.ex. `C:\ISETN\5eme semestre\ML\mini projet\bin\ffmpeg.exe`).

---

## 3. Installation

```bash
pip install -r requirements.txt
```

> `openai-whisper` inclut `torch` (gros paquet). Si vous voulez une
> alternative plus légère à la place d'un ASR, modifiez
> `asr_utils.py` (p.ex. `faster-whisper` ou `speechrecognition`).

## 4. Entraînement du modèle (Étape 1)

```bash
python train_model.py
```

À la fin : `models/vectorizer.pkl`, `models/model.pkl` et
`models/metrics.json` sont créés. Dataset utilisé :
`dataset_sentiment_ecommerce/textes.csv` (24 exemples, classes
positif/negatif/neutre, extrait du zip fourni).

## 5. Lancement de l'API (Étape 3)

```bash
python -m uvicorn main:app --reload
# ou
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Documentation interactive : http://127.0.0.1:8000/docs (Swagger UI).

## 6. Mini-app web de test (Étape 4 - Option B)

```bash
streamlit run webapp/streamlit_app.py --server.port 8501
```

Puis ouvrez http://localhost:8501 et testez le sentiment à partir d'un
texte, d'une image ou d'un fichier audio. L'app interroge l'API en
arrière-plan et affiche sentiment + barre de confiance.

## 7. Exemples d'appels API

```bash
# Texte
curl -X POST http://127.0.0.1:8000/predict/text \
     -H "Content-Type: application/json" \
     -d '{"texte": "Le produit est conforme, livraison rapide, je recommande vivement !"}'
```

```json
{
  "source": "text",
  "texte": "Le produit est conforme, livraison rapide, je recommande vivement !",
  "texte_extrait": "produit conforme livraison rapide recommande vivement",
  "sentiment_predit": "positif",
  "confiance": 0.4494,
  "classes_possibles": ["negatif", "neutre", "positif"]
}
```

```bash
# Image (multipart)
curl -X POST http://127.0.0.1:8000/predict/image -F "file=@dataset_sentiment_ecommerce/images/I01.png"

# Audio (multipart)
curl -X POST http://127.0.0.1:8000/predict/audio -F "file=@dataset_sentiment_ecommerce/audios/A01.mp3"

# Détection automatique
curl -X POST http://127.0.0.1:8000/predict -F "file=@dataset_sentiment_ecommerce/images/I01.png"
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"texte": "Livraison en retard, très déçu."}'

# État du service
curl http://127.0.0.1:8000/health
```

## 8. Tests (Étape 5)

```bash
python -m pytest test_api.py -v
```

18 tests : endpoints `/predict/text`, `/predict/image`, `/predict/audio`,
`/predict`, structure des réponses JSON, codes d'erreur attendus (422 pour
charges inadaptées, 413 pour fichiers trop volumineux, code des routes
`/health` et `/`).

## 9. Artefacts et repository

```
mini projet/
├── sentiment_pipeline.py        # Classifieur texte partagé (charge/prédit)
├── train_model.py               # Étape 1 : entraînement + évaluation
├── ocr_utils.py                 # Étape 2 : OCR image
├── asr_utils.py                 # Étape 2 : transcription audio
├── main.py                      # Étape 3 : API FastAPI
├── webapp/streamlit_app.py      # Étape 4 : mini-app web de test
├── test_api.py                  # Étape 5 : tests pytest
├── requirements.txt
├── models/                      # vectorizer.pkl, model.pkl, metrics.json
├── bin/ffmpeg.exe               # Binaire ffmpeg statique (imageio-ffmpeg)
├── assets/api_docs.png          # Capture Swagger
├── assets/webapp.png            # Capture de la mini-app Streamlit
└── dataset_sentiment_ecommerce/ # Dataset fourni (textes.csv, images/, audios/)
```

## 10. Captures d'écran

- `assets/api_docs.png` : documentation Swagger interactive (`/docs`).
- `assets/webapp.png` : mini-application web de test en fonctionnement.

## 11. Démonstration fonctionnelle

### Predictions en direct (via l'API)

```bash
GET /health -> {"status": "ok", "classes": ["negatif", "neutre", "positif"]}

POST /predict/text {"texte": "Le produit est conforme, livraison rapide en 24h, je recommande vivement !"}
-> {"source": "text", "sentiment_predit": "positif", "confiance": 0.4494, ...}

POST /predict {"texte": "Livraison en retard, service client injoignable. Très déçu."}
-> {"source": "auto", "sentiment_predit": "negatif", "confiance": 0.4216, ...}

POST /predict/image   (I01.png) -> 200, source: "image"
POST /predict/audio   (A01.mp3) -> 200, source: "audio", sentiment: "positif"
```

---

## 12. Choix techniques et explication des étapes

**Étape 1 — modélisation.** Nettoyage minimal des textes via
`sentiment_pipeline.clean_text` (minuscules, ponctuation supprimée,
stopwords français courants retirés) — le même nettoyage est appliqué à
l'entraînement et à la prédiction pour rester cohérent. Vectorisation
avec `TfidfVectorizer`, entraînement d'un `LogisticRegression`
(classes `negatif`, `neutre`, `positif`), évaluation (matrice de confusion,
rapport de classification) et sauvegarde de `model.pkl` /
`vectorizer.pkl` via `joblib` (format pickle standard).

**Étape 2 — prétraitement multimodal.**
- Images : `PIL.Image.open` + `pytesseract.image_to_string` (OCR) → texte
  brut → modèle texte.
- Audio : `whisper.load_model("base")` + `model.transcribe` → texte brut →
  modèle texte. Le binaire `ffmpeg` (déposé dans `bin/` et ajouté à `PATH`
  par `sentiment_pipeline.configure_external_tools_path()`) est requis par
  Whisper pour le décodage.

**Étape 3 — API.** `FastAPI` avec :
- schémas Pydantic pour validation stricte (`TextRequest.texte` : non vide,
  max 10000 caractères ; `File(...)` pour les uploads),
- routes REST modulaires (`/predict/text`, `/predict/image`,
  `/predict/audio`, `/predict`),
- validation des types MIME et tailles (422 pour types non supportés,
  413 pour fichiers > 5 Mo images / 20 Mo audio),
- réponse JSON unifiée : `source`, `texte`, `texte_extrait`,
  `sentiment_predit`, `confiance` (probabilité de la classe prédite),
  `classes_possibles`,
- `GET /health` pour le monitoring du modèle,
- utilisation de `python -m uvicorn main:app --reload`.

**Étape 4 — mini-app web de test.** Streamlit (Option B), une page
interactive : choix de la source (Texte / Image / Audio), champs de
saisie ou téléversement de fichier, appel en arrière-plan à l'API FastAPI,
affichage du sentiment (émoji + barre de confiance) et du texte extrait.

**Étape 5 — tests.** `test_api.py` avec `fastapi.testclient.TestClient`
(httpx) : tests unitaires et d'intégration des routes, structure JSON
attendue, et codes d'erreur (422/413) pour charges inadaptées.

---

## 13. Notes pédagogiques

- Le modèle est entraîné sur le petit jeu `textes.csv` fourni : la
  précision est donc limitée, et les prédictions sur texte OCR peuvent être
  imprécises (la qualité de l'OCR dépend aussi de la qualité de la
  capture). C'est un point d'attention important : avec plus de données
  ou un modèle plus puissant (transformers), la précision augmenterait.
- La cascade OCR/ASR → texte → classifieur : même logique de prédiction
  finale, sources hétérogènes — point central du mini-projet.
- Pour les étudiants : relancez `python train_model.py` après avoir
  modifié `textes.csv`, les artefacts de `models/` seront réécrits.
