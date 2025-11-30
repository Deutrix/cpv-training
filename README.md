Da, skroz je ok da bude samo Python + FastAPI – to je čak idealno za Coolify.
Napravimo setup sa:

Model 1 – TF-IDF + LinearSVC (supervised, koristi tvojih 230k nabavki)

Model 2 – SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2) + “najbliži CPV” (semantic search)

FastAPI koji nudi:

/predict/tfidf

/predict/semantic

/predict/ensemble (jednostavan spoj oba)

paraphrase-multilingual-MiniLM-L12-v2 je dobar izbor: mali, brz, radi dobro na srpskom + engleskom.

Ispod imaš praktično kompletan “mini-repo” koji možeš nalepiti u svoj git, trenirati lokalno i onda hostovati preko Coolify-ja.

1. Struktura projekta

Predlog strukture:

cpv-classifier/
  data/
    postupci/              # tvojih 11 JSON fajlova sa nabavkama
      Postupci_1.json
      Postupci_2.json
      ...
    cpv_codes.json         # šifrarnik CPV (ovaj što si uploadovao)
  models/
    (biće generisano posle treniranja)
  train_tfidf_svm.py
  build_cpv_embeddings.py
  main.py                  # FastAPI app
  requirements.txt
  Dockerfile
  README.md (opciono)


JSON iz postupaka: polja Name, NameENG, TypeContract, ProcedureType, CPVExtended itd.
CPV šifrarnik: Number, Name, NameEng, …

2. Treniranje TF-IDF + LinearSVC modela

train_tfidf_svm.py

import json
import glob
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATA_DIR = Path("data/postupci")
MODEL_PATH = Path("models/cpv_tfidf_linearsvc.joblib")


def load_procurements():
    texts = []
    labels = []

    json_files = glob.glob(str(DATA_DIR / "*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {DATA_DIR}")

    for fp in json_files:
        print(f"Loading {fp}")
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data:
            name = (row.get("Name") or "").strip()
            name_eng = (row.get("NameENG") or "").strip()
            type_contract = (row.get("TypeContract") or "").strip()
            procedure_type = (row.get("ProcedureType") or "").strip()

            text_parts = [name, name_eng, type_contract, procedure_type]
            text = " | ".join([t for t in text_parts if t])

            cpv_extended = row.get("CPVExtended") or ""
            if " - " in cpv_extended:
                cpv_code = cpv_extended.split(" - ", 1)[0].strip()
            else:
                continue

            if not text or not cpv_code:
                continue

            texts.append(text)
            labels.append(cpv_code)

    return texts, labels


def main():
    print("Loading data...")
    X, y = load_procurements()
    print(f"Loaded {len(X)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    clf = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            ("svm", LinearSVC()),
        ]
    )

    print("Training model...")
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()


Pokretanje (lokalno):

python train_tfidf_svm.py


U models/ ćeš dobiti:
cpv_tfidf_linearsvc.joblib

3. Građenje CPV embeddinga (transformer varijanta)

Koristimo sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 i šifrarnik cpv_codes.json.

build_cpv_embeddings.py

import json
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

CPV_CODES_PATH = Path("data/cpv_codes.json")
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUT_PATH = Path("models/cpv_semantic_index.joblib")


def load_cpv_codes():
    with open(CPV_CODES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = []
    texts = []

    for row in data:
        number = (row.get("Number") or "").strip()
        name = (row.get("Name") or "").strip()
        name_eng = (row.get("NameEng") or "").strip()

        if not number:
            continue

        # Kombinacija srpskog i engleskog opisa za bolju semantiku
        full_text = f"{number} - {name} | {name_eng}"
        codes.append(number)
        texts.append(full_text)

    return codes, texts


def main():
    print("Loading CPV codes...")
    codes, texts = load_cpv_codes()
    print(f"Loaded {len(codes)} CPV codes")

    print(f"Loading embedding model: {EMB_MODEL_NAME}")
    model = SentenceTransformer(EMB_MODEL_NAME)

    print("Encoding CPV texts...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.array(embeddings, dtype="float32")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": EMB_MODEL_NAME,
            "codes": codes,
            "texts": texts,
            "embeddings": embeddings,
        },
        OUT_PATH,
    )
    print(f"Saved semantic index to {OUT_PATH}")


if __name__ == "__main__":
    main()


Pokretanje:

python build_cpv_embeddings.py


Ovo kreira models/cpv_semantic_index.joblib sa:

listom CPV kodova,

tekstualnim opisima,

embedding matriksom.

4. FastAPI servis (oba pristupa + ensemble)

main.py

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Paths
TFIDF_MODEL_PATH = Path("models/cpv_tfidf_linearsvc.joblib")
CPV_SEMANTIC_INDEX_PATH = Path("models/cpv_semantic_index.joblib")

app = FastAPI(
    title="CPV Classifier API",
    version="1.0.0",
)

# --- Load TF-IDF + LinearSVC model ---
if not TFIDF_MODEL_PATH.exists():
    raise RuntimeError(f"Missing TF-IDF model: {TFIDF_MODEL_PATH}")

tfidf_model = joblib.load(TFIDF_MODEL_PATH)

# --- Load semantic index (CPV embeddings) ---
if not CPV_SEMANTIC_INDEX_PATH.exists():
    raise RuntimeError(f"Missing semantic index: {CPV_SEMANTIC_INDEX_PATH}")

semantic_index = joblib.load(CPV_SEMANTIC_INDEX_PATH)
cpv_codes: List[str] = semantic_index["codes"]
cpv_texts: List[str] = semantic_index["texts"]
cpv_embeddings: np.ndarray = semantic_index["embeddings"]
emb_model_name: str = semantic_index["model_name"]

# Load embedding model (same as used for index)
embedder = SentenceTransformer(emb_model_name)


class PredictRequest(BaseModel):
    text: str
    top_k: Optional[int] = 5


class SemanticCandidate(BaseModel):
    cpv: str
    label_text: str
    score: float


class TfidfResponse(BaseModel):
    cpv: str


class SemanticResponse(BaseModel):
    best: SemanticCandidate
    top_k: List[SemanticCandidate]


class EnsembleResponse(BaseModel):
    tfidf_cpv: str
    semantic_best: SemanticCandidate
    final_cpv: str
    note: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/tfidf", response_model=TfidfResponse)
def predict_tfidf(payload: PredictRequest):
    text = payload.text.strip()
    cpv_code = tfidf_model.predict([text])[0]
    return TfidfResponse(cpv=cpv_code)


def _semantic_search(text: str, top_k: int = 5) -> List[SemanticCandidate]:
    if top_k <= 0:
        top_k = 5

    emb = embedder.encode(
        [text],
        normalize_embeddings=True,
    )[0]

    scores = np.dot(cpv_embeddings, emb)  # cosine similarity (normalized)
    top_idx = np.argsort(-scores)[:top_k]

    candidates: List[SemanticCandidate] = []
    for idx in top_idx:
        candidates.append(
            SemanticCandidate(
                cpv=cpv_codes[int(idx)],
                label_text=cpv_texts[int(idx)],
                score=float(scores[int(idx)]),
            )
        )
    return candidates


@app.post("/predict/semantic", response_model=SemanticResponse)
def predict_semantic(payload: PredictRequest):
    text = payload.text.strip()
    candidates = _semantic_search(text, top_k=payload.top_k or 5)
    best = candidates[0]
    return SemanticResponse(best=best, top_k=candidates)


@app.post("/predict/ensemble", response_model=EnsembleResponse)
def predict_ensemble(payload: PredictRequest):
    text = payload.text.strip()

    # 1) TF-IDF
    tfidf_cpv = tfidf_model.predict([text])[0]

    # 2) Semantic
    candidates = _semantic_search(text, top_k=payload.top_k or 5)
    best = candidates[0]

    # Jednostavno pravilo spajanja:
    # - ako se TF-IDF i semantic poklapaju -> uzmi taj CPV
    # - ako se ne poklapaju:
    #     - ako je semantic score jako visok (npr. > 0.6), daj semantic,
    #       inače TF-IDF, ali vrati oba za referencu
    if tfidf_cpv == best.cpv:
        final_cpv = tfidf_cpv
        note = "TF-IDF and semantic agree."
    else:
        if best.score >= 0.60:
            final_cpv = best.cpv
            note = "Semantic overriden TF-IDF (high similarity)."
        else:
            final_cpv = tfidf_cpv
            note = "TF-IDF used; semantic only as suggestion."

    return EnsembleResponse(
        tfidf_cpv=tfidf_cpv,
        semantic_best=best,
        final_cpv=final_cpv,
        note=note,
    )


Primena:

TF-IDF samo:

curl -X POST http://localhost:8000/predict/tfidf \
  -H "Content-Type: application/json" \
  -d '{"text":"Medicinski kiseonik u bocama"}'


Semantic samo:

curl -X POST http://localhost:8000/predict/semantic \
  -H "Content-Type: application/json" \
  -d '{"text":"Medicinski kiseonik u bocama","top_k":5}'


Ensemble:

curl -X POST http://localhost:8000/predict/ensemble \
  -H "Content-Type: application/json" \
  -d '{"text":"Medicinski kiseonik u bocama"}'

5. requirements.txt

requirements.txt

fastapi
uvicorn[standard]
scikit-learn
joblib
numpy
sentence-transformers
pydantic


(Instalacija: pip install -r requirements.txt)

6. Dockerfile za Coolify

Najjednostavnije je da:

treniranje odradiš lokalno (ili u posebnom jobu),

commituješ models/ u repozitorijum,

a Docker image učitava samo gotove modele.

Dockerfile

FROM python:3.11

# Workdir
WORKDIR /app

# Copy requirements first (bolji cache)
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /app

# Expose port (Coolify će ga očitati)
EXPOSE 8000

# Start FastAPI preko uvicorn-a
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

7. Kako na Coolify (kratko)

Napravi git repo sa gore navedenom strukturom.

Lokalno pokreni:

python train_tfidf_svm.py

python build_cpv_embeddings.py
i commit-uj sadržaj models/ foldera.

U Coolify:

New App → Git Repository (poveži repo),

odaberi branch,

Coolify će videti Dockerfile,

stavi PORT=8000 (ako treba),

deploy.

Servis će nuditi FastAPI na /<tvoj-url>/
(npr. https://cpv.deutrix.com/predict/ensemble).

8. Da li je MiniLM model dobar izbor?

Da:

multilingual – pokriva srpski + engleski (i mešavine),

mali – brz na CPU (bitno za jeftin hosting),

dobar za sentence similarity – baš ono što ti treba za “najbliži CPV”.

Kombinacijom:

TF-IDF + SVC (uči iz tvojih 230k istorijskih nabavki)

MiniLM semantic search (razume opis CPV kodova + tekst fakture)

dobijaš:

robustan supervised model za poznate pattern-e

plus semantic “safety net” za opisno/čudno formulisane stavke.

Ako hoćeš, sledeći korak može biti:

dodavanje logovanja (da snimamo šta korisnici šalju i šta model vraća),

mali admin endpoint “/debug” gde vidiš TF-IDF top N klasa i semantic top N,

ili “batch” endpoint da klasifikuješ više stavki fakture odjednom.


uvicorn main:app --host 0.0.0.0 --port 8000 --reload