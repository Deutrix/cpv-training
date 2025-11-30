import os
# Fix for protobuf compatibility issue with TensorFlow
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paths
TFIDF_MODEL_PATH = Path("models/cpv_tfidf_linearsvc.joblib")
CPV_SEMANTIC_INDEX_PATH = Path("models/cpv_semantic_index.joblib")
BERT_MODEL_DIR = Path("models/bertic_cpv_classifier")

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

# --- Load BERTić CPV classifier ---
if not BERT_MODEL_DIR.exists():
    raise RuntimeError(f"Missing BERT model directory: {BERT_MODEL_DIR}")

bert_tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL_DIR))
bert_model = AutoModelForSequenceClassification.from_pretrained(str(BERT_MODEL_DIR))
label_map = joblib.load(BERT_MODEL_DIR / "label_map.joblib")
bert_id2label = label_map["id2label"]
bert_model.eval()


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


class BertResponse(BaseModel):
    cpv: str
    confidence: float


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


@app.post("/predict/bert", response_model=BertResponse)
def predict_bert(payload: PredictRequest):
    text = payload.text.strip()

    enc = bert_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = bert_model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        conf, pred_id = torch.max(probs, dim=-1)

    cpv_code = bert_id2label[int(pred_id.item())]
    confidence = float(conf.item())

    return BertResponse(cpv=cpv_code, confidence=confidence)
