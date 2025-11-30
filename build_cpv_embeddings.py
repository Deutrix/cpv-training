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
