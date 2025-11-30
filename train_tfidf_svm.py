import json
import glob
from pathlib import Path
from collections import Counter

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


def build_pipeline():
    return Pipeline(
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


def main():
    print("Loading data...")
    X_all, y_all = load_procurements()
    print(f"Loaded {len(X_all)} samples, {len(set(y_all))} unique CPV codes")

    # --- priprema za evaluaciju (izbaci klase sa < 2 primera) ---
    counts = Counter(y_all)
    keep_mask = [counts[label] >= 2 for label in y_all]

    X_eval = [x for x, keep in zip(X_all, keep_mask) if keep]
    y_eval = [y for y, keep in zip(y_all, keep_mask) if keep]

    print(
        f"For evaluation: keeping {len(X_eval)} samples "
        f"across {len(set(y_eval))} classes (min 2 samples per class)."
    )

    if len(set(y_eval)) > 1 and len(y_eval) > 0:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_eval,
                y_eval,
                test_size=0.1,
                random_state=42,
                stratify=y_eval,
            )

            clf_eval = build_pipeline()

            print("Training model for evaluation...")
            clf_eval.fit(X_train, y_train)

            print("Evaluating on held-out set...")
            y_pred = clf_eval.predict(X_test)
            print(classification_report(y_test, y_pred, digits=3))

        except ValueError as e:
            # fallback ako i dalje nešto pukne
            print("Stratified split failed, falling back to non-stratified split.")
            print(f"Reason: {e}")

            X_train, X_test, y_train, y_test = train_test_split(
                X_eval,
                y_eval,
                test_size=0.1,
                random_state=42,
                stratify=None,
            )

            clf_eval = build_pipeline()
            print("Training model for evaluation (non-stratified)...")
            clf_eval.fit(X_train, y_train)

            print("Evaluating on held-out set (non-stratified)...")
            y_pred = clf_eval.predict(X_test)
            print(classification_report(y_test, y_pred, digits=3))
    else:
        print("Not enough data/classes for a meaningful evaluation split.")

    # --- finalni model na SVIM primerima (uključujući retke klase) ---
    print("Training final model on ALL data...")
    clf_final = build_pipeline()
    clf_final.fit(X_all, y_all)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_final, MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
