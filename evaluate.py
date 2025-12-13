import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import torch
from predictor import CPVPredictor

def evaluate():
    print("Loading data...")
    try:
        df = pd.read_csv('training_data.csv')
    except FileNotFoundError:
        print("Error: 'training_data.csv' not found.")
        return

    # cleanup as in train.py
    if 'input' in df.columns and 'output' in df.columns:
        df = df.rename(columns={'input': 'text', 'output': 'label'})
    
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(str)
    
    # Recreate the split
    print("Recreating validation split (seed=42)...")
    _, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=None)
    
    print(f"Validation set size: {len(val_df)}")
    
    if len(val_df) > 2000:
        print("Dataset too large, sampling 2000 items for faster evaluation...")
        val_df = val_df.sample(2000, random_state=42)
    
    print("Initializing Predictor...")
    predictor = CPVPredictor()
    
    y_true = []
    y_pred_top1 = []
    y_pred_top3 = [] # List of lists
    
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    
    print("Running predictions...")
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        text = row['text']
        true_label = row['label']
        
        # Predict
        # We rely on the combined logic of keyword -> RAG -> Model
        # But `predictor.predict` returns a structured dict. 
        # We need to decide our "final answer" logic.
        # Logic: 
        # 1. If keyword matches exist, pick the first one.
        # 2. Else if model predictions exist, pick the highest prob.
        # 3. Else pick RAG.
        
        results = predictor.predict(text, top_k=3)
        
        candidates = []
        
        # Priority 1: Keyword Matches
        if results['keyword_matches']:
            for m in results['keyword_matches']:
                candidates.append(m['cpv_code'])
        
        # Priority 2: Model Predictions (append if space in candidates)
        if results['model_predictions']:
             for m in results['model_predictions']:
                if m['cpv_code'] not in candidates:
                    candidates.append(m['cpv_code'])
        
        # Priority 3: RAG (append if space)
        if results['rag_matches']:
            for m in results['rag_matches']:
                if m['cpv_code'] not in candidates:
                    candidates.append(m['cpv_code'])
                    
        # Truncate to top 3
        candidates = candidates[:3]
        
        # Fill if empty (shouldn't happen with trained model)
        if not candidates:
            candidates = ["00000000-0"] * 3
            
        # Pad if less than 3
        while len(candidates) < 3:
            candidates.append("None")

        top1 = candidates[0]
        
        y_true.append(true_label)
        y_pred_top1.append(top1)
        y_pred_top3.append(candidates)
        
        if top1 == true_label:
            correct_top1 += 1
        
        if true_label in candidates:
            correct_top3 += 1
            
        total += 1
        
    print("\n--- Evaluation Results ---")
    print(f"Total Samples: {total}")
    print(f"Top-1 Accuracy: {correct_top1/total:.4f} ({correct_top1}/{total})")
    print(f"Top-3 Accuracy: {correct_top3/total:.4f} ({correct_top3}/{total})")
    
    # Detailed classification report for Top-1 (limited to top classes to avoid spam)
    # print("\nClassification Report (Top 1):")
    # print(classification_report(y_true, y_pred_top1, zero_division=0))

if __name__ == "__main__":
    evaluate()
