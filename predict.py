import argparse
import os
from predictor import CPVPredictor

def main():
    parser = argparse.ArgumentParser(description='Predict CPV code from text using Hybrid approach.')
    parser.add_argument('text', type=str, help='Text to classify (e.g., tender name)')
    parser.add_argument('--model_dir', type=str, default='models/cpv-decoder', help='Path to trained model directory')
    parser.add_argument('--cpv_codes', type=str, default='cpv_codes.csv', help='Path to CPV codes CSV')
    args = parser.parse_args()

    # Initialize Predictor
    predictor = CPVPredictor(
        model_dir=args.model_dir, 
        cpv_codes_path=args.cpv_codes,
        embeddings_path='cpv_embeddings.pt'
    )
    
    print(f"\n--- Analysis for: '{args.text}' ---\n")
    
    results = predictor.predict(args.text)

    # --- 1. Keyword Match ---
    if results['keyword_matches']:
        print(f"✅ KEYWORD MATCHES (Top {len(results['keyword_matches'])}):")
        for i, res in enumerate(results['keyword_matches'], 1):
            print(f"   {i}. {res['cpv_code']} - {res['name']}")
    else:
        print("❌ No text match found.")

    # --- 2. Semantic RAG ---
    print(f"\n🔍 SEMANTIC RAG (Cosine Similarity):")
    if results['rag_matches']:
        for i, res in enumerate(results['rag_matches'], 1):
            print(f"   {i}. [{res['score']:.4f}] {res['cpv_code']} - {res['name']}")
    else:
        print("   No relevant matches found.")

    # --- 3. Model Prediction ---
    print(f"\n🧠 AI MODEL:")
    if results['model_predictions']:
        for i, res in enumerate(results['model_predictions'], 1):
            print(f"   {i}. [{res['probability']:.4f}] {res['cpv_code']} - {res['name']}")
    else:
        print("   Model not loaded or errored.")

    # --- 4. FINAL DECISION (Aggregated) ---
    print(f"\n🎯 FINAL DECISION (Aggregated):")
    if results.get('final_decision'):
        for i, res in enumerate(results['final_decision'], 1):
            sources_str = ', '.join(res['sources'])
            print(f"   {i}. [{res['confidence']:.4f}] {res['cpv_code']} - {res['name']}")
            print(f"      Sources: {sources_str}")
    else:
        print("   No final decision available.")
    
    print()  # Add blank line at end

if __name__ == "__main__":
    main()
