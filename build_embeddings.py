import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for CPV codes.')
    parser.add_argument('--model_dir', type=str, default='models/cpv-decoder', help='Path to trained model directory')
    parser.add_argument('--cpv_codes', type=str, default='cpv_codes.csv', help='Path to CPV codes CSV')
    parser.add_argument('--output_file', type=str, default='cpv_embeddings.pt', help='Output file for embeddings')
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory {args.model_dir} does not exist.")
        return

    if not os.path.exists(args.cpv_codes):
        print(f"Error: CPV codes file {args.cpv_codes} does not exist.")
        return

    print(f"Loading model from {args.model_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        # Load as AutoModel (without classification head) to get raw embeddings
        model = AutoModel.from_pretrained(args.model_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading CPV codes...")
    try:
        df = pd.read_csv(args.cpv_codes, dtype={'cpv_code': str})
        if 'name' not in df.columns:
            print("Error: 'name' column missing in CPV codes CSV.")
            return
    except Exception as e:
        print(f"Error reading CPV codes: {e}")
        return

    print(f"Generating embeddings for {len(df)} CPV codes...")
    
    cpv_codes = []
    embeddings_list = []
    
    batch_size = 32
    texts = df['name'].astype(str).tolist()
    codes = df['cpv_code'].astype(str).tolist()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_codes = codes[i:i+batch_size]
            
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            outputs = model(**inputs)
            
            # Use CLS token embedding (first token) as sentence representation
            # shape: (batch_size, hidden_dim)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            
            embeddings_list.append(batch_embeddings)
            cpv_codes.extend(batch_codes)

    all_embeddings = torch.cat(embeddings_list, dim=0)
    
    print(f"Saving embeddings to {args.output_file}...")
    torch.save({
        'cpv_codes': cpv_codes,
        'embeddings': all_embeddings
    }, args.output_file)
    
    print("Done!")

if __name__ == "__main__":
    main()
