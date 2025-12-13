import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

class CPVPredictor:
    def __init__(self, model_dir='models/cpv-decoder', cpv_codes_path='cpv_codes.csv', embeddings_path='cpv_embeddings.pt'):
        self.model_dir = model_dir
        self.cpv_codes_path = cpv_codes_path
        self.embeddings_path = embeddings_path
        
        self.tokenizer = None
        self.model = None # Classifier model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.cpv_descriptions = {}
        self.name_to_code = {}
        self.df_codes = None
        
        self.cpv_embeddings = None
        self.embedding_codes = []
        
        self._load_resources()

    def _load_resources(self):
        import sys
        # 1. Load CPV Codes
        if os.path.exists(self.cpv_codes_path):
            self.df_codes = pd.read_csv(self.cpv_codes_path, dtype={'cpv_code': str})
            if 'cpv_code' in self.df_codes.columns and 'name' in self.df_codes.columns:
                for _, row in self.df_codes.iterrows():
                    code = str(row['cpv_code']).strip()
                    name = str(row['name']).strip()
                    self.cpv_descriptions[code] = name
                    self.name_to_code[name.lower()] = code
        else:
             print(f"Warning: CPV codes file not found at {self.cpv_codes_path}", file=sys.stderr)

        # 2. Load Model & Tokenizer
        if os.path.exists(self.model_dir):
            print(f"Loading model from {self.model_dir}...", file=sys.stderr)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                self.model.to(self.device)
                self.model.eval()
                print("Model loaded successfully.", file=sys.stderr)
            except Exception as e:
                print(f"ERROR LOADING MODEL: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                raise e # Propagate error
        else:
             print(f"Warning: Model dir not found at {self.model_dir}", file=sys.stderr)
        
        # 3. Load Embeddings
        if os.path.exists(self.embeddings_path):
            print(f"Loading embeddings from {self.embeddings_path}...", file=sys.stderr)
            data = torch.load(self.embeddings_path, map_location=self.device)
            self.cpv_embeddings = data['embeddings']
            self.embedding_codes = data['cpv_codes']
            print("Embeddings loaded successfully.", file=sys.stderr)
        else:
             print(f"Warning: Embeddings file not found at {self.embeddings_path}", file=sys.stderr)

    def compute_final_decision(self, results, top_k=3):
        """
        Aggregate results from all three methods using weighted scoring.
        
        Weights:
        - Keyword Match: 0.4 (high confidence when exact match found)
        - Semantic RAG: 0.35 (medium-high confidence)
        - AI Model: 0.25 (medium confidence)
        
        Returns list of dicts with cpv_code, name, confidence, and sources
        """
        cpv_scores = {}  # {cpv_code: {'score': float, 'sources': set, 'name': str}}
        
        # Weight configuration
        KEYWORD_WEIGHT = 0.4
        RAG_WEIGHT = 0.35
        MODEL_WEIGHT = 0.25
        
        # 1. Process Keyword Matches (score = 1.0 for matches)
        for i, match in enumerate(results['keyword_matches']):
            code = match['cpv_code']
            # Give higher score to earlier matches (more relevant)
            position_score = 1.0 - (i * 0.1)  # 1.0, 0.9, 0.8
            score = position_score * KEYWORD_WEIGHT
            
            if code not in cpv_scores:
                cpv_scores[code] = {'score': 0.0, 'sources': set(), 'name': match['name']}
            cpv_scores[code]['score'] += score
            cpv_scores[code]['sources'].add('keyword')
        
        # 2. Process RAG Matches (score = cosine similarity)
        for match in results['rag_matches']:
            code = match['cpv_code']
            score = match['score'] * RAG_WEIGHT
            
            if code not in cpv_scores:
                cpv_scores[code] = {'score': 0.0, 'sources': set(), 'name': match['name']}
            cpv_scores[code]['score'] += score
            cpv_scores[code]['sources'].add('rag')
        
        # 3. Process Model Predictions (score = probability)
        for match in results['model_predictions']:
            code = match['cpv_code']
            score = match['probability'] * MODEL_WEIGHT
            
            if code not in cpv_scores:
                cpv_scores[code] = {'score': 0.0, 'sources': set(), 'name': match['name']}
            cpv_scores[code]['score'] += score
            cpv_scores[code]['sources'].add('model')
        
        # Sort by score and get top_k
        sorted_results = sorted(cpv_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        final_decision = []
        for code, data in sorted_results[:top_k]:
            final_decision.append({
                'cpv_code': code,
                'name': data['name'],
                'confidence': data['score'],
                'sources': list(data['sources'])
            })
        
        return final_decision

    def predict(self, text, top_k=3):
        results = {
            'keyword_matches': [],
            'rag_matches': [],
            'model_predictions': []
        }
        
        text_clean = text.strip()
        
        # --- 1. Keyword Match (Partial Exact) ---
        text_lower = text_clean.lower()
        keyword_matches = []
        
        def find_matches(search_term):
            matches = []
            for code, name in self.cpv_descriptions.items():
                if search_term in name.lower():
                    matches.append({
                        'cpv_code': code,
                        'name': name
                    })
            return matches

        # 1. Try exact substring
        keyword_matches = find_matches(text_lower)
        
        # 2. Stemming fallback
        if not keyword_matches and len(text_lower) > 4:
            stemmed = text_lower[:-1]
            keyword_matches = find_matches(stemmed)
        
        keyword_matches.sort(key=lambda x: len(x['name']))
        results['keyword_matches'] = keyword_matches[:top_k]

        # --- Combined Model & RAG Prediction ---
        # Optimize: Tokenize once, run model once with output_hidden_states=True
        if self.model:
            with torch.no_grad():
                inputs = self.tokenizer(text_clean, return_tensors='pt', truncation=True, max_length=128).to(self.device)
                
                # We need hidden states for RAG (embeddings) and logits for classification
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # A. Model Prediction
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                top_vals, top_idxs = torch.topk(probs, top_k)
                
                for val, idx in zip(top_vals[0], top_idxs[0]):
                    code = self.model.config.id2label[idx.item()]
                    results['model_predictions'].append({
                        'probability': val.item(),
                        'cpv_code': code,
                        'name': self.cpv_descriptions.get(code, "Unknown")
                    })
                
                # B. Semantic RAG (using hidden states)
                if self.cpv_embeddings is not None:
                    # 'hidden_states' is a tuple of (embeddings, layer_1, ..., layer_n)
                    # We want the last layer's CLS token (index 0)
                    last_hidden_state = outputs.hidden_states[-1]
                    query_embedding = last_hidden_state[:, 0, :] # CLS token
                    
                    # Cosine Similarity
                    from torch.nn.functional import cosine_similarity
                    sims = cosine_similarity(query_embedding, self.cpv_embeddings)
                    
                    top_vals_rag, top_idxs_rag = torch.topk(sims, top_k)
                    
                    for val, idx in zip(top_vals_rag, top_idxs_rag):
                        code = self.embedding_codes[idx.item()]
                        results['rag_matches'].append({
                            'score': val.item(),
                            'cpv_code': code,
                            'name': self.cpv_descriptions.get(code, "Unknown")
                        })
        
        # Compute final decision by aggregating all methods
        results['final_decision'] = self.compute_final_decision(results, top_k)
                    
        return results
