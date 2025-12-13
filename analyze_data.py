import pandas as pd
import numpy as np

try:
    df = pd.read_csv('training_data.csv')
    if 'input' in df.columns and 'output' in df.columns:
        df = df.rename(columns={'input': 'text', 'output': 'label'})
    
    print(f"Total rows: {len(df)}")
    print(f"Unique labels (CPV codes): {df['label'].nunique()}")
    
    # Label distribution
    counts = df['label'].value_counts()
    print(f"\nTop 5 frequent labels:\n{counts.head(5)}")
    print(f"\nBottom 5 frequent labels:\n{counts.tail(5)}")
    
    print(f"\nLabels with < 5 samples: {len(counts[counts < 5])}")
    print(f"Labels with 1 sample: {len(counts[counts == 1])}")
    
    # Text length stats (approx words)
    df['word_count'] = df['text'].astype(str).str.split().str.len()
    print(f"\nAvg word count: {df['word_count'].mean():.2f}")
    print(f"Max word count: {df['word_count'].max()}")
    print(f"95th percentile word count: {np.percentile(df['word_count'], 95):.2f}")

except Exception as e:
    print(f"Error: {e}")
