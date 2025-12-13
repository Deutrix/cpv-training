import pandas as pd

try:
    print("Reading training data...")
    # 'output' matches the column name in prepare_training_data.py
    df_train = pd.read_csv('training_data.csv', dtype={'output': str, 'label': str})
    
    # Handle potential column renaming if the file was modified or original header used
    label_col = 'output' if 'output' in df_train.columns else 'label'
    
    unique_train = set(df_train[label_col].dropna().unique())
    print(f"Unique CPV codes in training data: {len(unique_train)}")

    print("Reading CPV codes reference...")
    df_codes = pd.read_csv('cpv_codes.csv', dtype={'cpv_code': str})
    unique_codes = set(df_codes['cpv_code'].dropna().unique())
    print(f"Total CPV codes in listing: {len(unique_codes)}")

    missing = unique_codes - unique_train
    print(f"Codes missing from training data: {len(missing)}")
    
    if len(missing) > 0:
        print(f"Example missing codes: {list(missing)[:5]}")

except Exception as e:
    print(f"Error: {e}")
