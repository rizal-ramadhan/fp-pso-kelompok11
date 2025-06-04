import pandas as pd
import os

def validate_dataset():
    data_files = ['data/mental_health_lite.csv', 'data/mental_health_life_cut.csv']
    df = None
    
    for file_path in data_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f'✅ Dataset loaded from: {file_path}')
            break
    
    if df is None:
        raise FileNotFoundError('No dataset found')
    
    assert len(df) > 0, 'Dataset is empty'
    print(f'✅ Dataset validation passed - Shape: {df.shape}')
    print(f'📊 Columns: {list(df.columns)}')
    if 'mental_health_risk' in df.columns:
        print(f'📊 Target distribution: {df["mental_health_risk"].value_counts().to_dict()}')

if __name__ == "__main__":
    validate_dataset()
