import sys
import pandas as pd
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

for name, filepath in [("additional_train.csv", "data/additional_train.csv"), ("train.csv", "data/train.csv")]:
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"=== {name} ===")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Label distribution:\n", df['label'].value_counts() if 'label' in df.columns else 'No label col')
        print("Duplicates in post_message:", df.duplicated(subset=['post_message']).sum() if 'post_message' in df.columns else 'No post_message col')
        print()
    else:
        print(f"{name} not found")
