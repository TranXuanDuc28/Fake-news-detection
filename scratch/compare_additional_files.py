import sys
import pandas as pd
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

files = [
    "data/not_in_use/additional_train.csv",
    "data/not_in_use/massive_additional_train.csv",
    "data/not_in_use/additional_train_old.csv"
]

for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f"=== {f} ===")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Label distribution:\n", df['label'].value_counts() if 'label' in df.columns else 'No label column')
        print("First 2 rows:")
        print(df.head(2))
        print("-" * 50)
