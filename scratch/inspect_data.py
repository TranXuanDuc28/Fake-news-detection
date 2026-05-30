import os
import pandas as pd
import sys

# Set output encoding to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

add_path = "data/additional_train.csv"
if os.path.exists(add_path):
    df = pd.read_csv(add_path)
    print("--- VFND FAKE NEWS (Label 1) Samples ---")
    for i, msg in enumerate(df[df['label'] == 1]['post_message'].head(5), 1):
        print(f"{i}. {msg[:150]}...")
    print("\n--- VFND REAL NEWS (Label 0) Samples ---")
    for i, msg in enumerate(df[df['label'] == 0]['post_message'].head(5), 1):
        print(f"{i}. {msg[:150]}...")
else:
    print("additional_train.csv not found")
