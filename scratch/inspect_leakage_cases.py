import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_raw_data

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

train_df, val_df, test_df = load_raw_data("data", segment_words=False, additional_dataset="both")

def clean(text):
    return str(text).lower().strip().replace(" ", "")

train_cleaned_to_orig = {}
for idx, row in train_df.iterrows():
    c = clean(row["post_message"])
    if c not in train_cleaned_to_orig:
        train_cleaned_to_orig[c] = []
    train_cleaned_to_orig[c].append(row["post_message"])

print("=== LEAKAGE IN VAL ===")
val_count = 0
for idx, row in val_df.iterrows():
    c = clean(row["post_message"])
    if c in train_cleaned_to_orig:
        val_count += 1
        print(f"Val item: {row['post_message'][:150]}...")
        print(f"Train matches ({len(train_cleaned_to_orig[c])}):")
        for match in train_cleaned_to_orig[c][:2]:
            print(f"  - {match[:150]}...")
        print("-" * 50)

print(f"Total val leakage cases: {val_count}")

print("\n=== LEAKAGE IN TEST ===")
test_count = 0
for idx, row in test_df.iterrows():
    c = clean(row["post_message"])
    if c in train_cleaned_to_orig:
        test_count += 1
        print(f"Test item: {row['post_message'][:150]}...")
        print(f"Train matches ({len(train_cleaned_to_orig[c])}):")
        for match in train_cleaned_to_orig[c][:2]:
            print(f"  - {match[:150]}...")
        print("-" * 50)

print(f"Total test leakage cases: {test_count}")
