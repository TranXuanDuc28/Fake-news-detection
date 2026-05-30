import pandas as pd
import os

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

# Clean text a bit to check for semantic duplicates (ignoring spaces/casing)
def clean(text):
    return str(text).lower().strip().replace(" ", "")

train_cleaned = train['post_message'].fillna("").apply(clean)
val_cleaned = val['post_message'].fillna("").apply(clean)
test_cleaned = test['post_message'].fillna("").apply(clean)

train_set = set(train_cleaned)
val_set = set(val_cleaned)
test_set = set(test_cleaned)

print(f"Total train: {len(train)}")
print(f"Total val: {len(val)}")
print(f"Total test: {len(test)}")

print("\n--- LEAKAGE CHECK ---")
val_in_train = sum(1 for x in val_cleaned if x in train_set)
test_in_train = sum(1 for x in test_cleaned if x in train_set)
test_in_val = sum(1 for x in test_cleaned if x in val_set)

print(f"Validation samples leaked in Train: {val_in_train} ({val_in_train/len(val)*100:.2f}%)")
print(f"Test samples leaked in Train: {test_in_train} ({test_in_train/len(test)*100:.2f}%)")
print(f"Test samples leaked in Validation: {test_in_val} ({test_in_val/len(test)*100:.2f}%)")

# Check for duplicates within train itself
train_dups = len(train) - len(train_cleaned.unique())
print(f"Duplicates within Train: {train_dups}")
