import pandas as pd
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load the base splits
train_base = pd.read_csv("data/train.csv")
val_base = pd.read_csv("data/val.csv")
test_base = pd.read_csv("data/test.csv")

# Load vfnd splits
vfnd_train = pd.read_csv("data/vfnd_train.csv")
vfnd_val = pd.read_csv("data/vfnd_val.csv")
vfnd_test = pd.read_csv("data/vfnd_test.csv")

# Load tingia splits
tingia_train = pd.read_csv("data/tingia_train.csv")
tingia_val = pd.read_csv("data/tingia_val.csv")
tingia_test = pd.read_csv("data/tingia_test.csv")

def clean(text):
    return str(text).lower().strip().replace(" ", "")

# Create sets
sets = {
    "train_base": set(train_base["post_message"].apply(clean)),
    "val_base": set(val_base["post_message"].apply(clean)),
    "test_base": set(test_base["post_message"].apply(clean)),
    "vfnd_train": set(vfnd_train["post_message"].apply(clean)),
    "vfnd_val": set(vfnd_val["post_message"].apply(clean)),
    "vfnd_test": set(vfnd_test["post_message"].apply(clean)),
    "tingia_train": set(tingia_train["post_message"].apply(clean)),
    "tingia_val": set(tingia_val["post_message"].apply(clean)),
    "tingia_test": set(tingia_test["post_message"].apply(clean)),
}

print("=== CHECKING INTERNAL SPLIT OVERLAPS ===")
# Check if train_base overlaps with val_base/test_base
print("train_base & val_base:", len(sets["train_base"].intersection(sets["val_base"])))
print("train_base & test_base:", len(sets["train_base"].intersection(sets["test_base"])))

# Check if vfnd splits overlap internally
print("vfnd_train & vfnd_val:", len(sets["vfnd_train"].intersection(sets["vfnd_val"])))
print("vfnd_train & vfnd_test:", len(sets["vfnd_train"].intersection(sets["vfnd_test"])))

# Check if tingia splits overlap internally
print("tingia_train & tingia_val:", len(sets["tingia_train"].intersection(sets["tingia_val"])))
print("tingia_train & tingia_test:", len(sets["tingia_train"].intersection(sets["tingia_test"])))

print("\n=== CHECKING CROSS-DATASET LEAKAGE ===")
# Does vfnd_val or vfnd_test overlap with train_base?
print("vfnd_val & train_base:", len(sets["vfnd_val"].intersection(sets["train_base"])))
print("vfnd_test & train_base:", len(sets["vfnd_test"].intersection(sets["train_base"])))

# Does tingia_val or tingia_test overlap with train_base?
print("tingia_val & train_base:", len(sets["tingia_val"].intersection(sets["train_base"])))
print("tingia_test & train_base:", len(sets["tingia_test"].intersection(sets["train_base"])))

# Does vfnd_train overlap with val_base or test_base?
print("vfnd_train & val_base:", len(sets["vfnd_train"].intersection(sets["val_base"])))
print("vfnd_train & test_base:", len(sets["vfnd_train"].intersection(sets["test_base"])))

# Does tingia_train overlap with val_base or test_base?
print("tingia_train & val_base:", len(sets["tingia_train"].intersection(sets["val_base"])))
print("tingia_train & test_base:", len(sets["tingia_train"].intersection(sets["test_base"])))
