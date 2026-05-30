import pandas as pd
import os
import sys

# Reconfigure stdout to use UTF-8 to prevent charmap encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_raw_data

train_df, val_df, test_df = load_raw_data("data", segment_words=False, additional_dataset="both")

def clean(text):
    return str(text).lower().strip().replace(" ", "")

train_messages = set(train_df["post_message"].apply(clean))
val_messages = set(val_df["post_message"].apply(clean))
test_messages = set(test_df["post_message"].apply(clean))

val_leakage = val_messages.intersection(train_messages)
test_leakage = test_messages.intersection(train_messages)

print(f"Train size: {len(train_df)}")
print(f"Val size: {len(val_df)}")
print(f"Test size: {len(test_df)}")
print(f"Leakage in Val (present in Train): {len(val_leakage)} ({len(val_leakage)/len(val_df):.2%})")
print(f"Leakage in Test (present in Train): {len(test_leakage)} ({len(test_leakage)/len(test_df):.2%})")
