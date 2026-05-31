import pandas as pd
import os

filepath = "data/not_in_use/augmented_dataset.csv"
if os.path.exists(filepath):
    df = pd.read_csv(filepath)
    print("Shape:", df.shape)
    print("Label counts:\n", df['label'].value_counts() if 'label' in df.columns else 'No label column')
    print("Unique messages:", df['post_message'].nunique() if 'post_message' in df.columns else 'No message column')
else:
    print("Not found")
