import pandas as pd
import os
import subprocess
import io
from sklearn.model_selection import train_test_split

def clean(text):
    return str(text).lower().strip().replace(" ", "")

# 1. Load original train, val, test (ReINTEL)
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

reintel_texts = set()
for df in [train, val, test]:
    reintel_texts.update(df['post_message'].fillna("").apply(clean))

print(f"Total ReINTEL unique texts: {len(reintel_texts)}")

# 2. Get the clean VFND original (227 rows) from commit 115f355
out_vfnd = subprocess.check_output(['git', 'show', '115f355:data/additional_train.csv']).decode('utf-8')
vfnd_df = pd.read_csv(io.StringIO(out_vfnd))
print(f"VFND original size: {len(vfnd_df)}")

# 3. Load the 4,067-row version which was the combined file from commit c5cfaf7
out_combined = subprocess.check_output(['git', 'show', 'c5cfaf7:data/additional_train.csv']).decode('utf-8')
combined_df = pd.read_csv(io.StringIO(out_combined))
print(f"Combined additional dataset (c5cfaf7) size: {len(combined_df)}")

# 4. Extract "My crawled data": rows in combined_df that are NOT in ReINTEL and NOT in VFND
vfnd_texts = set(vfnd_df['post_message'].fillna("").apply(clean))
my_crawled_rows = []
for idx, row in combined_df.iterrows():
    cleaned_txt = clean(row['post_message'])
    # If it is in ReINTEL or VFND, skip it
    if cleaned_txt in reintel_texts or cleaned_txt in vfnd_texts:
        continue
    my_crawled_rows.append(row)

my_crawled_df = pd.DataFrame(my_crawled_rows)
print(f"Extracted 'My crawled data' size: {len(my_crawled_df)}")

# Define helper to split a dataframe into 80/10/10 with stratification
def split_80_10_10(df, prefix):
    # Split 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['label']
    )
    # Split temp into 50% val, 50% test (each is 10% of total)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
    )
    
    # Save to data/
    train_path = f"data/{prefix}_train.csv"
    val_path = f"data/{prefix}_val.csv"
    test_path = f"data/{prefix}_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Split {prefix} successfully:")
    print(f"  -> Train: {len(train_df)} rows")
    print(f"  -> Val:   {len(val_df)} rows")
    print(f"  -> Test:  {len(test_df)} rows")

# Split VFND
split_80_10_10(vfnd_df, "vfnd")

# Split Custom Crawled
split_80_10_10(my_crawled_df, "crawled")
