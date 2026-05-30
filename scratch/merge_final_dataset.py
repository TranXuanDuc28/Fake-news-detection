import pandas as pd
import os

try:
    df1 = pd.read_csv("data/additional_train.csv")
    print(f"Loaded additional_train.csv: {len(df1)} rows")
except:
    df1 = pd.DataFrame()
    
try:
    df2 = pd.read_csv("data/augmented_dataset.csv")
    print(f"Loaded augmented_dataset.csv: {len(df2)} rows")
except:
    df2 = pd.DataFrame()

# Merge
final_df = pd.concat([df1, df2], ignore_index=True)
# Drop exact duplicates just in case
final_df = final_df.drop_duplicates(subset=['post_message'])
# Shuffle
final_df = final_df.sample(frac=1).reset_index(drop=True)

# Save it to data/massive_additional_train.csv
final_df.to_csv("data/massive_additional_train.csv", index=False)
print(f"Successfully merged! Final massive dataset size: {len(final_df)} rows.")

# We will overwrite additional_train.csv so the existing train.py just picks it up
final_df.to_csv("data/additional_train.csv", index=False)
print("Overwrote data/additional_train.csv for easy training.")
