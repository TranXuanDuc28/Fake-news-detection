import pandas as pd
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load the main train dataset
df_train = pd.read_csv("data/train.csv")
# Load the additional train dataset
df_additional = pd.read_csv("data/additional_train.csv")

# Combine them just like train.py does when --additional_dataset both is used
df_combined = pd.concat([df_train, df_additional], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=['post_message'])

total = len(df_combined)
real_count = len(df_combined[df_combined['label'] == 0])
fake_count = len(df_combined[df_combined['label'] == 1])

print(f"Tổng số mẫu sau gộp: {total}")
print(f" - TIN THẬT (REAL): {real_count} mẫu ({real_count/total*100:.1f}%)")
print(f" - TIN GIẢ (FAKE): {fake_count} mẫu ({fake_count/total*100:.1f}%)")
