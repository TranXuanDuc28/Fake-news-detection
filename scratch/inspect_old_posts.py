import pandas as pd
import sys

# Ensure utf-8 output encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/additional_train_old.csv')

print("--- Label 0 (Real) count:", len(df[df['label']==0]))
print(df[df['label']==0]['post_message'].head(5))

print("\n--- Label 1 (Fake) count:", len(df[df['label']==1]))
print(df[df['label']==1]['post_message'].head(5))
