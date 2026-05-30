import pandas as pd

train = pd.read_csv("data/train.csv")
additional = pd.read_csv("data/additional_train.csv")

def clean(text):
    return str(text).lower().strip().replace(" ", "")

train_set = set(train['post_message'].fillna("").apply(clean))
additional_set = set(additional['post_message'].fillna("").apply(clean))

overlap = train_set.intersection(additional_set)
print(f"Total train unique: {len(train_set)}")
print(f"Total additional unique: {len(additional_set)}")
print(f"Overlap between train and additional: {len(overlap)}")

# Print some overlap examples if they exist
if len(overlap) > 0:
    print("\nOverlap examples:")
    for i, x in enumerate(list(overlap)[:5]):
        print(f"{i+1}: {x[:100]}...")
