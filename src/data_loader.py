import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

_WARNED_PYVI = False

def clean_vietnamese_text(text, segment_words=False):
    """Cleans Vietnamese text by removing URLs, HTML tags, emails, social mentions, and weird symbols."""
    global _WARNED_PYVI
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove social media mentions/usernames
    text = re.sub(r'@\w+', '', text)
    # Keep only alphanumeric (including Vietnamese characters), spaces, and basic punctuation
    text = re.sub(r'[^\w\s,\.\?\!\-\:\#]', '', text)
    # Standardize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    if segment_words:
        if HAS_PYVI:
            text = ViTokenizer.tokenize(text)
        else:
            if not _WARNED_PYVI:
                print("Warning: pyvi is not installed or failed to import. Skipping word segmentation.")
                _WARNED_PYVI = True
            
    return text


class Vocab:
    """Simple vocabulary builder for the LSTM model."""
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.vocab_size = 2

    def build_vocab(self, texts):
        freqs = {}
        for text in texts:
            # Clean and split text
            tokens = self._tokenize(text)
            for word in tokens:
                freqs[word] = freqs.get(word, 0) + 1

        for word, freq in freqs.items():
            if freq >= self.min_freq:
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def _tokenize(self, text):
        # Basic split by space as it is already cleaned
        return str(text).split()

    def encode(self, text, max_len=128):
        tokens = self._tokenize(text)
        idxs = [self.word2idx.get(tok, self.word2idx["<unk>"]) for tok in tokens]
        if len(idxs) < max_len:
            idxs = idxs + [self.word2idx["<pad>"]] * (max_len - len(idxs))
        else:
            idxs = idxs[:max_len]
        return torch.tensor(idxs, dtype=torch.long)


class LSTMDataset(Dataset):
    """Dataset for the LSTM model."""
    def __init__(self, df, vocab, max_len=128):
        self.labels = df["label"].values
        self.texts = df["post_message"].values
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = self.vocab.encode(text, max_len=self.max_len)
        return input_ids, torch.tensor(label, dtype=torch.long)


class TransformerDataset(Dataset):
    """Dataset for the Transformer model."""
    def __init__(self, df, tokenizer, max_len=128):
        self.labels = df["label"].values
        self.texts = df["post_message"].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension added by tokenizer
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["label"] = torch.tensor(label, dtype=torch.long)
        return item


def load_raw_data(data_dir="data", segment_words=False, additional_dataset="none", use_additional=None):
    """Loads train, val, and test dataframes, handles missing values, checks for additional datasets, and applies text cleaning."""
    # Map legacy use_additional boolean parameter for backward compatibility
    if use_additional is not None:
        if not use_additional:
            additional_dataset = "none"
        elif additional_dataset == "none":
            additional_dataset = "legacy"

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    additional_dfs = {"train": [], "val": [], "test": []}
    
    # Check additional_dataset mode and load matching splits
    if additional_dataset == "vfnd":
        print("--> Đang tải bộ dữ liệu phụ: VFND (Original)...")
        additional_dfs["train"].append(pd.read_csv(os.path.join(data_dir, "vfnd_train.csv")))
        additional_dfs["val"].append(pd.read_csv(os.path.join(data_dir, "vfnd_val.csv")))
        additional_dfs["test"].append(pd.read_csv(os.path.join(data_dir, "vfnd_test.csv")))
    elif additional_dataset == "tingia":
        print("--> Đang tải bộ dữ liệu phụ: TinGia Crawled (tingia.gov.vn)...")
        additional_dfs["train"].append(pd.read_csv(os.path.join(data_dir, "tingia_train.csv")))
        additional_dfs["val"].append(pd.read_csv(os.path.join(data_dir, "tingia_val.csv")))
        additional_dfs["test"].append(pd.read_csv(os.path.join(data_dir, "tingia_test.csv")))
    elif additional_dataset == "both":
        print("--> Đang tải bộ dữ liệu phụ: Cả VFND và TinGia Crawled...")
        additional_dfs["train"].append(pd.read_csv(os.path.join(data_dir, "vfnd_train.csv")))
        additional_dfs["train"].append(pd.read_csv(os.path.join(data_dir, "tingia_train.csv")))
        additional_dfs["val"].append(pd.read_csv(os.path.join(data_dir, "vfnd_val.csv")))
        additional_dfs["val"].append(pd.read_csv(os.path.join(data_dir, "tingia_val.csv")))
        additional_dfs["test"].append(pd.read_csv(os.path.join(data_dir, "vfnd_test.csv")))
        additional_dfs["test"].append(pd.read_csv(os.path.join(data_dir, "tingia_test.csv")))
    elif additional_dataset == "legacy":
        # Load data/additional_train.csv if it exists
        legacy_path = os.path.join(data_dir, "additional_train.csv")
        if os.path.exists(legacy_path):
            print(f"--> Phát hiện bộ dữ liệu bổ sung: {legacy_path}. Tiến hành gộp dữ liệu...")
            additional_dfs["train"].append(pd.read_csv(legacy_path))
        # Load validation and test files from VFND and TinGia to match the 'both' setting
        additional_dfs["val"].append(pd.read_csv(os.path.join(data_dir, "vfnd_val.csv")))
        additional_dfs["val"].append(pd.read_csv(os.path.join(data_dir, "tingia_val.csv")))
        additional_dfs["test"].append(pd.read_csv(os.path.join(data_dir, "vfnd_test.csv")))
        additional_dfs["test"].append(pd.read_csv(os.path.join(data_dir, "tingia_test.csv")))
            
    # Concatenate and clean
    if additional_dfs["train"]:
        print(f"--> Gộp thêm tập Train từ: {additional_dataset}")
        train_df = pd.concat([train_df] + additional_dfs["train"], ignore_index=True)
    if additional_dfs["val"]:
        print(f"--> Gộp thêm tập Val từ: {additional_dataset}")
        val_df = pd.concat([val_df] + additional_dfs["val"], ignore_index=True)
    if additional_dfs["test"]:
        print(f"--> Gộp thêm tập Test từ: {additional_dataset}")
        test_df = pd.concat([test_df] + additional_dfs["test"], ignore_index=True)

    # Perform deduplication to prevent data leakage and redundancy
    before_train = len(train_df)
    train_df = train_df.drop_duplicates(subset=["post_message"], keep="first").reset_index(drop=True)
    if len(train_df) < before_train:
        print(f"--> Đã loại bỏ {before_train - len(train_df)} dòng trùng lặp trong tập Train.")
        
    before_val = len(val_df)
    val_df = val_df.drop_duplicates(subset=["post_message"], keep="first").reset_index(drop=True)
    if len(val_df) < before_val:
        print(f"--> Đã loại bỏ {before_val - len(val_df)} dòng trùng lặp trong tập Val.")
        
    before_test = len(test_df)
    test_df = test_df.drop_duplicates(subset=["post_message"], keep="first").reset_index(drop=True)
    if len(test_df) < before_test:
        print(f"--> Đã loại bỏ {before_test - len(test_df)} dòng trùng lặp trong tập Test.")

    # Fill NaN post messages with empty string
    train_df["post_message"] = train_df["post_message"].fillna("")
    val_df["post_message"] = val_df["post_message"].fillna("")
    test_df["post_message"] = test_df["post_message"].fillna("")
    
    # Apply text cleaning
    train_df["post_message"] = train_df["post_message"].apply(lambda x: clean_vietnamese_text(x, segment_words=segment_words))
    val_df["post_message"] = val_df["post_message"].apply(lambda x: clean_vietnamese_text(x, segment_words=segment_words))
    test_df["post_message"] = test_df["post_message"].apply(lambda x: clean_vietnamese_text(x, segment_words=segment_words))
    
    return train_df, val_df, test_df



def get_dataloaders(data_dir="data", model_type="lstm", batch_size=16, max_len=128, subset_size=None, tokenizer_name="distilbert-base-multilingual-cased", oversample=False, segment_words=None, additional_dataset="none", use_additional=None):
    """Creates PyTorch dataloaders for the specified model type with optional oversampling."""
    if segment_words is None:
        segment_words = ("phobert" in tokenizer_name.lower())
    train_df, val_df, test_df = load_raw_data(data_dir, segment_words=segment_words, additional_dataset=additional_dataset, use_additional=use_additional)
    
    # Optional sub-sampling for faster hyperparameter search
    if subset_size is not None:
        train_df = train_df.sample(n=min(subset_size, len(train_df)), random_state=42).reset_index(drop=True)
        val_subset_size = int(subset_size * 0.2)
        val_df = val_df.sample(n=min(val_subset_size, len(val_df)), random_state=42).reset_index(drop=True)
    
    # Apply random oversampling on training set to handle class imbalance
    if oversample:
        df_real = train_df[train_df['label'] == 0]
        df_fake = train_df[train_df['label'] == 1]
        if len(df_fake) < len(df_real):
            df_fake_oversampled = df_fake.sample(len(df_real), replace=True, random_state=42)
            train_df = pd.concat([df_real, df_fake_oversampled], ignore_index=True).reset_index(drop=True)
            
    if model_type == "lstm":
        # Build vocabulary on training set
        vocab = Vocab()
        vocab.build_vocab(train_df["post_message"].values)
        
        train_dataset = LSTMDataset(train_df, vocab, max_len=max_len)
        val_dataset = LSTMDataset(val_df, vocab, max_len=max_len)
        test_dataset = LSTMDataset(test_df, vocab, max_len=max_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, vocab
        
    elif model_type == "transformer":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        train_dataset = TransformerDataset(train_df, tokenizer, max_len=max_len)
        val_dataset = TransformerDataset(val_df, tokenizer, max_len=max_len)
        test_dataset = TransformerDataset(test_df, tokenizer, max_len=max_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, tokenizer
    else:
        raise ValueError(f"Unknown model type: {model_type}")

