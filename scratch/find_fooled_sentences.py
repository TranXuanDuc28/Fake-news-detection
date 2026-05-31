import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer

# Adjust paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predict import clean_vietnamese_text, VocabHelper
from src.lstm_model import BiLSTMClassifier, LSTMClassifier
from src.transformer_model import TransformerClassifier

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    lstm_path = "models/best_lstm.pt"
    lstm_1d_path = "models/best_lstm_1d.pt"
    trans_path = "models/best_transformer_phobert.pt"
    test_path = "data/test.csv"

    # Thresholds
    lstm_thresh = 0.63
    lstm_1d_thresh = 0.50
    trans_thresh = 0.50

    # 1. Load LSTM
    print("Loading BiLSTM...")
    checkpoint_lstm = torch.load(lstm_path, map_location=device)
    vocab_w2i = checkpoint_lstm["vocab_word2idx"]
    lstm_vocab = VocabHelper(vocab_w2i)
    hyperparams_lstm = checkpoint_lstm.get("hyperparameters", {})
    lstm_segment = hyperparams_lstm.get("segment_words", False)
    lstm_model = BiLSTMClassifier(
        vocab_size=len(vocab_w2i),
        embedding_dim=hyperparams_lstm.get("embedding_dim", 128) or 128,
        hidden_dim=hyperparams_lstm.get("hidden_dim", 128) or 128,
        dropout=hyperparams_lstm.get("dropout", 0.3)
    )
    lstm_model.load_state_dict(checkpoint_lstm["model_state_dict"])
    lstm_model.to(device)
    lstm_model.eval()

    # 2. Load LSTM 1D
    print("Loading LSTM 1D...")
    checkpoint_lstm_1d = torch.load(lstm_1d_path, map_location=device)
    vocab_w2i_1d = checkpoint_lstm_1d["vocab_word2idx"]
    lstm_1d_vocab = VocabHelper(vocab_w2i_1d)
    hyperparams_lstm_1d = checkpoint_lstm_1d.get("hyperparameters", {})
    lstm_1d_segment = hyperparams_lstm_1d.get("segment_words", False)
    lstm_1d_model = LSTMClassifier(
        vocab_size=len(vocab_w2i_1d),
        embedding_dim=hyperparams_lstm_1d.get("embedding_dim", 128) or 128,
        hidden_dim=hyperparams_lstm_1d.get("hidden_dim", 128) or 128,
        dropout=hyperparams_lstm_1d.get("dropout", 0.3)
    )
    lstm_1d_model.load_state_dict(checkpoint_lstm_1d["model_state_dict"])
    lstm_1d_model.to(device)
    lstm_1d_model.eval()

    # 3. Load Transformer
    print("Loading Transformer...")
    checkpoint_trans = torch.load(trans_path, map_location=device)
    trans_model_name = checkpoint_trans.get("hyperparameters", {}).get("transformer_model_name", "vinai/phobert-base")
    trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
    trans_segment = checkpoint_trans.get("hyperparameters", {}).get("segment_words", True)
    trans_model = TransformerClassifier(
        model_name=trans_model_name,
        dropout=checkpoint_trans["hyperparameters"].get("dropout", 0.3),
        freeze_backbone=True
    )
    trans_model.load_state_dict(checkpoint_trans["model_state_dict"])
    trans_model.to(device)
    trans_model.eval()

    # Read test CSV
    print(f"Reading test data from {test_path}...")
    df = pd.read_csv(test_path)
    
    results = []
    
    # 1. Run BiLSTM predictions on all rows (extremely fast)
    print("Running BiLSTM pre-filtering...")
    bilstm_errors = []
    for idx, row in df.iterrows():
        text = str(row['post_message'])
        label = int(row['label'])  # 0: Real, 1: Fake
        
        # Predict BiLSTM
        in_lstm = lstm_vocab.encode(text, max_len=128, segment_words=lstm_segment).to(device)
        with torch.no_grad():
            out_lstm = lstm_model(in_lstm)
            prob_lstm = torch.softmax(out_lstm, dim=1).squeeze(0)
            p_lstm_fake = prob_lstm[1].item()
            pred_lstm = 1 if p_lstm_fake >= lstm_thresh else 0
            
        # Predict LSTM 1D
        in_lstm_1d = lstm_1d_vocab.encode(text, max_len=128, segment_words=lstm_1d_segment).to(device)
        with torch.no_grad():
            out_lstm_1d = lstm_1d_model(in_lstm_1d)
            prob_lstm_1d = torch.softmax(out_lstm_1d, dim=1).squeeze(0)
            p_lstm_1d_fake = prob_lstm_1d[1].item()
            pred_lstm_1d = 1 if p_lstm_1d_fake >= lstm_1d_thresh else 0
            
        if pred_lstm != label:
            bilstm_errors.append({
                "idx": idx,
                "text": text,
                "label": label,
                "pred_lstm": pred_lstm,
                "p_lstm_fake": p_lstm_fake,
                "pred_lstm_1d": pred_lstm_1d,
                "p_lstm_1d_fake": p_lstm_1d_fake
            })

    print(f"Found {len(bilstm_errors)} candidates where BiLSTM is incorrect. Evaluating Transformer on these...")
    
    # 2. Run Transformer only on those candidates
    for i, candidate in enumerate(bilstm_errors):
        text = candidate["text"]
        label = candidate["label"]
        pred_lstm = candidate["pred_lstm"]
        p_lstm_fake = candidate["p_lstm_fake"]
        pred_lstm_1d = candidate["pred_lstm_1d"]
        p_lstm_1d_fake = candidate["p_lstm_1d_fake"]
        
        # Predict Transformer
        cleaned_text = clean_vietnamese_text(text, segment_words=trans_segment)
        inputs = trans_tokenizer(cleaned_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out_trans = trans_model(input_ids=input_ids, attention_mask=attention_mask)
            prob_trans = torch.softmax(out_trans, dim=1).squeeze(0)
            p_trans_fake = prob_trans[1].item()
            pred_trans = 1 if p_trans_fake >= trans_thresh else 0
            
        # Check if Transformer is correct
        if pred_trans == label:
            results.append({
                "text": text,
                "label": "FAKE" if label == 1 else "REAL",
                "bilstm_pred": "FAKE" if pred_lstm == 1 else "REAL",
                "bilstm_prob": p_lstm_fake if pred_lstm == 1 else (1.0 - p_lstm_fake),
                "lstm1d_pred": "FAKE" if pred_lstm_1d == 1 else "REAL",
                "lstm1d_prob": p_lstm_1d_fake if pred_lstm_1d == 1 else (1.0 - p_lstm_1d_fake),
                "trans_pred": "FAKE" if pred_trans == 1 else "REAL",
                "trans_prob": p_trans_fake if pred_trans == 1 else (1.0 - p_trans_fake)
            })
            print(f"Match found! ({len(results)}/15)")
            
        if len(results) >= 15:
            break
            
    print(f"\nFound {len(results)} examples where BiLSTM is fooled but Transformer is correct:")
    for i, res in enumerate(results):
        print(f"\n[{i+1}] Label: {res['label']}")
        print(f"Text: {res['text']}")
        print(f"BiLSTM pred: {res['bilstm_pred']} (Prob: {res['bilstm_prob']:.4f})")
        print(f"LSTM1D pred: {res['lstm1d_pred']} (Prob: {res['lstm1d_prob']:.4f})")
        print(f"Transformer pred: {res['trans_pred']} (Prob: {res['trans_prob']:.4f})")

if __name__ == "__main__":
    main()
