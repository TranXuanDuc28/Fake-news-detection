import os
import sys
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_local import (
    TEST_CASES,
    load_lstm_model,
    load_trans_model,
    clean_vietnamese_text,
)

def evaluate_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load LSTM
    lstm_model, lstm_vocab, lstm_segment = load_lstm_model("models/best_lstm.pt", device)
    
    # 2. Load Transformer
    trans_model, trans_tokenizer, trans_name, trans_segment = load_trans_model("models/best_transformer_phobert.pt", device)
    if trans_model is None:
        trans_model, trans_tokenizer, trans_name, trans_segment = load_trans_model("models/best_transformer.pt", device)

    # Label lists
    y_true = []
    y_pred_lstm = []
    y_pred_trans = []
    
    # Predict
    for case in TEST_CASES:
        text = case["text"]
        label_str = case["label"]
        # label_str is "TIN THẬT (REAL)" (0) or "TIN GIẢ (FAKE)" (1)
        true_label = 1 if "FAKE" in label_str else 0
        y_true.append(true_label)
        
        # LSTM predict
        if lstm_model:
            try:
                inputs = lstm_vocab.encode(text, max_len=128, segment_words=lstm_segment).to(device)
                with torch.no_grad():
                    logits = lstm_model(inputs)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    pred = 1 if probs[1].item() >= 0.5 else 0
                    y_pred_lstm.append(pred)
            except Exception as e:
                y_pred_lstm.append(0) # Default fallback
        
        # Trans predict
        if trans_model:
            try:
                cleaned_text = clean_vietnamese_text(text, segment_words=trans_segment)
                inputs = trans_tokenizer(cleaned_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                with torch.no_grad():
                    logits = trans_model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    pred = 1 if probs[1].item() >= 0.5 else 0
                    y_pred_trans.append(pred)
            except Exception as e:
                y_pred_trans.append(0) # Default fallback

    # Compute metrics
    print("\n" + "="*50)
    print("EVALUATION ON 50 BALANCED TEST CASES")
    print("="*50)
    
    if lstm_model and len(y_pred_lstm) == len(y_true):
        acc = accuracy_score(y_true, y_pred_lstm)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred_lstm, average='binary', zero_division=0)
        print(f"--- BiLSTM ---")
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"Precision: {p*100:.2f}%")
        print(f"Recall:    {r*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        
    if trans_model and len(y_pred_trans) == len(y_true):
        acc = accuracy_score(y_true, y_pred_trans)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred_trans, average='binary', zero_division=0)
        print(f"\n--- PhoBERT (Transformer) ---")
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"Precision: {p*100:.2f}%")
        print(f"Recall:    {r*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")

if __name__ == "__main__":
    evaluate_models()
