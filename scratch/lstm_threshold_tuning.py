import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Set system output to UTF-8 to prevent encoding errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import get_dataloaders
from src.lstm_model import BiLSTMClassifier
from src.data_loader import Vocab  # Needed if Vocab is loaded

def tune_threshold(model_path="models/best_lstm.pt", data_dir="data", additional_dataset="both"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {model_path} on device: {device}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return
        
    checkpoint = torch.load(model_path, map_location=device)
    hyperparams = checkpoint["hyperparameters"]
    vocab_word2idx = checkpoint["vocab_word2idx"]
    
    # Reconstruct vocab
    vocab = Vocab()
    vocab.word2idx = vocab_word2idx
    vocab.vocab_size = len(vocab_word2idx)
    
    # Load loaders
    print("Loading data splits...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        data_dir=data_dir,
        model_type="lstm",
        batch_size=16,
        additional_dataset=additional_dataset
    )
    
    # Reconstruct model
    model = BiLSTMClassifier(
        vocab_size=vocab.vocab_size,
        embedding_dim=hyperparams.get("embedding_dim", 100),
        hidden_dim=hyperparams.get("hidden_dim", 64),
        dropout=hyperparams.get("dropout", 0.3)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Evaluate function to get soft probabilities
    def get_probs_and_targets(loader):
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(targets.numpy())
        return np.array(all_probs), np.array(all_targets)
        
    print("Extracting probabilities on Val set...")
    val_probs, val_targets = get_probs_and_targets(val_loader)
    print("Extracting probabilities on Test set...")
    test_probs, test_targets = get_probs_and_targets(test_loader)
    
    # Search for best threshold on Val set
    best_threshold = 0.5
    best_val_f1 = 0.0
    
    thresholds = np.linspace(0.01, 0.99, 99)
    print("\n--- Threshold Sweep on Validation Set ---")
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(val_targets, preds, average='binary', zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_threshold = t
            
    print(f"\nBest Decision Threshold on Val: {best_threshold:.4f} (Val F1: {best_val_f1*100:.2f}%)")
    
    # Evaluate at default threshold 0.5 on Test set
    default_test_preds = (test_probs >= 0.5).astype(int)
    dp, dr, df1, _ = precision_recall_fscore_support(test_targets, default_test_preds, average='binary', zero_division=0)
    d_acc = accuracy_score(test_targets, default_test_preds)
    
    # Evaluate at optimal threshold on Test set
    opt_test_preds = (test_probs >= best_threshold).astype(int)
    op, or_, of1, _ = precision_recall_fscore_support(test_targets, opt_test_preds, average='binary', zero_division=0)
    o_acc = accuracy_score(test_targets, opt_test_preds)
    
    print("\n" + "="*50)
    print(f"TEST METRICS COMPARISON (Threshold: 0.50 vs {best_threshold:.2f})")
    print("="*50)
    print(f"Default (t = 0.50)  | Accuracy: {d_acc*100:.2f}% | Precision: {dp*100:.2f}% | Recall: {dr*100:.2f}% | F1-score: {df1*100:.2f}%")
    print(f"Optimal (t = {best_threshold:.2f})  | Accuracy: {o_acc*100:.2f}% | Precision: {op*100:.2f}% | Recall: {or_*100:.2f}% | F1-score: {of1*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tune decision threshold for BiLSTM model.")
    parser.add_argument("--model_path", type=str, default="models/best_lstm.pt", help="Path to best_lstm.pt model file")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset files")
    parser.add_argument("--additional_dataset", type=str, default="both", choices=["none", "vfnd", "tingia", "both"], help="Additional datasets used")
    args = parser.parse_args()
    
    tune_threshold(
        model_path=args.model_path,
        data_dir=args.data_dir,
        additional_dataset=args.additional_dataset
    )
