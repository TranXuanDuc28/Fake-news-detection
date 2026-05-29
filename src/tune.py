import os
import sys
# Set system output to UTF-8 to prevent encoding errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path to allow direct scripts execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import json
import torch
import torch.nn as nn
from src.data_loader import get_dataloaders
from src.lstm_model import BiLSTMClassifier
from src.transformer_model import TransformerClassifier
from src.train import train_model


def run_parameter_sweep(save_dir="models", data_dir="data", output_path="data/tuning_results.json", epochs=3, subset_size=None, oversample=True, use_class_weights=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_gpu = torch.cuda.is_available()
    
    # We will use a subset for tuning to ensure it runs quickly.
    # On CPU: 800 samples is fast. On GPU (Colab): 2000 samples is fast.
    if subset_size is None:
        subset_size = 2000 if is_gpu else 800
    
    results = {
        "lstm": {
            "dropout_sweep": {},
            "batch_size_sweep": {},
            "lr_sweep": {}
        },
        "transformer": {
            "dropout_sweep": {},
            "batch_size_sweep": {},
            "lr_sweep": {}
        }
    }
    
    # Defaults
    default_lstm_lr = 1e-3
    default_lstm_dropout = 0.3
    default_lstm_batch = 16
    
    # If using GPU, we fine-tune the entire transformer. If CPU, we freeze the backbone.
    freeze_backbone = not is_gpu
    default_trans_lr = 2e-5 if is_gpu else 2e-3
    default_trans_dropout = 0.3
    default_trans_batch = 16
    
    print("="*50)
    print("STARTING HYPERPARAMETER TUNING SWEEP")
    print(f"Hardware: {'GPU (Full Fine-tuning)' if is_gpu else 'CPU (Frozen Backbone)'}")
    print(f"Tuning Subset Size: {subset_size} | Epochs per run: {epochs}")
    print(f"Saving checkpoints to: {save_dir}")
    print(f"Loading data from: {data_dir}")
    print("="*50)
    
    # ------------------ LSTM SWEEPS ------------------
    print("\n>>> Tuning LSTM Model...")
    
    # 1. LSTM Dropout Sweep (0.1, 0.3, 0.5)
    for dp in [0.1, 0.3, 0.5]:
        print(f"\n[LSTM Sweep] Testing Dropout = {dp}")
        run_save_dir = os.path.join(save_dir, f"lstm_lr{default_lstm_lr}_bs{default_lstm_batch}_dp{dp}")
        history, test_m = train_model(
            model_type="lstm", epochs=epochs, batch_size=default_lstm_batch,
            lr=default_lstm_lr, dropout=dp, subset_size=subset_size,
            save_dir=run_save_dir, data_dir=data_dir,
            oversample=oversample, use_class_weights=use_class_weights
        )
        results["lstm"]["dropout_sweep"][str(dp)] = {
            "val_f1_history": history["val_f1"],
            "val_loss_history": history["val_loss"],
            "final_metrics": test_m
        }
        
    # 2. LSTM Batch Size Sweep (8, 16, 32)
    for bs in [8, 16, 32]:
        print(f"\n[LSTM Sweep] Testing Batch Size = {bs}")
        run_save_dir = os.path.join(save_dir, f"lstm_lr{default_lstm_lr}_bs{bs}_dp{default_lstm_dropout}")
        history, test_m = train_model(
            model_type="lstm", epochs=epochs, batch_size=bs,
            lr=default_lstm_lr, dropout=default_lstm_dropout, subset_size=subset_size,
            save_dir=run_save_dir, data_dir=data_dir,
            oversample=oversample, use_class_weights=use_class_weights
        )
        results["lstm"]["batch_size_sweep"][str(bs)] = {
            "val_f1_history": history["val_f1"],
            "val_loss_history": history["val_loss"],
            "final_metrics": test_m
        }
        
    # 3. LSTM Learning Rate Sweep (1e-4, 1e-3, 5e-3)
    for lr in [1e-4, 1e-3, 5e-3]:
        print(f"\n[LSTM Sweep] Testing Learning Rate = {lr}")
        run_save_dir = os.path.join(save_dir, f"lstm_lr{lr}_bs{default_lstm_batch}_dp{default_lstm_dropout}")
        history, test_m = train_model(
            model_type="lstm", epochs=epochs, batch_size=default_lstm_batch,
            lr=lr, dropout=default_lstm_dropout, subset_size=subset_size,
            save_dir=run_save_dir, data_dir=data_dir,
            oversample=oversample, use_class_weights=use_class_weights
        )
        results["lstm"]["lr_sweep"][str(lr)] = {
            "val_f1_history": history["val_f1"],
            "val_loss_history": history["val_loss"],
            "final_metrics": test_m
        }
        
    # ------------------ TRANSFORMER SWEEPS ------------------
    print("\n>>> Tuning Transformer Model...")
    
    # 1. Transformer Dropout Sweep (0.1, 0.3, 0.5)
    for dp in [0.1, 0.3, 0.5]:
        print(f"\n[Transformer Sweep] Testing Dropout = {dp}")
        run_save_dir = os.path.join(save_dir, f"transformer_lr{default_trans_lr}_bs{default_trans_batch}_dp{dp}")
        history, test_m = train_model(
            model_type="transformer", epochs=epochs, batch_size=default_trans_batch,
            lr=default_trans_lr, dropout=dp, freeze_backbone=freeze_backbone, subset_size=subset_size,
            save_dir=run_save_dir, data_dir=data_dir,
            oversample=oversample, use_class_weights=use_class_weights
        )
        results["transformer"]["dropout_sweep"][str(dp)] = {
            "val_f1_history": history["val_f1"],
            "val_loss_history": history["val_loss"],
            "final_metrics": test_m
        }
        
    # 2. Transformer Batch Size Sweep (8, 16, 32)
    for bs in [8, 16, 32]:
        print(f"\n[Transformer Sweep] Testing Batch Size = {bs}")
        run_save_dir = os.path.join(save_dir, f"transformer_lr{default_trans_lr}_bs{bs}_dp{default_trans_dropout}")
        history, test_m = train_model(
            model_type="transformer", epochs=epochs, batch_size=bs,
            lr=default_trans_lr, dropout=default_trans_dropout, freeze_backbone=freeze_backbone, subset_size=subset_size,
            save_dir=run_save_dir, data_dir=data_dir,
            oversample=oversample, use_class_weights=use_class_weights
        )
        results["transformer"]["batch_size_sweep"][str(bs)] = {
            "val_f1_history": history["val_f1"],
            "val_loss_history": history["val_loss"],
            "final_metrics": test_m
        }
        
    # 3. Transformer Learning Rate Sweep
    # If on GPU (unfrozen), search standard: [1e-5, 2e-5, 5e-5]. If on CPU (frozen), search: [1e-3, 2e-3, 5e-3]
    lrs = [1e-5, 2e-5, 5e-5] if is_gpu else [1e-3, 2e-3, 5e-3]
    for lr in lrs:
        print(f"\n[Transformer Sweep] Testing Learning Rate = {lr}")
        run_save_dir = os.path.join(save_dir, f"transformer_lr{lr}_bs{default_trans_batch}_dp{default_trans_dropout}")
        history, test_m = train_model(
            model_type="transformer", epochs=epochs, batch_size=default_trans_batch,
            lr=lr, dropout=default_trans_dropout, freeze_backbone=freeze_backbone, subset_size=subset_size,
            save_dir=run_save_dir, data_dir=data_dir,
            oversample=oversample, use_class_weights=use_class_weights
        )
        results["transformer"]["lr_sweep"][str(lr)] = {
            "val_f1_history": history["val_f1"],
            "val_loss_history": history["val_loss"],
            "final_metrics": test_m
        }
        
    # Save sweep results to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*50)
    print(f"HYPERPARAMETER TUNING COMPLETED. Results saved to '{output_path}'")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Hyperparameter Sweep for LSTM and Transformer models")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save model checkpoints during tuning")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory where dataset files are located")
    parser.add_argument("--output", type=str, default="data/tuning_results.json", help="Path to save tuning results JSON file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs per training run")
    parser.add_argument("--subset", type=int, default=None, help="Subset size for quick tuning (overrides default CPU/GPU sizes)")
    parser.add_argument("--no_oversample", action="store_false", dest="oversample", help="Disable random oversampling on train dataset")
    parser.add_argument("--use_class_weights", action="store_true", help="Use weighted CrossEntropyLoss (only active if oversample is disabled)")
    
    args = parser.parse_args()
    
    run_parameter_sweep(
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        subset_size=args.subset,
        oversample=args.oversample,
        use_class_weights=args.use_class_weights
    )


