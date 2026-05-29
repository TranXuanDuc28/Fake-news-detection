import os
import sys
# Set system output to UTF-8 to prevent encoding errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path to allow direct scripts execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.data_loader import get_dataloaders
from src.lstm_model import BiLSTMClassifier
from src.transformer_model import TransformerClassifier


# Set seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def train_epoch(model, dataloader, optimizer, criterion, device, model_type):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        if model_type == "lstm":
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
        else: # transformer
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_targets, all_preds)
    
    return avg_loss, acc

def evaluate(model, dataloader, criterion, device, model_type, return_predictions=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            if model_type == "lstm":
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            else: # transformer
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    acc = accuracy_score(all_targets, all_preds)
    # Binary metrics (class 1 = fake news)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    
    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "precision_binary": p_bin,
        "recall_binary": r_bin,
        "f1_binary": f1_bin,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro
    }
    
    if return_predictions:
        return metrics, all_preds, all_targets
    return metrics

def train_model(model_type, epochs=5, batch_size=16, lr=1e-3, dropout=0.3, freeze_backbone=True, subset_size=None, save_dir="models", data_dir="data", oversample=True, use_class_weights=False, patience=5, resume=True, transformer_model_name="vinai/phobert-base", segment_words=None):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training {model_type.upper()} model on device: {device} ---")
    
    # Resolve segment_words actual value for printing
    resolved_segment_words = segment_words if segment_words is not None else ("phobert" in transformer_model_name.lower() if model_type == "transformer" else False)
    print(f"Params: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, Dropout={dropout}, Subset={subset_size}, Oversample={oversample}, UseClassWeights={use_class_weights}, Patience={patience}, SegmentWords={resolved_segment_words}")
    
    # Get DataLoaders
    if model_type == "lstm":
        train_loader, val_loader, test_loader, vocab = get_dataloaders(
            data_dir=data_dir, model_type="lstm", batch_size=batch_size, subset_size=subset_size, oversample=oversample, segment_words=segment_words
        )
        model = BiLSTMClassifier(
            vocab_size=vocab.vocab_size,
            embedding_dim=128,
            hidden_dim=128,
            dropout=dropout
        )
    else: # transformer
        train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
            data_dir=data_dir, model_type="transformer", batch_size=batch_size, subset_size=subset_size, oversample=oversample, tokenizer_name=transformer_model_name, segment_words=segment_words
        )
        model = TransformerClassifier(
            model_name=transformer_model_name,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        
    model = model.to(device)
    
    # Define Loss with optional Class Weights
    if use_class_weights and not oversample:
        labels = train_loader.dataset.labels
        num_real = (labels == 0).sum()
        num_fake = (labels == 1).sum()
        total = num_real + num_fake
        
        weight_real = total / (2.0 * num_real)
        weight_fake = total / (2.0 * num_fake)
        class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Weighted Loss Active: Weights = [Real: {weight_real:.4f}, Fake: {weight_fake:.4f}]")
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Define learning rate scheduler (halves learning rate if validation F1 does not improve for 2 epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_f1 = -1.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_f1": [], "val_acc": []}

    start_epoch = 0
    checkpoint_path = os.path.join(save_dir, f"checkpoint_latest_{model_type}.pt")
    if resume and os.path.exists(checkpoint_path):
        print(f"--> Phát hiện checkpoint cũ tại {checkpoint_path}. Tiến hành khôi phục...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_f1 = checkpoint["best_val_f1"]
            history = checkpoint["history"]
            # Restore optimizer lr in case it changed
            for param_group in optimizer.param_groups:
                print(f"--> Tốc độ học (Learning Rate) khôi phục: {param_group['lr']:.6f}")
            print(f"--> Khôi phục thành công! Tiếp tục huấn luyện từ Epoch {start_epoch + 1}/{epochs}")
        except Exception as e:
            print(f"--> Cảnh báo: Lỗi khi tải checkpoint: {e}. Tiến hành huấn luyện lại từ đầu.")

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, model_type)
        val_metrics = evaluate(model, val_loader, criterion, device, model_type)
        
        # Save metrics history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1_binary"])
        history["val_acc"].append(val_metrics["accuracy"])
        
        current_lr = optimizer.param_groups[0]['lr']
        # Step the scheduler based on validation F1 score
        scheduler.step(val_metrics["f1_binary"])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"--> Tốc độ học (Learning Rate) giảm từ {current_lr:.6f} xuống {new_lr:.6f}")
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val F1 (Bin): {val_metrics['f1_binary']*100:.2f}% | "
              f"Val Acc: {val_metrics['accuracy']*100:.2f}% | LR: {new_lr:.6f}")
        
        # Save best checkpoint (based on binary F1-score)
        if val_metrics["f1_binary"] > best_val_f1:
            best_val_f1 = val_metrics["f1_binary"]
            patience_counter = 0
            best_model_path = os.path.join(save_dir, f"best_{model_type}.pt")
            
            # Save weights and metadata
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_type": model_type,
                "hyperparameters": {
                    "lr": new_lr,
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "segment_words": resolved_segment_words,
                    "freeze_backbone": freeze_backbone if model_type == "transformer" else None,
                    "transformer_model_name": transformer_model_name if model_type == "transformer" else None
                }
            }
            # For LSTM we must save vocab so we can encode words for inference
            if model_type == "lstm":
                checkpoint["vocab_word2idx"] = vocab.word2idx
                
            torch.save(checkpoint, best_model_path)
            # Save metadata json for quick reading
            meta_path = os.path.join(save_dir, f"best_{model_type}_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=4)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n--> Early Stopping kích hoạt! Không cải thiện trên validation F1 trong {patience} epoch liên tiếp.")
                break
                
        # Save latest checkpoint for resuming
        checkpoint_latest = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
            "history": history
        }
        torch.save(checkpoint_latest, checkpoint_path)
                
    # Clean up latest checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print("--> Đã dọn dẹp tệp checkpoint tạm thời.")
        except Exception as e:
            pass
            
    # Evaluate best model on test set
    print(f"\nEvaluating best {model_type.upper()} model on Test Set...")
    best_model_path = os.path.join(save_dir, f"best_{model_type}.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics, test_preds, test_targets = evaluate(model, test_loader, criterion, device, model_type, return_predictions=True)
        print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"Test Precision (Binary): {test_metrics['precision_binary']*100:.2f}%")
        print(f"Test Recall (Binary): {test_metrics['recall_binary']*100:.2f}%")
        print(f"Test F1-score (Binary): {test_metrics['f1_binary']*100:.2f}%")
        print(f"Test F1-score (Macro): {test_metrics['f1_macro']*100:.2f}%")
        
        # Save test results
        test_results = test_metrics.copy()
        test_results["predictions"] = [int(x) for x in test_preds]
        test_results["targets"] = [int(x) for x in test_targets]
        
        with open(os.path.join(save_dir, f"test_{model_type}_results.json"), "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=4)
            
        # Save training history in the run directory for dynamic loading
        with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
            
    return history, test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM/Transformer model for Vietnamese Fake News Detection")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"], help="Model type to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, choices=[8, 16, 32], help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, choices=[0.1, 0.3, 0.5], help="Dropout rate")
    parser.add_argument("--unfreeze_backbone", action="store_false", dest="freeze_backbone", default=True, help="Unfreeze transformer backbone for full fine-tuning (recommended for GPU)")
    parser.add_argument("--subset", type=int, default=None, help="Subset size for fast training (optional)")
    parser.add_argument("--no_oversample", action="store_false", dest="oversample", help="Disable random oversampling on train dataset")
    parser.add_argument("--use_class_weights", action="store_true", help="Use weighted CrossEntropyLoss (only active if oversample is disabled)")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory where dataset files are located")
    parser.add_argument("--history_file", type=str, default=None, help="Optional path to save final training history JSON")
    parser.add_argument("--patience", type=int, default=5, help="Patience epochs for early stopping")
    parser.add_argument("--no_resume", action="store_false", dest="resume", help="Disable automatically resuming from latest checkpoint")
    parser.add_argument("--segment_words", action="store_true", default=None, help="Force Vietnamese word segmentation (using pyvi)")
    parser.add_argument("--no_segment_words", action="store_false", dest="segment_words", help="Disable Vietnamese word segmentation")
    
    parser.add_argument("--transformer_model_name", type=str, default="vinai/phobert-base", help="Pre-trained transformer model name (e.g. vinai/phobert-base, distilbert-base-multilingual-cased)")
    
    args = parser.parse_args()
    
    # Run training
    history, test_metrics = train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        subset_size=args.subset,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        oversample=args.oversample,
        use_class_weights=args.use_class_weights,
        patience=args.patience,
        resume=args.resume,
        transformer_model_name=args.transformer_model_name,
        segment_words=args.segment_words
    )
    
    if args.history_file:
        os.makedirs(os.path.dirname(args.history_file), exist_ok=True)
        with open(args.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {args.history_file}")


