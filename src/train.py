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
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.data_loader import get_dataloaders
from src.lstm_model import BiLSTMClassifier, LSTMClassifier
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
    
    # Wrap dataloader in a beautiful tqdm progress bar
    progress_bar = tqdm(dataloader, desc="  Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        
        if model_type in ["lstm", "lstm_1d"]:
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
        
        loss_val = loss.item()
        total_loss += loss_val
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        
        # Display current loss in the progress bar
        progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})
        
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_targets, all_preds)
    
    return avg_loss, acc

def evaluate(model, dataloader, criterion, device, model_type, return_predictions=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    # Wrap validation dataloader in a tqdm progress bar
    progress_bar = tqdm(dataloader, desc="  Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            if model_type in ["lstm", "lstm_1d"]:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
            else: # transformer
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
            loss = criterion(outputs, targets)
            loss_val = loss.item()
            total_loss += loss_val
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
            # Display current loss in the progress bar
            progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})
            
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

def load_pretrained_embeddings(vocab_w2i, segment_words=False, data_dir="data", embedding_dim=100):
    """
    Downloads and loads the pre-trained embeddings (100-dim PhoW2V or 300-dim FastText).
    Maps the vocabulary of our dataset to the pre-trained vectors.
    """
    import urllib.request
    import zipfile
    
    # 1. Choose embedding type based on dimension
    if embedding_dim == 300:
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.vi.vec"
        zip_name = ""
        txt_name = "wiki.vi.vec"
    else:
        embedding_dim = 100 # Ensure it is 100 for PhoW2V
        if segment_words:
            url = "https://public.vinai.io/word2vec_vi_words_100dims.zip"
            zip_name = "word2vec_vi_words_100dims.zip"
            txt_name = "word2vec_vi_words_100dims.txt"
        else:
            url = "https://public.vinai.io/word2vec_vi_syllables_100dims.zip"
            zip_name = "word2vec_vi_syllables_100dims.zip"
            txt_name = "word2vec_vi_syllables_100dims.txt"
        
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, zip_name) if zip_name else ""
    txt_path = os.path.join(data_dir, txt_name)
    
    # 2. Download file if not exists
    if not os.path.exists(txt_path):
        if zip_path:
            # VinAI PhoW2V (Zipped)
            if not os.path.exists(zip_path):
                try:
                    print(f"--> Downloading PhoW2V pre-trained embeddings from {url}...")
                    req = urllib.request.Request(
                        url, 
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    )
                    with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                        out_file.write(response.read())
                    print("--> Download completed successfully!")
                except Exception as e:
                    if os.path.exists(zip_path):
                        try: os.remove(zip_path)
                        except: pass
                    print(f"--> Urllib download failed: {e}. Trying fallback with wget/curl...")
                    try:
                        import subprocess
                        print("--> Attempting download via wget...")
                        subprocess.run(["wget", "-q", "-O", zip_path, url], check=True)
                        print("--> Download completed via wget!")
                    except Exception as wget_err:
                        if os.path.exists(zip_path):
                            try: os.remove(zip_path)
                            except: pass
                        print(f"--> Wget failed: {wget_err}. Attempting download via curl...")
                        try:
                            import subprocess
                            subprocess.run(["curl", "-s", "-L", "-o", zip_path, url], check=True)
                            print("--> Download completed via curl!")
                        except Exception as curl_err:
                            if os.path.exists(zip_path):
                                try: os.remove(zip_path)
                                except: pass
                            print(f"--> Primary URL failed. Trying mirror on Hugging Face...")
                            if not segment_words:
                                hf_mirror_url = "https://huggingface.co/ducdatit2002/vietnamese-emotion-text-classification/resolve/main/word2vec_vi_syllables_100dims.txt?download=true"
                                print(f"--> Downloading syllable mirror from Hugging Face: {hf_mirror_url} ...")
                                try:
                                    import subprocess
                                    subprocess.run(["wget", "-q", "-O", txt_path, hf_mirror_url], check=True)
                                    print("--> Mirror download completed via wget!")
                                except Exception as hf_wget_err:
                                    try:
                                        import subprocess
                                        subprocess.run(["curl", "-s", "-L", "-o", txt_path, hf_mirror_url], check=True)
                                        print("--> Mirror download completed via curl!")
                                    except Exception as hf_curl_err:
                                        print(f"--> Mirror download failed: {hf_curl_err}")
                                        pass
                            else:
                                print("--> No word-level mirror available. Fallback failed.")
                                pass
        else:
            # Facebook FastText (Direct .vec download)
            try:
                print(f"--> Downloading Facebook FastText pre-trained embeddings from {url}...")
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(txt_path, 'wb') as out_file:
                    out_file.write(response.read())
                print("--> Download completed successfully!")
            except Exception as e:
                if os.path.exists(txt_path):
                    try: os.remove(txt_path)
                    except: pass
                print(f"--> Urllib download failed: {e}. Trying fallback with wget/curl...")
                try:
                    import subprocess
                    print("--> Attempting download via wget...")
                    subprocess.run(["wget", "-q", "-O", txt_path, url], check=True)
                    print("--> Download completed via wget!")
                except Exception as wget_err:
                    if os.path.exists(txt_path):
                        try: os.remove(txt_path)
                        except: pass
                    print(f"--> Wget failed: {wget_err}. Attempting download via curl...")
                    try:
                        import subprocess
                        subprocess.run(["curl", "-s", "-L", "-o", txt_path, url], check=True)
                        print("--> Download completed via curl!")
                    except Exception as curl_err:
                        if os.path.exists(txt_path):
                            try: os.remove(txt_path)
                            except: pass
                        print(f"--> FastText download failed completely: {curl_err}")
                        return None
        
        # Unzip if zip was downloaded
        if zip_path and os.path.exists(zip_path):
            print(f"--> Extracting {zip_name}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
                    if txt_files:
                        target_file = txt_files[0]
                        zip_ref.extract(target_file, data_dir)
                        extracted_path = os.path.join(data_dir, target_file)
                        if extracted_path != txt_path:
                            if os.path.exists(txt_path):
                                os.remove(txt_path)
                            os.rename(extracted_path, txt_path)
                    else:
                        zip_ref.extractall(data_dir)
                print("--> Extraction completed!")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            except Exception as e:
                print(f"--> Error extracting zip file: {e}")
                
        # 2.5 Verify if file exists and is valid size, otherwise try kagglehub fallback (only for PhoW2V)
        file_valid = False
        if os.path.exists(txt_path):
            if os.path.getsize(txt_path) > 10000000: # > 10 MB (should be ~1.18 GB / ~600MB)
                file_valid = True
            else:
                print("--> WARNING: Downloaded file is too small (possibly HTML/pointer page). Removing it.")
                try: os.remove(txt_path)
                except: pass
                
        if not file_valid and embedding_dim == 100:
            print("--> Trying fallback download via kagglehub...")
            try:
                import kagglehub
                import shutil
                # Determine correct dataset path based on syllable or word
                dataset_name = "thnhphong/word2vec-vi-syllables-100dims"
                if segment_words:
                    dataset_name = "thnhphong/word2vec-vi-words-100dims"
                
                print(f"--> Downloading from Kaggle via kagglehub: {dataset_name} ...")
                download_dir = kagglehub.dataset_download(dataset_name)
                print(f"--> kagglehub download completed at: {download_dir}")
                
                # Find any .txt file in the download directory
                txt_files = [f for f in os.listdir(download_dir) if f.endswith(".txt")]
                if txt_files:
                    src_txt_path = os.path.join(download_dir, txt_files[0])
                    shutil.copy(src_txt_path, txt_path)
                    print(f"--> Copied {txt_files[0]} to {txt_path}")
                else:
                    # Maybe it is zipped?
                    zip_files = [f for f in os.listdir(download_dir) if f.endswith(".zip")]
                    if zip_files:
                        src_zip_path = os.path.join(download_dir, zip_files[0])
                        with zipfile.ZipFile(src_zip_path, 'r') as zip_ref:
                            txt_files_zip = [f for f in zip_ref.namelist() if f.endswith('.txt')]
                            if txt_files_zip:
                                target_file = txt_files_zip[0]
                                zip_ref.extract(target_file, data_dir)
                                extracted_path = os.path.join(data_dir, target_file)
                                if extracted_path != txt_path:
                                    if os.path.exists(txt_path):
                                        os.remove(txt_path)
                                    os.rename(extracted_path, txt_path)
                            else:
                                zip_ref.extractall(data_dir)
                        print(f"--> Extracted zip file from kagglehub!")
            except Exception as kaggle_err:
                print(f"--> Kagglehub download failed: {kaggle_err}")
                return None

    # 3. Read txt file and load vectors
    print(f"--> Loading pre-trained vectors from {txt_path}...")
    pretrained_dict = {}
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if i == 0 and len(parts) <= 2:
                    continue
                word = parts[0]
                try:
                    vector = [float(x) for x in parts[1:]]
                    if len(vector) == embedding_dim:
                        pretrained_dict[word] = vector
                except ValueError:
                    continue
        print(f"--> Loaded {len(pretrained_dict)} pre-trained word vectors.")
    except Exception as e:
        print(f"--> Error reading embeddings file: {e}")
        return None
        
    # 4. Map to our vocabulary
    vocab_size = len(vocab_w2i)
    weight_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))
    weight_matrix[0] = np.zeros(embedding_dim)
    
    matched_count = 0
    for word, idx in vocab_w2i.items():
        if idx == 0:
            continue
        vec = pretrained_dict.get(word, None)
        if vec is None:
            vec = pretrained_dict.get(word.replace(" ", "_"), None)
            
        if vec is not None:
            weight_matrix[idx] = np.array(vec)
            matched_count += 1
            
    print(f"--> Vocab coverage: {matched_count}/{vocab_size} words ({matched_count/vocab_size:.2%}) initialized from pre-trained embeddings.")
    return torch.tensor(weight_matrix, dtype=torch.float32)

def train_model(model_type, epochs=5, batch_size=16, lr=1e-3, dropout=0.3, freeze_backbone=True, subset_size=None, save_dir="models", data_dir="data", oversample=True, use_class_weights=False, patience=5, resume=True, transformer_model_name="vinai/phobert-base", segment_words=None, embedding_dim=64, hidden_dim=64, use_pretrained_emb=False, additional_dataset="none", use_eda=False):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training {model_type.upper()} model on device: {device} ---")
    
    # Enforce mutual exclusivity of Oversampling and Class Weights
    if oversample and use_class_weights:
        print("--> WARNING: Both Oversample and UseClassWeights were set to True.")
        print("    To prevent double-favoring the minority class, UseClassWeights has been set to False.")
        use_class_weights = False
        
    # Enforce embedding_dim if use_pretrained_emb is active
    if model_type in ["lstm", "lstm_1d"] and use_pretrained_emb:
        if embedding_dim not in [100, 300]:
            embedding_dim = 100
        
    # Resolve segment_words actual value for printing
    resolved_segment_words = segment_words if segment_words is not None else ("phobert" in transformer_model_name.lower() if model_type == "transformer" else False)
    print(f"Params: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, Dropout={dropout}, Subset={subset_size}, Oversample={oversample}, UseClassWeights={use_class_weights}, Patience={patience}, SegmentWords={resolved_segment_words}, EmbeddingDim={embedding_dim}, HiddenDim={hidden_dim}, UsePretrainedEmb={use_pretrained_emb}, AdditionalDataset={additional_dataset}, UseEDA={use_eda}")
    
    # Get DataLoaders
    if model_type in ["lstm", "lstm_1d"]:
        train_loader, val_loader, test_loader, vocab = get_dataloaders(
            data_dir=data_dir, model_type=model_type, batch_size=batch_size, subset_size=subset_size, oversample=oversample, segment_words=segment_words, additional_dataset=additional_dataset, use_eda=use_eda
        )
        
        pretrained_weights = None
        if use_pretrained_emb:
            pretrained_weights = load_pretrained_embeddings(
                vocab.word2idx, segment_words=resolved_segment_words, data_dir=data_dir, embedding_dim=embedding_dim
            )
            if pretrained_weights is None:
                print("--> WARNING: Failed to load pre-trained embeddings. Falling back to random initialization.")
                use_pretrained_emb = False
                embedding_dim = 64  # Reset to default
                
        if model_type == "lstm":
            model = BiLSTMClassifier(
                vocab_size=vocab.vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else: # lstm_1d
            model = LSTMClassifier(
                vocab_size=vocab.vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        
        if pretrained_weights is not None:
            model.embedding.weight.data.copy_(pretrained_weights)
    else: # transformer
        train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
            data_dir=data_dir, model_type="transformer", batch_size=batch_size, subset_size=subset_size, oversample=oversample, tokenizer_name=transformer_model_name, segment_words=segment_words, additional_dataset=additional_dataset, use_eda=use_eda
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
        
    # Add weight decay (L2 regularization) for LSTM to reduce overfitting
    wd = 1e-4 if model_type in ["lstm", "lstm_1d"] else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
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
                    "embedding_dim": embedding_dim if model_type in ["lstm", "lstm_1d"] else None,
                    "hidden_dim": hidden_dim if model_type in ["lstm", "lstm_1d"] else None,
                    "use_pretrained_emb": use_pretrained_emb if model_type in ["lstm", "lstm_1d"] else None,
                    "freeze_backbone": freeze_backbone if model_type == "transformer" else None,
                    "transformer_model_name": transformer_model_name if model_type == "transformer" else None,
                    "additional_dataset": additional_dataset
                }
            }
            # For LSTM we must save vocab so we can encode words for inference
            if model_type in ["lstm", "lstm_1d"]:
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
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "lstm_1d", "transformer"], help="Model type to train")
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
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension for LSTM model")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for LSTM model")
    parser.add_argument("--use_pretrained_emb", action="store_true", help="Use pre-trained PhoW2V word embeddings for LSTM")
    parser.add_argument("--additional_dataset", type=str, default="none", choices=["none", "vfnd", "tingia", "both", "legacy"], help="Additional dataset to merge into train/val/test splits")
    parser.add_argument("--no_additional", action="store_true", help="Shortcut to disable additional dataset (equivalent to --additional_dataset none)")
    parser.add_argument("--use_eda", action="store_true", help="Load EDA-augmented training datasets instead of clean ones")
    
    args = parser.parse_args()
    
    # Overwrite additional_dataset with 'none' if --no_additional flag is passed
    if args.no_additional:
        args.additional_dataset = "none"
        
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
        segment_words=args.segment_words,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        use_pretrained_emb=args.use_pretrained_emb,
        additional_dataset=args.additional_dataset,
        use_eda=args.use_eda
    )
    
    if args.history_file:
        os.makedirs(os.path.dirname(args.history_file), exist_ok=True)
        with open(args.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {args.history_file}")


