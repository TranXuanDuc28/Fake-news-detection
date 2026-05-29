import os
import sys

# Set system output to UTF-8 to prevent encoding errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import re
import torch
from transformers import AutoTokenizer

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

# Clean Vietnamese text function identical to the one in src.data_loader
def clean_vietnamese_text(text, segment_words=False):
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
            print("Warning: pyvi is not installed, skipping word segmentation.")
            
    return text

# Vocab helper to encode texts for the LSTM model
class VocabHelper:
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.unk_idx = word2idx.get("<unk>", 1)
        self.pad_idx = word2idx.get("<pad>", 0)
        
    def encode(self, text, max_len=128, segment_words=False):
        cleaned_text = clean_vietnamese_text(text, segment_words=segment_words)
        tokens = cleaned_text.split()
        idxs = [self.word2idx.get(tok, self.unk_idx) for tok in tokens]
        if len(idxs) < max_len:
            idxs = idxs + [self.pad_idx] * (max_len - len(idxs))
        else:
            idxs = idxs[:max_len]
        return torch.tensor([idxs], dtype=torch.long)

# ANSI Escape Codes for nice colorized terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def main():
    parser = argparse.ArgumentParser(description="Predict whether a Vietnamese text is Real or Fake News using trained models.")
    parser.add_argument("--text", type=str, help="Single Vietnamese text to analyze. If not provided, the script runs in interactive mode.")
    parser.add_argument("--lstm_path", type=str, default="models/best_lstm.pt", help="Path to best_lstm.pt")
    parser.add_argument("--trans_path", type=str, default="models/best_transformer.pt", help="Path to best_transformer.pt")
    
    args = parser.parse_args()
    
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{CYAN}Đang chạy trên thiết bị: {BOLD}{device.type.upper()}{RESET}")
    
    # 2. Load LSTM Model
    lstm_model = None
    lstm_vocab = None
    lstm_segment_words = False
    if os.path.exists(args.lstm_path):
        try:
            print(f"Đang tải mô hình BiLSTM từ {args.lstm_path}...")
            checkpoint = torch.load(args.lstm_path, map_location=device)
            vocab_w2i = checkpoint["vocab_word2idx"]
            lstm_vocab = VocabHelper(vocab_w2i)
            
            lstm_segment_words = checkpoint.get("hyperparameters", {}).get("segment_words", False)
            
            from src.lstm_model import BiLSTMClassifier
            lstm_model = BiLSTMClassifier(
                vocab_size=len(vocab_w2i),
                embedding_dim=128,
                hidden_dim=128,
                dropout=checkpoint["hyperparameters"].get("dropout", 0.3)
            )
            lstm_model.load_state_dict(checkpoint["model_state_dict"])
            lstm_model.to(device)
            lstm_model.eval()
            print(f"{GREEN}✓ Tải mô hình BiLSTM thành công! (Tách từ: {lstm_segment_words}){RESET}")
        except Exception as e:
            print(f"{RED}✗ Lỗi tải mô hình BiLSTM: {e}{RESET}")
    else:
        print(f"{YELLOW}⚠ Không tìm thấy mô hình BiLSTM tại '{args.lstm_path}'. Sẽ bỏ qua dự đoán BiLSTM.{RESET}")

    # 3. Load Transformer Model
    trans_model = None
    trans_tokenizer = None
    trans_segment_words = False
    if os.path.exists(args.trans_path):
        try:
            print(f"Đang tải mô hình Transformer từ {args.trans_path}...")
            checkpoint = torch.load(args.trans_path, map_location=device)
            
            # Dynamic model name from checkpoint hyperparameters, fallback to distilbert if missing
            global trans_model_name
            trans_model_name = checkpoint.get("hyperparameters", {}).get("transformer_model_name", None)
            if not trans_model_name:
                trans_model_name = "distilbert-base-multilingual-cased"
                
            print(f"Đang sử dụng mô hình backbone: {trans_model_name}")
            trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
            
            trans_segment_words = checkpoint.get("hyperparameters", {}).get("segment_words", None)
            if trans_segment_words is None:
                trans_segment_words = ("phobert" in trans_model_name.lower())
                
            from src.transformer_model import TransformerClassifier
            trans_model = TransformerClassifier(
                model_name=trans_model_name,
                dropout=checkpoint["hyperparameters"].get("dropout", 0.3),
                freeze_backbone=True
            )
            trans_model.load_state_dict(checkpoint["model_state_dict"])
            trans_model.to(device)
            trans_model.eval()
            print(f"{GREEN}✓ Tải mô hình Transformer thành công! (Tách từ: {trans_segment_words}){RESET}")
        except Exception as e:
            print(f"{RED}✗ Lỗi tải mô hình Transformer: {e}{RESET}")
    else:
        print(f"{YELLOW}⚠ Không tìm thấy mô hình Transformer tại '{args.trans_path}'. Sẽ bỏ qua dự đoán Transformer.{RESET}")

    if lstm_model is None and trans_model is None:
        print(f"{RED}{BOLD}LỖI: Không tải được mô hình nào! Vui lòng kiểm tra lại đường dẫn trọng số.{RESET}")
        sys.exit(1)

    def run_prediction(text):
        if not text.strip():
            print(f"{YELLOW}⚠ Vui lòng nhập nội dung văn bản.{RESET}")
            return
            
        print("\n" + "="*60)
        print(f"{BOLD}Văn bản gốc:{RESET} {text[:150]}..." if len(text) > 150 else text)
        print("-"*60)
        
        # LSTM Prediction
        if lstm_model is not None:
            try:
                inputs = lstm_vocab.encode(text, max_len=128, segment_words=lstm_segment_words).to(device)
                with torch.no_grad():
                    logits = lstm_model(inputs)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    prob_fake = probs[1].item()
                    prob_real = probs[0].item()
                    
                label_str = f"{RED}{BOLD}TIN GIẢ (FAKE NEWS){RESET}" if prob_fake >= 0.5 else f"{GREEN}{BOLD}TIN THẬT (REAL NEWS){RESET}"
                confidence = prob_fake if prob_fake >= 0.5 else prob_real
                print(f"{BOLD}[BiLSTM Model]:{RESET} {label_str} | Độ tin cậy: {confidence*100:.2f}% (Fake: {prob_fake*100:.1f}%, Real: {prob_real*100:.1f}%)")
            except Exception as e:
                print(f"[BiLSTM Model]: Lỗi dự đoán - {e}")
                
        # Transformer Prediction
        if trans_model is not None:
            try:
                cleaned_text = clean_vietnamese_text(text, segment_words=trans_segment_words)
                inputs = trans_tokenizer(cleaned_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                with torch.no_grad():
                    logits = trans_model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    prob_fake = probs[1].item()
                    prob_real = probs[0].item()
                    
                label_str = f"{RED}{BOLD}TIN GIẢ (FAKE NEWS){RESET}" if prob_fake >= 0.5 else f"{GREEN}{BOLD}TIN THẬT (REAL NEWS){RESET}"
                confidence = prob_fake if prob_fake >= 0.5 else prob_real
                print(f"{BOLD}[Transformer]:{RESET}    {label_str} | Độ tin cậy: {confidence*100:.2f}% (Fake: {prob_fake*100:.1f}%, Real: {prob_real*100:.1f}%)")
            except Exception as e:
                print(f"[Transformer]: Lỗi dự đoán - {e}")
        print("="*60 + "\n")

    # 4. Run prediction
    if args.text:
        run_prediction(args.text)
    else:
        # Interactive mode
        print(f"\n{CYAN}{BOLD}=== HỆ THỐNG PHÂN TÍCH TIN GIẢ TIẾNG VIỆT ==={RESET}")
        print("Nhập hoặc dán nội dung bài viết vào bên dưới để kiểm định.")
        print(f"Gõ {BOLD}'exit'{RESET} hoặc {BOLD}'quit'{RESET} để thoát chương trình.\n")
        
        while True:
            try:
                user_input = input(f"{BOLD}Nhập văn bản cần kiểm định > {RESET}")
                if user_input.strip().lower() in ["exit", "quit"]:
                    print("Cảm ơn bạn đã sử dụng hệ thống!")
                    break
                run_prediction(user_input)
            except KeyboardInterrupt:
                print("\nThoát chương trình. Tạm biệt!")
                break
            except Exception as e:
                print(f"{RED}Lỗi nhập liệu: {e}{RESET}")

if __name__ == "__main__":
    main()
