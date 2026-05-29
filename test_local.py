import os
import sys
import torch
from transformers import AutoTokenizer

# Set system output to UTF-8 to prevent encoding errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

# Clean Vietnamese text function identical to the one in src.data_loader
def clean_vietnamese_text(text, segment_words=False):
    import re
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s,\.\?\!\-\:\#]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if segment_words:
        if HAS_PYVI:
            text = ViTokenizer.tokenize(text)
        else:
            print("Warning: pyvi is not installed, skipping word segmentation.")
            
    return text

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

# ANSI Escape Codes for colorized terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Define test cases
TEST_CASES = [
    {
        "text": "Bộ Y tế khuyến cáo người dân đeo khẩu trang và rửa tay sát khuẩn thường xuyên để phòng ngừa dịch bệnh lây lan trong cộng đồng.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "TIN NÓNG KHẨN CẤP: Một loài vi-rút mới cực kỳ nguy hiểm đang lây lan qua không khí làm chết người chỉ trong 5 phút. Hãy chia sẻ gấp thông tin này để cứu sống người thân của bạn!",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Trường Đại học Bách khoa Hà Nội công bố phương án tuyển sinh năm học mới, tăng chỉ tiêu xét tuyển bằng điểm thi tốt nghiệp THPT.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Bí mật chấn động thế giới: Các nhà khoa học đã tìm ra phương pháp chữa khỏi hoàn toàn bệnh ung thư giai đoạn cuối chỉ bằng cách uống nước chanh ấm pha muối mỗi sáng.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Ủy ban Nhân dân Thành phố Hồ Chí Minh đề xuất mở rộng tuyến đường vành đai 2 nhằm giảm thiểu ùn tắc giao thông tại các cửa ngõ thành phố.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Đoàn thể thao Việt Nam đã giành thêm 2 huy chương vàng tại kỳ SEA Games, vươn lên vị trí thứ ba trên bảng tổng sắp huy chương.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Từ ngày 1/7, tăng mức lương cơ sở đối với cán bộ, công chức, viên chức và lực lượng vũ trang lên 1,8 triệu đồng/tháng.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Ngân hàng Nhà nước Việt Nam quyết định giảm các mức lãi suất điều hành nhằm hỗ trợ doanh nghiệp và thúc đẩy tăng trưởng kinh tế.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Nhiều trường tiểu học tại Hà Nội tổ chức ngày hội sách nhằm lan tỏa văn hóa đọc và khuyến khích học sinh tìm tòi tri thức mới.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Cảnh báo khẩn cấp từ công an: Tuyệt đối không nghe cuộc gọi từ các đầu số lạ này kẻo bị hack toàn bộ tài khoản ngân hàng và mất sạch tiền trong 3 giây!",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Ăn tỏi sống ngâm giấm mỗi ngày sẽ giúp tiêu diệt hoàn toàn tế bào ung thư vú và ung thư phổi, bác sĩ giấu kín vì sợ mất khách!",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Rúng động: Người đàn ông ở miền Tây bắt được sinh vật lạ giống hệt rồng thần trong truyền thuyết, ai xem cũng phải chia sẻ để lấy may mắn.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Hãng hàng không quốc gia tặng 2 vé máy bay miễn phí cho tất cả khách hàng kỷ niệm 50 năm thành lập, nhấn vào link này để nhận ngay quà tặng!",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Mẹ thiên nhiên trừng phạt: Trái Đất chuẩn bị bước vào 3 ngày tối tăm liên tục do hiện tượng thiên văn kỳ lạ, mọi người cần tích trữ lương thực ngay!",
        "label": "TIN GIẢ (FAKE)"
    }
]

def load_lstm_model(model_path, device):
    if not os.path.exists(model_path):
        return None, None, False
    try:
        checkpoint = torch.load(model_path, map_location=device)
        vocab_w2i = checkpoint["vocab_word2idx"]
        vocab = VocabHelper(vocab_w2i)
        
        segment_words = checkpoint.get("hyperparameters", {}).get("segment_words", False)
        
        from src.lstm_model import BiLSTMClassifier
        model = BiLSTMClassifier(
            vocab_size=len(vocab_w2i),
            embedding_dim=128,
            hidden_dim=128,
            dropout=checkpoint["hyperparameters"].get("dropout", 0.3)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, vocab, segment_words
    except Exception as e:
        print(f"{RED}Lỗi tải mô hình BiLSTM từ {model_path}: {e}{RESET}")
        return None, None, False

def load_trans_model(model_path, device):
    if not os.path.exists(model_path):
        return None, None, None, False
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        trans_model_name = checkpoint.get("hyperparameters", {}).get("transformer_model_name", None)
        if not trans_model_name:
            trans_model_name = "distilbert-base-multilingual-cased"
            
        tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
        
        segment_words = checkpoint.get("hyperparameters", {}).get("segment_words", None)
        if segment_words is None:
            segment_words = ("phobert" in trans_model_name.lower())
            
        from src.transformer_model import TransformerClassifier
        model = TransformerClassifier(
            model_name=trans_model_name,
            dropout=checkpoint["hyperparameters"].get("dropout", 0.3),
            freeze_backbone=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, tokenizer, trans_model_name, segment_words
    except Exception as e:
        print(f"{RED}Lỗi tải mô hình Transformer từ {model_path}: {e}{RESET}")
        return None, None, None, False

def run_local_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{CYAN}{BOLD}=== HỆ THỐNG KIỂM THỬ MÔ HÌNH TIN GIẢ TRÊN LOCAL ==={RESET}")
    print(f"Thiết bị chạy: {BOLD}{device.type.upper()}{RESET}")
    print(f"Trạng thái PyVi (Word Segmentation): {BOLD}{'Đã cài đặt' if HAS_PYVI else 'Chưa cài đặt'}{RESET}\n")

    # 1. Tìm các file trọng số khả dụng
    lstm_paths = ["models/best_lstm.pt"]
    trans_paths = ["models/best_transformer.pt", "models/best_transformer_phobert.pt", "models/best_transformer_distilbert.pt"]
    
    lstm_path = next((p for p in lstm_paths if os.path.exists(p)), None)
    trans_path = next((p for p in trans_paths if os.path.exists(p)), None)
    
    # Load LSTM
    lstm_model, lstm_vocab, lstm_segment = None, None, False
    if lstm_path:
        print(f"Đang tải BiLSTM từ: {BOLD}{lstm_path}{RESET}...")
        lstm_model, lstm_vocab, lstm_segment = load_lstm_model(lstm_path, device)
        if lstm_model:
            print(f"{GREEN}✓ Tải BiLSTM thành công! (Tách từ: {lstm_segment}){RESET}")
    else:
        print(f"{YELLOW}⚠ Không tìm thấy file trọng số BiLSTM (best_lstm.pt){RESET}")
        
    # Load Transformer
    trans_model, trans_tokenizer, trans_name, trans_segment = None, None, None, False
    if trans_path:
        print(f"Đang tải Transformer từ: {BOLD}{trans_path}{RESET}...")
        trans_model, trans_tokenizer, trans_name, trans_segment = load_trans_model(trans_path, device)
        if trans_model:
            print(f"{GREEN}✓ Tải Transformer ({trans_name}) thành công! (Tách từ: {trans_segment}){RESET}")
    else:
        print(f"{YELLOW}⚠ Không tìm thấy file trọng số Transformer (best_transformer.pt / best_transformer_phobert.pt / best_transformer_distilbert.pt){RESET}")

    if not lstm_model and not trans_model:
        print(f"\n{RED}{BOLD}LỖI: Không tìm thấy bất kỳ file trọng số mô hình nào trong thư mục 'models/'!{RESET}")
        print("Vui lòng huấn luyện mô hình hoặc copy tệp trọng số vào thư mục 'models/' trước.")
        return

    print("\n" + "="*80)
    print(f"{BOLD}BẮT ĐẦU CHẠY THỬ NGHIỆM TRÊN CÁC KỊCH BẢN VĂN BẢN MẪU:{RESET}")
    print("="*80)

    for idx, case in enumerate(TEST_CASES, 1):
        text = case["text"]
        expected = case["label"]
        print(f"\n{BOLD}Câu mẫu {idx}:{RESET} {text}")
        print(f"{CYAN}Nhãn thực tế mong đợi:{RESET} {expected}")
        print("-" * 50)
        
        # 1. Dự đoán bằng BiLSTM
        if lstm_model:
            try:
                inputs = lstm_vocab.encode(text, max_len=128, segment_words=lstm_segment).to(device)
                with torch.no_grad():
                    logits = lstm_model(inputs)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    prob_fake = probs[1].item()
                    prob_real = probs[0].item()
                
                pred_label = f"{RED}TIN GIẢ (FAKE){RESET}" if prob_fake >= 0.5 else f"{GREEN}TIN THẬT (REAL){RESET}"
                confidence = prob_fake if prob_fake >= 0.5 else prob_real
                print(f"  └─ {BOLD}[BiLSTM]:{RESET}      {pred_label} | Độ tin cậy: {confidence*100:.2f}% (Fake: {prob_fake*100:.1f}%, Real: {prob_real*100:.1f}%)")
            except Exception as e:
                print(f"  └─ [BiLSTM]: Lỗi dự đoán - {e}")
                
        # 2. Dự đoán bằng Transformer
        if trans_model:
            try:
                cleaned_text = clean_vietnamese_text(text, segment_words=trans_segment)
                inputs = trans_tokenizer(cleaned_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                with torch.no_grad():
                    logits = trans_model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    prob_fake = probs[1].item()
                    prob_real = probs[0].item()
                
                pred_label = f"{RED}TIN GIẢ (FAKE){RESET}" if prob_fake >= 0.5 else f"{GREEN}TIN THẬT (REAL){RESET}"
                confidence = prob_fake if prob_fake >= 0.5 else prob_real
                print(f"  └─ {BOLD}[Transformer]:{RESET} {pred_label} | Độ tin cậy: {confidence*100:.2f}% (Fake: {prob_fake*100:.1f}%, Real: {prob_real*100:.1f}%)")
            except Exception as e:
                print(f"  └─ [Transformer]: Lỗi dự đoán - {e}")
                
        print("-" * 80)

if __name__ == "__main__":
    run_local_test()
