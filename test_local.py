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
    },
    {
        "text": "Thủ tướng Chính phủ yêu cầu các bộ, ngành khẩn trương triển khai các biện pháp bình ổn thị trường vàng, không để xảy ra tình trạng đầu cơ, làm giá.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Cảnh báo: Bọn tội phạm đang dùng công nghệ AI giả giọng nói của người thân gọi điện xin tiền cứu cấp, chỉ cần nghe máy 10 giây là tài khoản bị tự động rút sạch.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Việt Nam chính thức đưa vào vận hành hệ thống đăng ký hộ tịch trực tuyến toàn quốc, giúp rút ngắn thời gian xử lý thủ tục hành chính cho người dân.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Chúc mừng bạn đã trúng một chiếc điện thoại iPhone 15 Pro Max từ chương trình tri ân của tập đoàn công nghệ Vingroup. Vui lòng bấm vào liên kết vin-tri-an.net để cung cấp thông tin nhận giải!",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Kỳ thi tốt nghiệp THPT năm nay sẽ được tổ chức trong hai ngày với đóng góp của bốn bài thi độc lập, đảm bảo quy chế phòng thi nghiêm ngặt trên cả nước.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Bí quyết trường thọ cực kỳ đơn giản: Chỉ cần ăn một quả chuối chín đắp lá lốt nướng vào mỗi buổi tối sẽ tiêu diệt hoàn toàn độc tố tích tụ trong gan cả đời.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Tổ chức Y tế Thế giới (WHO) khuyến nghị tăng cường giám sát các biến thể mới của virus SARS-CoV-2 nhằm chủ động ứng phó dịch bệnh.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Tin nóng Bình Dương: Phát hiện ổ dịch mới nguy hiểm với hơn 500 người nhiễm bệnh, chính quyền đang chuẩn bị phong tỏa toàn thành phố từ đêm nay.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Ngân hàng Thương mại Cổ phần Sài Gòn (SCB) công bố biểu phí dịch vụ tài khoản mới áp dụng cho khách hàng cá nhân kể từ đầu tháng sau.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Cảnh báo khẩn: Tập đoàn Vạn Thịnh Phát câu kết với ngân hàng để chiếm đoạt toàn bộ sổ tiết kiệm của khách hàng gửi tiền, người dân hãy đến rút sạch tiền ngay lập tức!",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Sở Giáo dục và Đào tạo Hà Nội thông báo lịch nghỉ Tết Nguyên đán chính thức của học sinh các cấp trên địa bàn thành phố.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Ăn lá đu đủ đun sôi cùng sả trong 10 ngày sẽ làm tiêu biến hoàn toàn khối u ung thư cổ tử cung, bài thuốc dân gian thần kỳ không cần hóa trị.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Bộ Tài chính đề xuất giảm thuế giá trị gia tăng (VAT) từ 10% xuống 8% đối với một số nhóm hàng hóa, dịch vụ trong năm nay.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Chấn động: Một phụ nữ tại Hải Phòng sinh ra em bé có cánh và biết bay ngay khi vừa chào đời, các nhà khoa học đang đến nghiên cứu trực tiếp.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Thủ tướng Phạm Minh Chính chủ trì cuộc họp thường kỳ chính phủ đánh giá tình hình kinh tế - xã hội tháng qua và bàn giải pháp thúc đẩy đầu tư công.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Cảnh báo lừa đảo: Cuộc gọi giả danh cơ quan công an yêu cầu chuyển tiền vào tài khoản tạm giữ để phục vụ điều tra chuyên án ma túy.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Thần y tiết lộ: Chỉ cần nhỏ 3 giọt nước cốt tỏi vào mắt mỗi tối sẽ trị dứt điểm mọi bệnh cận thị, loạn thị mà không cần phẫu thuật.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Ủy ban Thường vụ Quốc hội cho ý kiến về dự án Luật Bảo hiểm xã hội sửa đổi, tập trung vào các chế độ rút bảo hiểm một lần.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Nhật Bản phát minh ra loại vaccine mới có thể ngăn ngừa lão hóa và giúp con người trẻ mãi không già, dự kiến bán ra thị trường năm sau.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Tổng cục Thống kê công báo chỉ số giá tiêu dùng (CPI) tăng nhẹ so với cùng kỳ năm trước, lạm phát vẫn trong tầm kiểm soát.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Tin giật gân: Trùm phát xít Hitler vẫn còn sống ở tuổi 135 tại một căn hầm bí mật ở Nam Cực và đang chuẩn bị lực lượng quay trở lại thế giới.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Công ty Điện lực Việt Nam (EVN) khuyến nghị người dân tiết kiệm điện trong những ngày nắng nóng đỉnh điểm để tránh quá tải hệ thống.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Phương pháp giảm cân thần tốc: Chỉ cần uống hỗn hợp giấm ăn và muối trắng thay nước lọc mỗi ngày sẽ giảm ngay 15kg trong một tuần.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Bộ Công an triệt phá đường dây đánh bạc qua mạng với quy mô giao dịch lên đến hàng nghìn tỷ đồng, bắt giữ hàng chục đối tượng liên quan.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Tập đoàn công nghệ Apple chính thức ra mắt dòng sản phẩm kính thực tế ảo mới với nhiều cải tiến công nghệ đột phá tại Mỹ.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Bí mật vũ trụ: Các nhà du hành vũ trụ phát hiện căn cứ của người ngoài hành tinh trên vùng tối của Mặt Trăng nhưng bị Mỹ ép ký cam kết bảo mật.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Việt Nam xuất khẩu lô xoài cát Hòa Lộc đầu tiên sang thị trường Hoa Kỳ sau khi vượt qua các tiêu chuẩn kiểm dịch thực vật nghiêm ngặt.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Khẩn cấp: Sóng thần cao 50 mét đang hướng thẳng vào các tỉnh ven biển miền Trung Việt Nam sau trận động đất mạnh ở Thái Bình Dương.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Bộ Y tế cấp phép lưu hành cho vắc xin phòng bệnh sốt xuất huyết đầu tiên tại Việt Nam, bắt đầu tiêm chủng dịch vụ cho người dân.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Uống nước tiểu của chính mình mỗi sáng là phương pháp thanh lọc cơ thể, thải sạch mọi độc tố và ngăn ngừa hoàn toàn bệnh tiểu đường.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Nhiều địa phương trên cả nước tổ chức lễ ra quân hưởng ứng Ngày Môi trường Thế giới, tiến hành dọn rác và trồng cây xanh.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Mạng di động 6G chính thức phủ sóng toàn cầu từ hôm nay, cho phép truyền tải dữ liệu nhanh gấp 1000 lần mạng 5G thông thường.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Vận động viên Nguyễn Thị Oanh xuất sắc giành huy chương vàng cự ly 3.000m vượt chướng ngại vật tại giải điền kinh vô địch châu Á.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Xem bói nốt ruồi trực tuyến đoán trước tương lai giàu sang hay nghèo khổ cực chuẩn, bấm vào link này để được thầy phán miễn phí.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Hội đồng Bảo an Liên Hợp Quốc thông qua nghị quyết kêu gọi các bên ngừng bắn ngay lập tức và tăng cường viện trợ nhân đạo tại khu vực xung đột.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Cảnh báo khẩn: Trái Đất sẽ mất hoàn toàn trọng lực trong vòng 2 giờ vào ngày mai do ảnh hưởng từ bão mặt trời siêu mạnh.",
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
        
        hyperparams = checkpoint.get("hyperparameters", {})
        segment_words = hyperparams.get("segment_words", False)
        embedding_dim = hyperparams.get("embedding_dim", 128)
        hidden_dim = hyperparams.get("hidden_dim", 128)
        if embedding_dim is None: embedding_dim = 128
        if hidden_dim is None: hidden_dim = 128
        
        from src.lstm_model import BiLSTMClassifier
        model = BiLSTMClassifier(
            vocab_size=len(vocab_w2i),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=hyperparams.get("dropout", 0.3)
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

def run_local_test(lstm_path="models/best_lstm.pt", trans_path="models/best_transformer.pt", lstm_threshold=0.63, trans_threshold=0.50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{CYAN}{BOLD}=== HỆ THỐNG KIỂM THỬ MÔ HÌNH TIN GIẢ TRÊN LOCAL ==={RESET}")
    print(f"Thiết bị chạy: {BOLD}{device.type.upper()}{RESET}")
    print(f"Trạng thái PyVi (Word Segmentation): {BOLD}{'Đã cài đặt' if HAS_PYVI else 'Chưa cài đặt'}{RESET}\n")

    # If the user specified paths don't exist, search for fallbacks
    if lstm_path and not os.path.exists(lstm_path):
        fallback_paths = ["models/best_lstm.pt"]
        lstm_path = next((p for p in fallback_paths if os.path.exists(p)), None)
        
    if trans_path and not os.path.exists(trans_path):
        fallback_paths = ["models/best_transformer.pt", "models/best_transformer_phobert.pt", "models/best_transformer_distilbert.pt"]
        trans_path = next((p for p in fallback_paths if os.path.exists(p)), None)
    
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
                
                pred_label = f"{RED}TIN GIẢ (FAKE){RESET}" if prob_fake >= lstm_threshold else f"{GREEN}TIN THẬT (REAL){RESET}"
                confidence = prob_fake if prob_fake >= lstm_threshold else prob_real
                print(f"  └─ {BOLD}[BiLSTM]:{RESET}      {pred_label} | Ngưỡng: {lstm_threshold:.2f} | Độ tin cậy: {confidence*100:.2f}% (Fake: {prob_fake*100:.1f}%, Real: {prob_real*100:.1f}%)")
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
                
                pred_label = f"{RED}TIN GIẢ (FAKE){RESET}" if prob_fake >= trans_threshold else f"{GREEN}TIN THẬT (REAL){RESET}"
                confidence = prob_fake if prob_fake >= trans_threshold else prob_real
                print(f"  └─ {BOLD}[Transformer]:{RESET} {pred_label} | Ngưỡng: {trans_threshold:.2f} | Độ tin cậy: {confidence*100:.2f}% (Fake: {prob_fake*100:.1f}%, Real: {prob_real*100:.1f}%)")
            except Exception as e:
                print(f"  └─ [Transformer]: Lỗi dự đoán - {e}")
                
        print("-" * 80)

    # === CHẾ ĐỘ TƯƠNG TÁC TỪ NGƯỜI DÙNG ===
    print("\n" + "="*80)
    print(f"{CYAN}{BOLD}CHẾ ĐỘ TƯƠNG TÁC: NHẬP VĂN BẢN ĐỂ DỰ ĐOÁN TIN GIẢ{RESET}")
    print("="*80)
    print("Bạn có thể tự nhập bất kỳ tin tức nào dưới đây để kiểm tra dự đoán.")
    print("Nhập 'exit' hoặc 'q' để thoát chế độ tương tác.\n")
    
    while True:
        try:
            print(f"{BOLD}Nhập tin tức cần kiểm tra:{RESET}")
            user_text = input("> ").strip()
            if not user_text:
                continue
            if user_text.lower() in ["exit", "q", "quit"]:
                print(f"{YELLOW}Đã thoát chế độ tương tác.{RESET}")
                break
                
            print("-" * 50)
            
            # 1. Dự đoán bằng BiLSTM
            if lstm_model:
                try:
                    inputs = lstm_vocab.encode(user_text, max_len=128, segment_words=lstm_segment).to(device)
                    with torch.no_grad():
                        logits = lstm_model(inputs)
                        probs = torch.softmax(logits, dim=1).squeeze(0)
                        prob_fake = probs[1].item()
                        prob_real = probs[0].item()
                    
                    pred_label = f"{RED}TIN GIẢ (FAKE){RESET}" if prob_fake >= lstm_threshold else f"{GREEN}TIN THẬT (REAL){RESET}"
                    confidence = prob_fake if prob_fake >= lstm_threshold else prob_real
                    print(f"  └─ {BOLD}[BiLSTM]:{RESET}      {pred_label} | Ngưỡng: {lstm_threshold:.2f} | Độ tin cậy: {confidence*100:.2f}% (Fake: {prob_fake*100:.1f}%, Real: {prob_real*100:.1f}%)")
                except Exception as e:
                    print(f"  └─ [BiLSTM]: Lỗi dự đoán - {e}")
                    
            # 2. Dự đoán bằng Transformer
            if trans_model:
                try:
                    cleaned_text = clean_vietnamese_text(user_text, segment_words=trans_segment)
                    inputs = trans_tokenizer(cleaned_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    with torch.no_grad():
                        logits = trans_model(input_ids=input_ids, attention_mask=attention_mask)
                        probs = torch.softmax(logits, dim=1).squeeze(0)
                        prob_fake = probs[1].item()
                        prob_real = probs[0].item()
                    
                    pred_label = f"{RED}TIN GIẢ (FAKE){RESET}" if prob_fake >= trans_threshold else f"{GREEN}TIN THẬT (REAL){RESET}"
                    confidence = prob_fake if prob_fake >= trans_threshold else prob_real
                    print(f"  └─ {BOLD}[Transformer]:{RESET} {pred_label} | Ngưỡng: {trans_threshold:.2f} | Độ tin cậy: {confidence*100:.2f}% (Fake: {prob_fake*100:.1f}%, Real: {prob_real*100:.1f}%)")
                except Exception as e:
                    print(f"  └─ [Transformer]: Lỗi dự đoán - {e}")
                    
            print("-" * 50 + "\n")
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Đã thoát chế độ tương tác.{RESET}")
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run local benchmarks on model weights.")
    parser.add_argument("--lstm_path", type=str, default="models/best_lstm.pt", help="Path to best_lstm.pt")
    parser.add_argument("--trans_path", type=str, default="models/best_transformer.pt", help="Path to best_transformer.pt")
    parser.add_argument("--lstm_threshold", type=float, default=0.63, help="Decision threshold for LSTM (default 0.63)")
    parser.add_argument("--trans_threshold", type=float, default=0.50, help="Decision threshold for Transformer (default 0.50)")
    args = parser.parse_args()
    
    run_local_test(
        lstm_path=args.lstm_path,
        trans_path=args.trans_path,
        lstm_threshold=args.lstm_threshold,
        trans_threshold=args.trans_threshold
    )
