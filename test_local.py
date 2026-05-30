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
        "text": "Chúng_tôi bất_ngờ khi nhìn thấy ngôi nhà khá to , đẹp của một gia_đình có số hộ cận nghèo .Được biết Chủ_hộ là chị_gái bí_thư chi_bộ thôn",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Kỳ thị cũng là một loại virus, và sức mạnh của nó tương tự SARS-CoV-2: đẩy con người ra xa nhau hơn.  Người mắc COVID-19 rồi cũng sẽ được chữa khỏi, nhưng người mắc tâm lý kỳ thị sẽ chỉ khỏi bệnh khi chính mình trở thành mục tiêu của những người khác.  #COVID19 #DungKyThi #Diemtuan #BtvVietHoang",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Không_thể trồng_trọt , thả cá , nhà thì ngập mỗi khi trời mưa .Mùi phân gà , phân vịt và nước_thải lưu_cữu nhiều năm .Người_dân phải chờ đến bao_giờ nữa cán_bộ ơi ?",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Ngài Thẩm phán này mà giữ chức Chánh cao cao thì công an các trại tù chắc thất nghiệp dài dài vì xử xong là phạm nhân tự chết , có vô tù nữa đâu mà cần người giữ ?",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Người trong ảnh này mà công_an Cầu_Giấy đưa lên báo là ông Phương ( hay Thiện ) sinh năm 1962 , anh ruột của ông Phiến , sinh năm 1966 .Ông Thiện tóc bạc , ông Phiến tóc còn đen .",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "FPT Online cắt giảm 42,5% chi phí nhân công, có đồng nghiệp nào bên Vnexpress, Ngoisao và Ione bị giảm lương không ạ?",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "COV-19 về đến hồng_lĩnh rồi bà_con ơi , Hiện đang cách_ly mười mấy người ở trường lái hồng_lĩnh .Bà_con hạn_chế đi_lại nhé",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Tuổi_thơ chân_đất , đầu trần ...Ai cũng có thời như_vậy , nhưng khi lớn lên mỗi người lại có 1 con đường riêng .Những đón_nhận của ngày hôm_nay chính là cái đích trên 1 ngã rẽ nào đó mà chúng_ta đã từng lựa_chọn .< URL >",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Chung cư Hòa Bình với hơn 1.000 người dân ở <URL> được cơ quan chức năng cách ly sau khi phát hiện ca nhiễm virus SARS CoV-2 thứ 48 sống tại đây.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "GS đoạt Nobel Y học Pháp: SARS-CoV-2 là vaccine chống HIV từ phòng thí nghiệm Vũ Hán.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "WARNING May_mắn cho những_ai đã kịp xin visa du_học Mỹ tuần vừa_rồi .Mỹ đã chính_thức ngưng cấp visa cho du_học_sinh Việt_Nam và các nước ĐNA khác - vô_thời_hạn .Các em du_học_sinh xác_định rồi nhé !Học trong nước thì không học , đi du_học làm gì .Có đâu bằng VN .",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Tỷ phú Trung Quốc Quách Văn Qúy nói: 1.200 Thi thể bị thiêu mỗi ngày Hơn 250 triệu người bị cách ly... 1,5 triệu người được xác nhận đã nhiễm virus Corona, Covid19 Không phải 20.000, hoặc 30.000 mà hơn 50.000 người đã chết do bệnh dịch.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Thêm 45.000 ca mới 1 ngày, Mỹ vẫn căng nhể. Nhiều bang như California, Texas, Florida đều ghi nhận số ca mắc Covid-19 mới cao kỷ lục.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Sở GTVT < URL > đề_xuất vẫn cho xe taxi , xe taxi công_nghệ hoạt_động bình_thường , hạn_chế phương_tiện công_cộng .",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Nếu còn được có công ăn việc làm tử tế, vẫn được nhận khoản lương định kì mỗi tháng dù công ty đang chao đảo - hãy thực sự biết ơn vì bạn đã và đang may mắn hơn hàng triệu người ngoài kia...  #KenhDoiSong",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Ai chưa xem tin tức thì cập nhật ngay nhé. GS người Pháp Raoult - giám đốc Viện nghiên cứu Bệnh viện đại học (IUH) chuyên bệnh truyền nhiễm Địa Trung Hải ở Marseille - đã khẳng định dịch sẽ hết trong 3 tuần tới tại Pháp và 6 tuần trên thế giới.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Đúng là quả báo không chừa một ai! Kẻ tắc trách đã bị trừng trị. Thông báo với các bậc phụ huynh, tên lái xe trường Gateway Doãn Quý Phiến đã chính thức tử nạn - đền tội cho những gì hắn đã gây ra. Tôi còn đang chờ xem chuyện gì sẽ xảy ra với những kẻ liên quan!",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Bã dương_tính rồi bà_con ... anh người_yêu và bao nhiu người đã từng tiếp_xúc chơi tết ... gia_đình bã ... thôi rồi ... tiêu rồi ..Làm_sao kiểm_soát đc những ng từng tiếp_xúc với mụ đó",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Sau ATM gạo, ATM sách, liệu còn ATM gì nhỉ?  [<URL>](<URL>)",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Tăng mạnh nhất thế giới chứng khoán Việt Nam  Hay hỏng và lỗi lúc nhạy cảm nhất là chứng khoán Việt Nam!  Thua lỗ đau đớn phút 91 là nhà đầu tư... đề nghị ubck làm rõ phiên giao dịch ngày hôm nay  Không ngoại trừ thao túng",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Ở Bệnh Viện CuBa mình vừa có 3 ca nhiễm vi khuẩn ăn thịt người rồi mn ơi... Cảnh giác nhé",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "CHÚNG TA ĐỪNG QUÁ BUỒN: BÁO CHÍ FAKE NEWS NHƯ CNN CŨNG KHỐN NẠN LẮM, TẤN CÔNG SẾP TESLA VÔ LỐI VÌ MÁY TRỢ THỞ, SAU KHI SAI RỒI CŨNG CHẲNG THÈM XIN LỖI. Thế giới không điên mới là lạ? (có thể dùng google dịch ra tiếng Việt nếu không đọc được tiếng Anh)",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "\" Còn bạn nào khẩu nghiệp , có dám tới bưu điện Cầu Voi ban_đêm rồi thề với hai cô gái coi có bị vặn cổ không ! \" Đã có 1 anh bị rồi đấy",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Bannon thẳng_thừng chỉ_trích Trung_Quốc và các \" đồng_minh \" như Kissinger , Bill_Gates khiến truyền_thông Trung_Quốc \" gây_chiến \" với ông !",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Lạm_phát hạ nhiệt sẽ tạo điều_kiện để Ngân_hàng Nhà_nước giảm tiếp lãi_suất điều_hành trong nửa cuối năm , theo VnDirect .",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Một Bí thư Thành ủy như ông Trương Quang Nghĩa, một bộ trưởng, như ông Nguyễn Chí Dũng tự cách ly đủ 14 ngày như các công dân khác, dù âm tính - tự nó chứng minh sự bình đẳng. Và sự công khai ấy cũng là cách duy nhất tránh tin đồn.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "năm 1908, khi đang còn là học sinh tại Huế, thanh niên Nguyễn Tất Thành đã tham gia biểu tình đòi thực dân Pháp phải giảm sưu cao thuế nặng cho nhân dân Việt Nam và anh đã bị quân Pháp bắn thương ở tay",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "\"Có bằng chứng to lớn cho thấy đó là nơi nó (virus) xuất phát. Tôi nghĩ cả thế giới giờ có thể thấy rõ, Trung Quốc có lịch sử lây bệnh cho thế giới, và họ vận hành những phòng thí nghiệm không đạt chuẩn\", AFP dẫn lời Ngoại trưởng Pompeo nói hôm 3/5 trong chương trình This Week của đài ABC.",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Sáng 21/6 thủ tướng Phúc và Đại biểu quốc hội Hải Phòng tiếp xúc lãnh đạo tp Hoa phượng , nghe lãnh đạo Hoa Phượng nói “ Hải Phòng luôn quan tâm đến đời sống mọi tầng lớp nhân dân” cơ mà?!",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Trước thông_tin nói rằng , Tập_đoàn Hoành_Sơn cấp_nước cho Formosa nhưng không trả một đồng phí nào , ông Phạm_Hoành_Sơn , Tổng_Giám_đốc Công_ty Cổ_phần Tập_đoàn Hoành_Sơn khẳng_định rằng thông_tin trên là không đúng .",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "[CẬP NHẬT Virus Corona Vũ Hán] TS Harvard: Con virus này cực kỳ nghiêm trọng \"Nó tệ hại đến mức nào? Đây là hệ số phá hoại tương đương một thảm họa bom nhiệt hạch. Cả sự nghiệp nghiên cứu của tôi chưa từng gặp một hệ số thực tế lớn như vậy. Tôi không hề phóng đại đâu…”",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Nhưng lạ một điều là việc nhầm chỉ có tăng, chưa thấy trường hợp nào tiền điện giảm....  Quá nhiều sai trái...😡😡😡",
        "label": "TIN GIẢ (FAKE)"
    },
    {
        "text": "Một loạt tin mừng, trong số 60 bệnh nhân mắc COVID-19 đang được điều trị tại các cơ sở y tế nhiều bệnh nhân có tình trạng sức khoẻ tốt, chức năng sống được kiểm soát.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Việc chuyển đổi từ đất lúa kém hiệu quả sang trồng cây ăn trái ở Bình Thuận đã giúp đồng bào có thu nhập cao, ổn định cuộc sống.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Cảnh_sát Pháp đã biểu_tình hôm 12.6 qua trung_tâm Paris để phản_đối lệnh cấm sử_dụng đòn kẹp cổ và giới_hạn những việc được thực_hiện trong khi bắt_giữ nghi phạm .",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Bộ Tư pháp Mỹ đâm đơn kiện một công ty Trung Quốc vì đã bán gần nửa triệu khẩu trang N95 giả mạo và không đạt tiêu chuẩn cho khách hàng ở Mỹ hồi tháng 4, giữa lúc đại dịch Covid-19 hoành hành nước này. #thegioi #khẩutranggiả #N95 #TrungQuốc #Mỹ",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Lấy ý tưởng từ bệnh viện dã chiến, nhóm mạnh thường quân trẻ ở quận Bình Thạnh đã lập ra một chương trình từ thiện độc đáo có tên gọi “quán cơm dã chiến, trái tim yêu thương”.",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Ngày 20-5 , chính_quyền xã An_Bình , huyện Kiến_Xương , tỉnh Thái_Bình đã phải tổ_chức lại đại_hội Đảng_bộ nhiệm_kỳ 2020-2025 do có phiếu bầu gian_lận làm sai_lệch kết_quả bầu Ban_chấp_hành Đảng_bộ xã trong chương_trình đại_hội trước đó .",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "Mong mọi người hãy bình tĩnh trước dịch bệnh. Biết chắt lọc tin tức đừng quá tin vào những thông tin không chính thống, cuối cùng lại tiền mất tật mang",
        "label": "TIN THẬT (REAL)"
    },
    {
        "text": "[ B ] ( < URL > ) ịt cổng chính cũ , xây cổng sở mới lui về phía trong",
        "label": "TIN THẬT (REAL)"
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
        
        model_type = hyperparams.get("model_type", "lstm")
        
        if model_type == "lstm_1d":
            from src.lstm_model import LSTMClassifier
            model = LSTMClassifier(
                vocab_size=len(vocab_w2i),
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=hyperparams.get("dropout", 0.3)
            )
        else:
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
        print(f"{RED}Lỗi tải mô hình LSTM từ {model_path}: {e}{RESET}")
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

def run_local_test(lstm_path="models/best_lstm.pt", lstm_1d_path="models/best_lstm_1d.pt", trans_path="models/best_transformer.pt", lstm_threshold=0.63, lstm_1d_threshold=0.58, trans_threshold=0.50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{CYAN}{BOLD}=== HỆ THỐNG KIỂM THỬ MÔ HÌNH TIN GIẢ TRÊN LOCAL ==={RESET}")
    print(f"Thiết bị chạy: {BOLD}{device.type.upper()}{RESET}")
    print(f"Trạng thái PyVi (Word Segmentation): {BOLD}{'Đã cài đặt' if HAS_PYVI else 'Chưa cài đặt'}{RESET}\n")

    # If the user specified paths don't exist, search for fallbacks
    if lstm_path and not os.path.exists(lstm_path):
        fallback_paths = ["models/best_lstm.pt"]
        lstm_path = next((p for p in fallback_paths if os.path.exists(p)), None)

    if lstm_1d_path and not os.path.exists(lstm_1d_path):
        fallback_paths = ["models/best_lstm_1d.pt"]
        lstm_1d_path = next((p for p in fallback_paths if os.path.exists(p)), None)
        
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

    # Load LSTM 1D
    lstm_1d_model, lstm_1d_vocab, lstm_1d_segment = None, None, False
    if lstm_1d_path:
        print(f"Đang tải LSTM 1 chiều từ: {BOLD}{lstm_1d_path}{RESET}...")
        lstm_1d_model, lstm_1d_vocab, lstm_1d_segment = load_lstm_model(lstm_1d_path, device)
        if lstm_1d_model:
            print(f"{GREEN}✓ Tải LSTM 1 chiều thành công! (Tách từ: {lstm_1d_segment}){RESET}")
    else:
        print(f"{YELLOW}⚠ Không tìm thấy file trọng số LSTM 1 chiều (best_lstm_1d.pt){RESET}")
        
    # Load Transformer
    trans_model, trans_tokenizer, trans_name, trans_segment = None, None, None, False
    if trans_path:
        print(f"Đang tải Transformer từ: {BOLD}{trans_path}{RESET}...")
        trans_model, trans_tokenizer, trans_name, trans_segment = load_trans_model(trans_path, device)
        if trans_model:
            print(f"{GREEN}✓ Tải Transformer ({trans_name}) thành công! (Tách từ: {trans_segment}){RESET}")
    else:
        print(f"{YELLOW}⚠ Không tìm thấy file trọng số Transformer (best_transformer.pt / best_transformer_phobert.pt / best_transformer_distilbert.pt){RESET}")

    if not lstm_model and not lstm_1d_model and not trans_model:
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

        # 1.b. Dự đoán bằng LSTM 1 chiều
        if lstm_1d_model:
            try:
                inputs = lstm_1d_vocab.encode(text, max_len=128, segment_words=lstm_1d_segment).to(device)
                with torch.no_grad():
                    logits = lstm_1d_model(inputs)
                    probs = torch.softmax(logits, dim=1).squeeze(0)
                    prob_fake = probs[1].item()
                    prob_real = probs[0].item()
                
                pred_label = f"{RED}TIN GIẢ (FAKE){RESET}" if prob_fake >= lstm_1d_threshold else f"{GREEN}TIN THẬT (REAL){RESET}"
                confidence = prob_fake if prob_fake >= lstm_1d_threshold else prob_real
                print(f"  └─ {BOLD}[LSTM 1D]:{RESET}     {pred_label} | Ngưỡng: {lstm_1d_threshold:.2f} | Độ tin cậy: {confidence*100:.2f}% (Fake: {prob_fake*100:.1f}%, Real: {prob_real*100:.1f}%)")
            except Exception as e:
                print(f"  └─ [LSTM 1D]: Lỗi dự đoán - {e}")
                
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

            # 1.b. Dự đoán bằng LSTM 1 chiều
            if lstm_1d_model:
                try:
                    inputs = lstm_1d_vocab.encode(user_text, max_len=128, segment_words=lstm_1d_segment).to(device)
                    with torch.no_grad():
                        logits = lstm_1d_model(inputs)
                        probs = torch.softmax(logits, dim=1).squeeze(0)
                        prob_fake = probs[1].item()
                        prob_real = probs[0].item()
                    
                    pred_label = f"{RED}TIN GIẢ (FAKE){RESET}" if prob_fake >= lstm_1d_threshold else f"{GREEN}TIN THẬT (REAL){RESET}"
                    confidence = prob_fake if prob_fake >= lstm_1d_threshold else prob_real
                    print(f"  └─ {BOLD}[LSTM 1D]:{RESET}     {pred_label} | Ngưỡng: {lstm_1d_threshold:.2f} | Độ tin cậy: {confidence*100:.2f}% (Fake: {prob_fake*100:.1f}%, Real: {prob_real*100:.1f}%)")
                except Exception as e:
                    print(f"  └─ [LSTM 1D]: Lỗi dự đoán - {e}")
                    
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
    parser.add_argument("--lstm_1d_path", type=str, default="models/best_lstm_1d.pt", help="Path to best_lstm_1d.pt")
    parser.add_argument("--trans_path", type=str, default="models/best_transformer.pt", help="Path to best_transformer.pt")
    parser.add_argument("--lstm_threshold", type=float, default=0.63, help="Decision threshold for BiLSTM (default 0.63)")
    parser.add_argument("--lstm_1d_threshold", type=float, default=0.58, help="Decision threshold for LSTM 1D (default 0.58)")
    parser.add_argument("--trans_threshold", type=float, default=0.50, help="Decision threshold for Transformer (default 0.50)")
    args = parser.parse_args()
    
    run_local_test(
        lstm_path=args.lstm_path,
        lstm_1d_path=args.lstm_1d_path,
        trans_path=args.trans_path,
        lstm_threshold=args.lstm_threshold,
        lstm_1d_threshold=args.lstm_1d_threshold,
        trans_threshold=args.trans_threshold
    )
