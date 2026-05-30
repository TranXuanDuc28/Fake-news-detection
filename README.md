# Vietnamese Fake News Detection System (Hệ thống Nhận diện Tin giả Tiếng Việt)

Hệ thống Nhận diện Tin giả tiếng Việt ứng dụng học máy chuyên sâu với giao diện điều khiển (dashboard) trực quan, hiện đại. Dự án tích hợp đồng thời hai kiến trúc mô hình học máy: **BiLSTM (Mạng LSTM hai chiều)** và **phoBERT (Transformer)** để kiểm định, phân tích nội dung văn bản và đưa ra dự đoán cùng các từ khóa giải thích (explainability) một cách chính xác.

---

## 🌟 Các Tính Năng Nổi Bật

1. **Kiểm Định Tin Tức Tương Tác**:
   - Nhận diện tin tức là **TIN GIẢ (FAKE NEWS)** hay **TIN THẬT (REAL NEWS)** theo thời gian thực.
   - Thể hiện xác suất tin cậy của mô hình dưới dạng các thanh đo tỷ lệ trực quan (gradient gauge bars).
   - Đo lường và hiển thị chi tiết độ trễ xử lý (Latency) của từng mô hình (BiLSTM ~ 1-3ms, phoBERT ~ 20-40ms).

2. **Crawl Tin Tức Từ URL Tự Động**:
   - Người dùng chỉ cần dán đường dẫn (URL) bài báo từ bất kỳ trang tin tức nào (VnExpress, Dân Trí, Tuổi Trẻ,...).
   - Hệ thống tự động bóc tách tiêu đề và nội dung chính của bài viết bằng `BeautifulSoup` để đưa vào kiểm định mà không cần copy-paste thủ công.

3. **Từ Khóa Giải Thích (Explainability Keyword Attribution)**:
   - Áp dụng phương pháp đánh giá đóng góp đặc trưng bằng kỹ thuật **Occlusion (Leave-One-Out)** kết hợp với **Bảo toàn vị trí (Position-Preserving <pad> Masking)** để loại bỏ nhiễu định vị.
   - Hệ thống tự động phân tích và hiển thị các **Từ ghép (Từ phức)** đóng góp mạnh nhất vào kết quả dự đoán của cả tin FAKE và tin REAL dưới dạng các thẻ màu (Emerald xanh lá cho tin REAL, Rose đỏ cho tin FAKE).

4. **Phân Tích Heuristics Giật Gân (Clickbait)**:
   - Quét và phát hiện các cụm từ giật gân, phóng đại thường gặp trong tin tức giả mạo (ví dụ: `thần dược`, `tin sốc`, `bí mật quốc gia`, `chữa bách bệnh`,...).

5. **Bảng Điều Khiển Tuning Tham Số**:
   - Trực quan hóa dữ liệu lịch sử huấn luyện checkpoint và so sánh hiệu năng sweeps trên 3 tham số quan trọng: **Learning Rate**, **Batch Size**, và **Dropout Rate** bằng biểu đồ cột tương tác thông qua **Chart.js**.
   - Tự động phân tích và kết luận cấu hình tham số tối ưu nhất cho từng mô hình được chọn.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```
d:\Fake-news-detection\
├── app/
│   ├── main.py                     # Máy chủ FastAPI (Backend APIs cho crawl, predict, metrics)
│   └── templates/
│       └── index.html              # Giao diện dashboard HTML/CSS Glassmorphic cao cấp
├── src/
│   ├── lstm_model.py               # Định nghĩa kiến trúc mô hình BiLSTM Classifier (PyTorch)
│   └── transformer_model.py        # Định nghĩa kiến trúc mô hình phoBERT Classifier (PyTorch)
├── data/
│   ├── tuning_results.json         # Dữ liệu sweeps điều chỉnh siêu tham số
│   └── final_training_history.json # Lịch sử huấn luyện qua các epoch
├── models/
│   ├── best_lstm.pt                # Checkpoint trọng số mô hình BiLSTM tốt nhất
│   └── best_transformer_phobert.pt # Checkpoint trọng số mô hình phoBERT tốt nhất
├── scratch/                        # Các kịch bản kiểm thử lâm thời và debug
├── requirements.txt                # Khai báo các thư viện phụ thuộc của Python
└── README.md                       # Tài liệu hướng dẫn sử dụng dự án (File này)
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Yêu cầu hệ thống
* Python phiên bản từ `3.8` đến `3.11`.
* Môi trường có hỗ trợ CUDA/GPU (không bắt buộc, hệ thống tự động fallback sang CPU).

### 2. Cài đặt các thư viện phụ thuộc
Mở terminal tại thư mục gốc của dự án và chạy lệnh sau để cài đặt các package cần thiết (bao gồm PyTorch, Transformers, FastAPI, PyVi để phân đoạn từ tiếng Việt và BeautifulSoup4):

```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng
Chạy tệp backend chính để kích hoạt máy chủ FastAPI cùng cơ chế tự động tải lại (hot-reload) khi có thay đổi mã nguồn:

```bash
python app/main.py
```

### 4. Truy cập ứng dụng
Sau khi khởi chạy thành công, mở trình duyệt web và truy cập địa chỉ sau:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🔬 Thuật Toán Phân Tích Từ Khóa Giải Thích (Explainability)

Hệ thống đánh giá tầm quan trọng của các từ ghép tiếng Việt theo thuật toán sau:
1. Văn bản đầu vào được làm sạch và phân đoạn từ thông qua thư viện `pyvi` (ví dụ: `thảo mộc` -> `thảo_mộc`).
2. Trích xuất tất cả các từ ghép (các từ chứa ký tự `_`).
3. Lần lượt che (mask) từng từ ghép bằng token đệm `<pad>` hoặc `<unk>` tương ứng với mô hình đó mà không làm thay đổi chiều dài câu, giúp loại bỏ nhiễu lệch vị trí trong mạng tuần tự/tự chú ý (attention).
4. Thực hiện suy luận đồng thời để lấy ra sự thay đổi xác suất:
   $$\text{Score}(i) = P_{\text{gốc}} - P_{\text{sau khi che từ } i}$$
5. Thu thập các từ có $\text{Score} > 0.0$ (tức việc loại bỏ từ đó làm giảm niềm tin dự đoán của mô hình vào nhãn hiện tại) và hiển thị lên giao diện.
