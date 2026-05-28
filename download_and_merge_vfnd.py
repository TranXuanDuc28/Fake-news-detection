import os
import sys
import zipfile
import urllib.request
import json
import pandas as pd
import shutil

# Reconfigure stdout to use UTF-8 encoding (fixes Windows terminal encoding errors)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def download_and_merge_vfnd():

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. URL tải bộ dữ liệu VFND phiên bản 1.0 dạng zip từ GitHub
    zip_url = "https://github.com/thanhhocse96/vfnd-vietnamese-fake-news-datasets/archive/refs/tags/1.0.zip"
    zip_path = os.path.join(data_dir, "vfnd_temp.zip")
    extract_path = os.path.join(data_dir, "vfnd_extracted")
    
    print("=== BẮT ĐẦU TẢI VÀ GỘP BỘ DỮ LIỆU VFND ===")
    print(f"1. Đang tải tệp tin từ: {zip_url} ...")
    
    # Tải file zip về thư mục data/
    req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, "wb") as f_out:
            f_out.write(response.read())
        print("--> Tải tệp zip thành công!")
    except Exception as e:
        print(f"--> Lỗi khi tải tệp: {e}")
        return

    # 2. Giải nén tệp zip
    print(f"2. Đang giải nén tệp {zip_path} vào {extract_path} ...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("--> Giải nén thành công!")
    except Exception as e:
        print(f"--> Lỗi khi giải nén: {e}")
        return

    # 3. Quét các tệp JSON và phân loại nhãn
    print("3. Đang đọc và trích xuất dữ liệu từ các file JSON...")
    data = []
    
    # Walk qua toàn bộ thư mục giải nén để tìm các tệp JSON
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                # Xác định nhãn dựa trên đường dẫn thư mục chứa
                # 0: Real (Tin thật), 1: Fake (Tin giả)
                label = None
                normalized_root = root.replace("\\", "/").lower()
                if "real" in normalized_root:
                    label = 0
                elif "fake" in normalized_root:
                    label = 1
                
                # Nếu không xác định được nhãn hoặc nằm ở thư mục "Misleading", bỏ qua
                if label is None:
                    continue
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = json.load(f)
                        
                        # Trích xuất nội dung văn bản (thường nằm ở trường 'text' hoặc 'content')
                        text = content.get("text", "")
                        if not text:
                            text = content.get("content", "")
                        if not text:
                            text = content.get("title", "") # Fallback về tiêu đề nếu không có body
                            
                        text = str(text).strip()
                        if text:
                            data.append({
                                "post_message": text,
                                "label": label,
                                "source_file": file
                            })
                except Exception as e:
                    print(f"Lỗi khi đọc file {file}: {e}")

    print(f"--> Đã trích xuất xong {len(data)} bài viết từ VFND.")

    # 4. Ghi dữ liệu ra file additional_train.csv
    if len(data) > 0:
        df = pd.DataFrame(data)
        
        # Xem phân bổ nhãn của bộ dữ liệu VFND vừa tải
        real_count = (df['label'] == 0).sum()
        fake_count = (df['label'] == 1).sum()
        print(f"   + Tin thật (Real): {real_count} bài viết")
        print(f"   + Tin giả (Fake): {fake_count} bài viết")
        
        output_file = os.path.join(data_dir, "additional_train.csv")
        # Lưu file CSV chỉ gồm 2 cột chính
        df[["post_message", "label"]].to_csv(output_file, index=False, encoding="utf-8")
        print(f"4. Đã lưu bộ dữ liệu phụ thành công tại: {output_file}")
        print("=== HOÀN THÀNH GỘP DỮ LIỆU! ===")
        print("Kể từ bây giờ, khi chạy train.py hoặc notebook, dữ liệu này sẽ tự động được gộp vào tập Train chính!")
    else:
        print("--> Không trích xuất được dữ liệu hợp lệ nào.")

    # 5. Dọn dẹp các tệp tạm để tiết kiệm bộ nhớ
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        print("5. Đã dọn dẹp các thư mục giải nén tạm thời.")
    except Exception as e:
        print(f"Lỗi khi dọn dẹp thư mục tạm: {e}")

if __name__ == "__main__":
    download_and_merge_vfnd()
