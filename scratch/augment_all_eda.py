import sys
import os
import pandas as pd

# Thiết lập bảng mã UTF-8 cho stdout trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục hiện tại để có thể import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratch.augment_fake_news import augment_sentence

def augment_file(input_file, output_file, num_aug=2):
    print(f"\n--> Bắt đầu tăng cường tệp: {input_file} (num_aug={num_aug})")
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy tệp {input_file}")
        return

    # Đọc dữ liệu
    df = pd.read_csv(input_file)
    print(f"    Số lượng mẫu ban đầu: {len(df)}")
    
    fake_df = df[df['label'] == 1].copy()
    real_df = df[df['label'] == 0].copy()
    
    print(f"    - Tin thật gốc (Real): {len(real_df)}")
    print(f"    - Tin giả gốc (Fake): {len(fake_df)}")

    # Chỉ tăng cường cho tin giả (label=1)
    augmented_records = []
    for _, row in fake_df.iterrows():
        text = str(row['post_message'])
        aug_texts = augment_sentence(text, num_aug=num_aug)
        for aug_text in aug_texts:
            augmented_records.append({
                'post_message': aug_text,
                'label': 1
            })

    aug_df = pd.DataFrame(augmented_records)
    print(f"    -> Đã sinh thêm {len(aug_df)} mẫu tin giả được tăng cường.")

    # Gộp mẫu gốc và mẫu tăng cường
    final_df = pd.concat([real_df, fake_df, aug_df], ignore_index=True)
    # Trộn ngẫu nhiên
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"    Số lượng mẫu sau tăng cường: {len(final_df)} (Thật: {len(final_df[final_df['label'] == 0])}, Giả: {len(final_df[final_df['label'] == 1])})")

    # Lưu tệp
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"    Đã lưu tệp thành công tại: {output_file}")

def main():
    data_dir = "data"
    num_aug = 2
    
    files_to_process = {
        "train.csv": "train_eda.csv",
        "vfnd_train.csv": "vfnd_train_eda.csv",
        "tingia_train.csv": "tingia_train_eda.csv"
    }

    print("=== BẮT ĐẦU QUY TRÌNH TĂNG CƯỜNG DỮ LIỆU ĐỒNG LOẠT (num_aug=2) ===")
    
    for in_name, out_name in files_to_process.items():
        in_path = os.path.join(data_dir, in_name)
        out_path = os.path.join(data_dir, out_name)
        augment_file(in_path, out_path, num_aug=num_aug)
        
    print("\n=== HOÀN THÀNH QUY TRÌNH TĂNG CƯỜNG DỮ LIỆU ĐỒNG LOẠT! ===")

if __name__ == "__main__":
    main()
