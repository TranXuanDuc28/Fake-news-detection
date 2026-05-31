import sys
import os
import pandas as pd

# Thiết lập bảng mã UTF-8 cho stdout trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import hàm augment_sentence từ script augment_fake_news gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratch.augment_fake_news import augment_sentence

def main():
    input_path = "data/tingia_train.csv"
    output_path = "data/tingia_train.csv"
    num_aug = 3

    print(f"=== BẮT ĐẦU TĂNG CƯỜNG DỮ LIỆU TIN GIA HUẤN LUYỆN (num_aug={num_aug}) ===")
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy tệp {input_path}")
        return

    # Đọc dữ liệu huấn luyện hiện tại
    df = pd.read_csv(input_path)
    print(f"Số lượng mẫu ban đầu: {len(df)}")
    
    fake_df = df[df['label'] == 1].copy()
    real_df = df[df['label'] == 0].copy()
    
    print(f" - Tin thật (Original Real): {len(real_df)}")
    print(f" - Tin giả (Original Fake): {len(fake_df)}")

    # Chỉ tăng cường cho tin giả (label=1)
    augmented_records = []
    for _, row in fake_df.iterrows():
        text = str(row['post_message'])
        # Tạo thêm num_aug mẫu biến thể từ mẫu tin giả gốc
        aug_texts = augment_sentence(text, num_aug=num_aug)
        for aug_text in aug_texts:
            augmented_records.append({
                'post_message': aug_text,
                'label': 1
            })

    aug_df = pd.DataFrame(augmented_records)
    print(f" -> Đã tạo thêm {len(aug_df)} mẫu tin giả được tăng cường (num_aug={num_aug}).")

    # Gộp mẫu gốc và mẫu tăng cường
    # Giữ nguyên tin thật, gộp tin giả gốc + tin giả tăng cường
    final_df = pd.concat([real_df, fake_df, aug_df], ignore_index=True)
    # Trộn ngẫu nhiên các hàng
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Tổng số mẫu sau tăng cường: {len(final_df)}")
    print(f" - Tin thật (Real): {len(final_df[final_df['label'] == 0])}")
    print(f" - Tin giả (Fake): {len(final_df[final_df['label'] == 1])}")

    # Ghi đè vào tệp output
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Đã lưu tệp tin gia tăng cường thành công tại: {output_path}")

if __name__ == "__main__":
    main()
