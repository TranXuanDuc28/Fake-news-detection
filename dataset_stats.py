import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_raw_data

# Reconfigure stdout to use UTF-8 to prevent charmap encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def compute_text_stats(series):
    char_lens = series.str.len()
    word_lens = series.str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    return {
        "avg_chars": char_lens.mean() if len(char_lens) > 0 else 0,
        "max_chars": char_lens.max() if len(char_lens) > 0 else 0,
        "min_chars": char_lens.min() if len(char_lens) > 0 else 0,
        "avg_words": word_lens.mean() if len(word_lens) > 0 else 0,
        "max_words": word_lens.max() if len(word_lens) > 0 else 0,
        "min_words": word_lens.min() if len(word_lens) > 0 else 0
    }

def print_split_stats(name, df):
    total = len(df)
    real_count = len(df[df['label'] == 0])
    fake_count = len(df[df['label'] == 1])
    real_pct = (real_count / total * 100) if total > 0 else 0
    fake_pct = (fake_count / total * 100) if total > 0 else 0
    
    print(f"| {name:<25} | {total:<10,} | {real_count:<12,} ({real_pct:>6.2f}%) | {fake_count:<12,} ({fake_pct:>6.2f}%) |")
    return total, real_count, fake_count

def generate_statistics(data_dir="data"):
    print("=" * 90)
    print("                 THỐNG KÊ CHI TIẾT CÁC BỘ DỮ LIỆU (DATASET STATS)")
    print("=" * 90)
    
    datasets = {
        "1. BỘ GỐC REINTEL": {
            "ReINTEL Train": "train.csv",
            "ReINTEL Val": "val.csv",
            "ReINTEL Test": "test.csv"
        },
        "2. BỘ VFND (Original)": {
            "VFND Train": "vfnd_train.csv",
            "VFND Val": "vfnd_val.csv",
            "VFND Test": "vfnd_test.csv"
        },
        "3. BỘ TINGIA (Official Crawled)": {
            "TinGia Train": "tingia_train.csv",
            "TinGia Val": "tingia_val.csv",
            "TinGia Test": "tingia_test.csv"
        },
        "4. BỘ CUSTOM CRAWLED (Tạm thời bỏ)": {
            "Custom Crawled Train": "crawled_train.csv",
            "Custom Crawled Val": "crawled_val.csv",
            "Custom Crawled Test": "crawled_test.csv"
        }
    }
    
    loaded_dfs = {}
    
    for cat_name, files in datasets.items():
        print(f"\n{cat_name}:")
        print("-" * 90)
        print(f"| {'Tập chia (Split)':<25} | {'Tổng mẫu':<10} | {'Tin thật (Real)':<20} | {'Tin giả (Fake)':<20} |")
        print("-" * 90)
        
        cat_total = {"total": 0, "real": 0, "fake": 0}
        
        for name, filename in files.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    loaded_dfs[name] = df
                    t, r, f = print_split_stats(name, df)
                    cat_total["total"] += t
                    cat_total["real"] += r
                    cat_total["fake"] += f
                except Exception as e:
                    print(f"Lỗi khi đọc {filename}: {e}")
            else:
                print(f"| {name:<25} | {'Không tìm thấy tệp dữ liệu':<57} |")
                
        print("-" * 90)
        t_tot = cat_total["total"]
        if t_tot > 0:
            real_pct = (cat_total["real"] / t_tot * 100)
            fake_pct = (cat_total["fake"] / t_tot * 100)
            print(f"| {'TỔNG CỘNG NHÓM':<25} | {t_tot:<10,} | {cat_total['real']:<12,} ({real_pct:>6.2f}%) | {cat_total['fake']:<12,} ({fake_pct:>6.2f}%) |")
        print("=" * 90)

    # Combined training options stats
    for use_eda in [False, True]:
        mode_str = "TĂNG CƯỜNG EDA (num_aug=2)" if use_eda else "GỐC SẠCH (Chạy Random Oversampling hoặc chỉ Class Weight)"
        print("\n" + "=" * 90)
        print(f"   THỐNG KÊ KHI GỘP THEO TÙY CHỌN HUẤN LUYỆN - CHẾ ĐỘ: {mode_str}")
        print("=" * 90)
        
        options = ["none", "vfnd", "tingia", "both"]
        
        for opt_name in options:
            print(f"\nTùy chọn huấn luyện: --additional_dataset {opt_name} " + (f"--use_eda" if use_eda else ""))
            print("-" * 90)
            print(f"| {'Tập chia (Split)':<25} | {'Tổng mẫu':<10} | {'Tin thật (Real)':<20} | {'Tin giả (Fake)':<20} |")
            print("-" * 90)
            
            try:
                train_df, val_df, test_df = load_raw_data(data_dir, segment_words=False, additional_dataset=opt_name, use_eda=use_eda)
                print_split_stats("Merged Train", train_df)
                print_split_stats("Merged Val", val_df)
                print_split_stats("Merged Test", test_df)
                
                # Tính toán trùng lặp chéo (rò rỉ dữ liệu) để kiểm nghiệm
                def clean_norm(text):
                    return str(text).lower().strip().replace(" ", "")
                    
                train_set = set(train_df["post_message"].apply(clean_norm))
                val_set = set(val_df["post_message"].apply(clean_norm))
                test_set = set(test_df["post_message"].apply(clean_norm))
                
                val_leak = len(val_set.intersection(train_set))
                test_leak = len(test_set.intersection(train_set))
                
                print("-" * 90)
                print(f"| {'Rò rỉ Val -> Train':<25} | {val_leak:<10} | {'Đạt chuẩn khoa học (0.00%)' if val_leak == 0 else 'CẢNH BÁO RÒ RỈ':<57} |")
                print(f"| {'Rò rỉ Test -> Train':<25} | {test_leak:<10} | {'Đạt chuẩn khoa học (0.00%)' if test_leak == 0 else 'CẢNH BÁO RÒ RỈ':<57} |")
                
            except Exception as e:
                print(f"Lỗi tải dữ liệu cho tùy chọn {opt_name}: {e}")
            print("-" * 90)

    # Text length statistics
    print("\n" + "=" * 90)
    print("                 THỐNG KÊ ĐỘ DÀI VĂN BẢN (SỐ TỪ & SỐ KÝ TỰ CỦA CÁC BỘ GỐC)")
    print("=" * 90)
    for name, df in loaded_dfs.items():
        if "post_message" in df.columns:
            df["post_message"] = df["post_message"].fillna("")
            stats = compute_text_stats(df["post_message"])
            print(f"\n* Bộ dữ liệu: {name}")
            print(f"  - Số từ trung bình: {stats['avg_words']:.1f} từ (Lớn nhất: {stats['max_words']} từ, Nhỏ nhất: {stats['min_words']} từ)")
            print(f"  - Số ký tự trung bình: {stats['avg_chars']:.1f} ký tự (Lớn nhất: {stats['max_chars']} ký tự, Nhỏ nhất: {stats['min_chars']} ký tự)")
    print("=" * 90)

if __name__ == "__main__":
    generate_statistics()
