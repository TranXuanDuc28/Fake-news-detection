import os
import sys
import pandas as pd

# Reconfigure stdout to use UTF-8 to prevent charmap encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def compute_text_stats(series):
    # Length in characters
    char_lens = series.str.len()
    # Length in words (split by space)
    word_lens = series.str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    return {
        "avg_chars": char_lens.mean(),
        "max_chars": char_lens.max(),
        "min_chars": char_lens.min(),
        "avg_words": word_lens.mean(),
        "max_words": word_lens.max(),
        "min_words": word_lens.min()
    }

def print_split_stats(name, df):
    total = len(df)
    real_count = len(df[df['label'] == 0])
    fake_count = len(df[df['label'] == 1])
    real_pct = (real_count / total * 100) if total > 0 else 0
    fake_pct = (fake_count / total * 100) if total > 0 else 0
    
    print(f"| {name:<20} | {total:<10,} | {real_count:<12,} ({real_pct:>6.2f}%) | {fake_count:<12,} ({fake_pct:>6.2f}%) |")
    return total, real_count, fake_count

def generate_statistics(data_dir="data"):
    print("=" * 80)
    print("                 THỐNG KÊ CHI TIẾT BỘ DỮ LIỆU (DATASET STATS)")
    print("=" * 80)
    
    files = {
        "Train gốc (train.csv)": "train.csv",
        "Validation (val.csv)": "val.csv",
        "Test (test.csv)": "test.csv",
        "VFND (additional_train.csv)": "additional_train.csv"
    }
    
    dfs = {}
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                dfs[name] = pd.read_csv(filepath)
            except Exception as e:
                print(f"Lỗi khi đọc {filename}: {e}")
        else:
            if filename != "additional_train.csv":
                print(f"Cảnh báo: Không tìm thấy tệp {filename} tại {data_dir}")
                
    print(f"| {'Bộ chia (Split)':<20} | {'Tổng mẫu':<10} | {'Tin thật (Real)':<20} | {'Tin giả (Fake)':<20} |")
    print("-" * 80)
    
    totals = {"total": 0, "real": 0, "fake": 0}
    for name, df in dfs.items():
        t, r, f = print_split_stats(name, df)
        totals["total"] += t
        totals["real"] += r
        totals["fake"] += f
        
    print("-" * 80)
    real_pct = (totals["real"] / totals["total"] * 100) if totals["total"] > 0 else 0
    fake_pct = (totals["fake"] / totals["total"] * 100) if totals["total"] > 0 else 0
    print(f"| {'TỔNG CỘNG':<20} | {totals['total']:<10,} | {totals['real']:<12,} ({real_pct:>6.2f}%) | {totals['fake']:<12,} ({fake_pct:>6.2f}%) |")
    print("=" * 80)
    
    # Text length statistics
    print("\n>>> THỐNG KÊ ĐỘ DÀI VĂN BẢN (SỐ TỪ & SỐ KÝ TỰ):")
    for name, df in dfs.items():
        if "post_message" in df.columns:
            df["post_message"] = df["post_message"].fillna("")
            stats = compute_text_stats(df["post_message"])
            print(f"\n* Bộ dữ liệu: {name}")
            print(f"  - Số từ trung bình: {stats['avg_words']:.1f} từ (Lớn nhất: {stats['max_words']} từ, Nhỏ nhất: {stats['min_words']} từ)")
            print(f"  - Số ký tự trung bình: {stats['avg_chars']:.1f} ký tự (Lớn nhất: {stats['max_chars']} ký tự, Nhỏ nhất: {stats['min_chars']} ký tự)")
    print("=" * 80)

if __name__ == "__main__":
    generate_statistics()
