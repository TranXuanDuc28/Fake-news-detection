import json
import os
import sys

# Configure stdout to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Search for tuning results file
tuning_file = "data/tuning_results.json"
colab_path = "/content/drive/MyDrive/FakeNewsDetection_Results/data/tuning_results.json"

if not os.path.exists(tuning_file):
    if os.path.exists(colab_path):
        tuning_file = colab_path
    else:
        # Check other common paths
        alt_path = "../data/tuning_results.json"
        if os.path.exists(alt_path):
            tuning_file = alt_path
        else:
            print("❌ LỖI: Không tìm thấy file kết quả 'tuning_results.json'!")
            print("Vui lòng đảm bảo bạn đã chạy xong bước Tuning ở phần số 6.")
            sys.exit(1)

print(f"📂 Đang đọc kết quả tuning từ: {tuning_file}\n")

with open(tuning_file, "r", encoding="utf-8") as f:
    results = json.load(f)

for model_name in ["lstm", "transformer"]:
    print("=" * 65)
    print(f"🚀 BÁO CÁO TỔNG HỢP SIÊU THAM SỐ CHO MÔ HÌNH: {model_name.upper()}")
    print("=" * 65)
    
    model_data = results.get(model_name, {})
    best_overall_f1 = -1
    best_overall_config = {}
    
    # Track default hyperparams to reconstruct configurations
    defaults = {
        "lstm": {"lr": "1e-3", "bs": "16", "dp": "0.3"},
        "transformer": {"lr": "2e-5", "bs": "16", "dp": "0.3"}
    }[model_name]
    
    for sweep_type in ["dropout_sweep", "batch_size_sweep", "lr_sweep"]:
        display_name = sweep_type.replace('_', ' ').title()
        print(f"\n📍 Khảo sát: {display_name}")
        print(f" {'Giá trị':<12} | {'Val F1-score (Macro)':<22} | {'Val Loss':<12}")
        print(" " + "-" * 55)
        
        sweep_data = model_data.get(sweep_type, {})
        best_val = -1
        best_param = None
        
        for param, metrics in sweep_data.items():
            final_metrics = metrics.get("final_metrics", {})
            # Read macro F1 with multiple fallback mechanisms
            val_f1 = final_metrics.get("test_f1_macro", None)
            if val_f1 is None:
                val_f1 = final_metrics.get("f1_macro", None)
            if val_f1 is None:
                val_f1 = final_metrics.get("f1_binary", None)
            if val_f1 is None:
                val_f1 = final_metrics.get("test_f1", 0.0)
                
            if val_f1 <= 1.0:
                val_f1 = val_f1 * 100
                
            # Read loss with fallback to validation loss history
            val_loss = final_metrics.get("test_loss", None)
            if val_loss is None:
                val_loss = final_metrics.get("loss", None)
            if val_loss is None:
                loss_hist = metrics.get("val_loss_history", [])
                val_loss = loss_hist[-1] if loss_hist else 0.0
            
            print(f"  {param:<11} | {val_f1:>20.2f}% | {val_loss:>10.4f}")
            
            if val_f1 > best_val:
                best_val = val_f1
                best_param = param
                
        print(f" ➔ 🌟 Tốt nhất: {best_param} (F1: {best_val:.2f}%)")
        
    print("\n" + "-" * 65)
    print(f"📢 ĐỀ XUẤT CỦA HỆ THỐNG CHO {model_name.upper()}:")
    print(f"   Dựa trên các thực nghiệm đơn lẻ, bộ tham số tối ưu khuyến nghị là:")
    
    if model_name == "lstm":
        print("   - Tốc độ học (Learning Rate): 1e-3 (hoặc chọn mức tốt nhất ở LR Sweep)")
        print("   - Tỷ lệ bỏ rơi (Dropout):     (chọn mức tốt nhất ở Dropout Sweep)")
        print("   - Kích thước lô (Batch Size): (chọn mức tốt nhất ở Batch Size Sweep)")
    else:
        print("   - Tốc độ học (Learning Rate): 2e-5 (hoặc chọn mức tốt nhất ở LR Sweep)")
        print("   - Tỷ lệ bỏ rơi (Dropout):     (chọn mức tốt nhất ở Dropout Sweep)")
        print("   - Kích thước lô (Batch Size): (chọn mức tốt nhất ở Batch Size Sweep)")
    print("=" * 65 + "\n")
