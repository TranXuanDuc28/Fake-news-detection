import json
import os

notebook_path = "Fake_News_Detection_Colab.ipynb"

if not os.path.exists(notebook_path):
    print("Notebook file not found!")
    exit(1)

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

cells = notebook.get("cells", [])

for cell in cells:
    # 1. Update training code cell
    if cell.get("cell_type") == "code":
        source_text = "".join(cell.get("source", []))
        if 'print("--- BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH BiLSTM TỐI ƯU ---")' in source_text and "lstm_1d" not in source_text:
            cell["source"] = [
                "print(\"--- BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH BiLSTM TỐI ƯU ---\")\n",
                "!python src/train.py --model lstm --epochs 15 --patience 5 --save_dir {DRIVE_MODELS} --data_dir data --history_file {DRIVE_DATA}/lstm_history.json --no_oversample --use_class_weights --additional_dataset both\n",
                "\n",
                "print(\"\\n--- BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH LSTM 1 CHIỀU TỐI ƯU ---\")\n",
                "!python src/train.py --model lstm_1d --epochs 15 --patience 5 --save_dir {DRIVE_MODELS} --data_dir data --history_file {DRIVE_DATA}/lstm_1d_history.json --no_oversample --use_class_weights --additional_dataset both\n",
                "\n",
                "print(\"\\n--- BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH TRANSFORMER TỐI ƯU ---\")\n",
                "!python src/train.py --model transformer --epochs 15 --patience 5 --save_dir {DRIVE_MODELS} --data_dir data --history_file {DRIVE_DATA}/transformer_history.json --no_oversample --use_class_weights --additional_dataset both\n"
            ]
            print("Updated training code cell.")

    # 2. Update history sync description and code cell
    if cell.get("cell_type") == "markdown":
        source_text = "".join(cell.get("source", []))
        if "Gộp tệp lịch sử huấn luyện của cả 2 mô hình" in source_text:
            cell["source"] = [
                "## 8. Đồng bộ hóa lịch sử huấn luyện\n",
                "Gộp tệp lịch sử huấn luyện của cả 3 mô hình lại thành `final_training_history.json` lưu trên Google Drive."
            ]
            print("Updated history sync markdown cell.")

    if cell.get("cell_type") == "code":
        source_text = "".join(cell.get("source", []))
        if "lstm_h_path = os.path.join(DRIVE_DATA, \"lstm_history.json\")" in source_text and "lstm_1d_h_path" not in source_text:
            cell["source"] = [
                "import json\n",
                "import os\n",
                "\n",
                "lstm_h_path = os.path.join(DRIVE_DATA, \"lstm_history.json\")\n",
                "lstm_1d_h_path = os.path.join(DRIVE_DATA, \"lstm_1d_history.json\")\n",
                "trans_h_path = os.path.join(DRIVE_DATA, \"transformer_history.json\")\n",
                "final_h_path = os.path.join(DRIVE_DATA, \"final_training_history.json\")\n",
                "\n",
                "try:\n",
                "    combined_history = {}\n",
                "    \n",
                "    # 1. Load LSTM history\n",
                "    if os.path.exists(lstm_h_path):\n",
                "        with open(lstm_h_path) as f:\n",
                "            combined_history[\"lstm\"] = json.load(f)\n",
                "            \n",
                "    # 1.b Load LSTM 1D history\n",
                "    if os.path.exists(lstm_1d_h_path):\n",
                "        with open(lstm_1d_h_path) as f:\n",
                "            combined_history[\"lstm_1d\"] = json.load(f)\n",
                "            \n",
                "    # 2. Load Transformer history (with fallback to models/history.json)\n",
                "    if os.path.exists(trans_h_path):\n",
                "        with open(trans_h_path) as f:\n",
                "            combined_history[\"transformer\"] = json.load(f)\n",
                "    elif os.path.exists(os.path.join(DRIVE_MODELS, \"history.json\")):\n",
                "        with open(os.path.join(DRIVE_MODELS, \"history.json\")) as f:\n",
                "            combined_history[\"transformer\"] = json.load(f)\n",
                "        \n",
                "    with open(final_h_path, \"w\", encoding=\"utf-8\") as f:\n",
                "        json.dump(combined_history, f, indent=4)\n",
                "        \n",
                "    print(\"✅ Đã gộp và tạo thành công final_training_history.json trên Google Drive!\")\n",
                "except Exception as e:\n",
                "    print(f\"❌ Lỗi khi gộp lịch sử huấn luyện: {e}\")"
            ]
            print("Updated history sync code cell.")

    # 3. Update plotting description and code cell
    if cell.get("cell_type") == "markdown":
        source_text = "".join(cell.get("source", []))
        if "của cả 2 mô hình BiLSTM và Transformer." in source_text:
            cell["source"] = [
                "## 9. Vẽ biểu đồ Lịch sử huấn luyện (Training History)\n",
                "Vẽ trực quan hóa đường cong suy giảm lỗi (Loss) và chỉ số Val F1 qua các epoch của cả 3 mô hình BiLSTM, LSTM 1D và Transformer."
            ]
            print("Updated plotting markdown cell.")

    if cell.get("cell_type") == "code":
        source_text = "".join(cell.get("source", []))
        if 'models_to_plot = [m for m in ["lstm", "transformer"] if m in hist]' in source_text:
            cell["source"] = [
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "import torch\n",
                "\n",
                "try:\n",
                "    with open(final_h_path) as f:\n",
                "        hist = json.load(f)\n",
                "        \n",
                "    models_to_plot = [m for m in [\"lstm\", \"lstm_1d\", \"transformer\"] if m in hist]\n",
                "    num_models = len(models_to_plot)\n",
                "    \n",
                "    if num_models > 0:\n",
                "        fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 5.5), squeeze=False)\n",
                "        \n",
                "        for idx, model_key in enumerate(models_to_plot):\n",
                "            ax = axes[0, idx]\n",
                "            model_data = hist[model_key]\n",
                "            epochs = np.arange(1, len(model_data[\"train_loss\"]) + 1)\n",
                "            \n",
                "            # Lấy tên hiển thị của mô hình\n",
                "            if model_key == \"lstm\":\n",
                "                model_name = \"BiLSTM\"\n",
                "            elif model_key == \"lstm_1d\":\n",
                "                model_name = \"LSTM 1D\"\n",
                "            else:\n",
                "                model_name = \"Transformer\"\n",
                "                \n",
                "            if model_key == \"transformer\":\n",
                "                # Đọc tên model name từ checkpoint nếu có\n",
                "                try:\n",
                "                    checkpoint_file = os.path.join(DRIVE_MODELS, \"best_transformer.pt\")\n",
                "                    if not os.path.exists(checkpoint_file):\n",
                "                        checkpoint_file = os.path.join(DRIVE_MODELS, \"best_transformer_phobert.pt\")\n",
                "                    if not os.path.exists(checkpoint_file):\n",
                "                        checkpoint_file = os.path.join(DRIVE_MODELS, \"best_transformer_distilbert.pt\")\n",
                "                        \n",
                "                    checkpoint = torch.load(checkpoint_file, map_location=\"cpu\")\n",
                "                    backbone = checkpoint.get(\"hyperparameters\", {}).get(\"transformer_model_name\", \"transformer\")\n",
                "                    model_name = backbone.split(\"/\")[-1].upper()\n",
                "                except:\n",
                "                    pass\n",
                "            \n",
                "            # Vẽ đồ thị Train Loss và Val Loss\n",
                "            ax.plot(epochs, model_data[\"train_loss\"], label=\"Train Loss\", color=\"#1f77b4\", marker=\"o\", linewidth=2)\n",
                "            ax.plot(epochs, model_data[\"val_loss\"], label=\"Val Loss\", color=\"#ff7f0e\", marker=\"s\", linewidth=2)\n",
                "            \n",
                "            ax.set_title(f\"Loss Curve - {model_name}\", fontsize=14, fontweight=\"bold\")\n",
                "            ax.set_xlabel(\"Epochs\", fontsize=12)\n",
                "            ax.set_ylabel(\"Loss\", fontsize=12)\n",
                "            ax.legend(fontsize=11)\n",
                "            ax.grid(True, linestyle=\"--\", alpha=0.6)\n",
                "            ax.set_xticks(epochs)\n",
                "            \n",
                "        plt.tight_layout()\n",
                "        plt.show()\n",
                "    else:\n",
                "        print(\"⚠ Không tìm thấy dữ liệu lịch sử của mô hình nào.\")\n",
                "except Exception as e:\n",
                "    print(f\"❌ Lỗi vẽ đồ thị: {e}\")"
            ]
            print("Updated plotting code cell.")

    # 4. Update classification report description and code cell
    if cell.get("cell_type") == "markdown":
        source_text = "".join(cell.get("source", []))
        if "Confusion Matrix cho cả 2 mô hình." in source_text:
            cell["source"] = [
                "## 10. Hiển thị Classification Report & Confusion Matrix\n",
                "Ô này sẽ tải trực tiếp kết quả dự đoán test set, in ra báo cáo phân loại chi tiết (Precision, Recall, F1 của Tin thật/Tin giả) và vẽ ma trận nhầm lẫn Confusion Matrix cho cả 3 mô hình."
            ]
            print("Updated metrics markdown cell.")

    if cell.get("cell_type") == "code":
        source_text = "".join(cell.get("source", []))
        if 'for model_type in ["lstm", "transformer"]:' in source_text:
            cell["source"] = [
                "import seaborn as sns\n",
                "from sklearn.metrics import classification_report, confusion_matrix\n",
                "\n",
                "for model_type in [\"lstm\", \"lstm_1d\", \"transformer\"]:\n",
                "    results_path = os.path.join(DRIVE_MODELS, f\"test_{model_type}_results.json\")\n",
                "    if os.path.exists(results_path):\n",
                "        print(\"\\n\" + \"=\"*65)\n",
                "        print(f\" 📊 BÁO CÁO PHÂN LOẠI CHI TIẾT CHO MÔ HÌNH: {model_type.upper()}\")\n",
                "        print(\"=\"*65)\n",
                "        \n",
                "        with open(results_path) as f:\n",
                "            res = json.load(f)\n",
                "            \n",
                "        targets = res.get(\"targets\", [])\n",
                "        preds = res.get(\"predictions\", [])\n",
                "        \n",
                "        if targets and preds:\n",
                "            # 1. Classification report\n",
                "            print(\"\\nClassification Report:\")\n",
                "            print(classification_report(targets, preds, target_names=[\"Tin thật (Real)\", \"Tin giả (Fake)\"]))\n",
                "            \n",
                "            # 2. Confusion Matrix Heatmap\n",
                "            cm = confusion_matrix(targets, preds)\n",
                "            plt.figure(figsize=(5, 4))\n",
                "            sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", \n",
                "                        xticklabels=[\"Real\", \"Fake\"], \n",
                "                        yticklabels=[\"Real\", \"Fake\"])\n",
                "            plt.title(f\"Confusion Matrix - {model_type.upper()}\", fontsize=11, fontweight=\"bold\")\n",
                "            plt.ylabel(\"Nhãn thực tế (True Label)\")\n",
                "            plt.xlabel(\"Nhãn dự đoán (Predicted Label)\")\n",
                "            plt.show()\n",
                "        else:\n",
                "            print(\"⚠️ Không tìm thấy dữ liệu targets/predictions trong tệp kết quả.\")\n",
                "    else:\n",
                "        print(f\"⚠️ Không tìm thấy tệp kết quả test cho {model_type.upper()} tại {results_path}\")"
            ]
            print("Updated metrics code cell.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Saved notebook successfully.")
