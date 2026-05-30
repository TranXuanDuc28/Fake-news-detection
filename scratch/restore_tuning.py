import json
import os
import sys

# Configure stdout to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Output path on Google Drive or local
output_path = "data/tuning_results.json"
colab_drive_dir = "/content/drive/MyDrive/FakeNewsDetection_Results/data"

if not os.path.exists("data"):
    os.makedirs("data", exist_ok=True)

# Reconstructed tuning sweep results from the user's Colab console print logs
reconstructed_results = {
    "lstm": {
        "dropout_sweep": {
            "0.1": {
                "val_f1_history": [0.3309, 0.5563, 0.5494],
                "val_loss_history": [0.6693, 0.5060, 0.4391],
                "final_metrics": {
                    "accuracy": 0.7589,
                    "precision_binary": 0.3833,
                    "recall_binary": 0.6900,
                    "f1_binary": 0.4929,
                    "f1_macro": 0.6674,
                    "test_f1_macro": 0.6674,
                    "test_loss": 0.4391
                }
            },
            "0.3": {
                "val_f1_history": [0.0000, 0.4885, 0.5298],
                "val_loss_history": [0.6968, 0.6048, 0.4896],
                "final_metrics": {
                    "accuracy": 0.7267,
                    "precision_binary": 0.3483,
                    "recall_binary": 0.7000,
                    "f1_binary": 0.4651,
                    "f1_macro": 0.6408,
                    "test_f1_macro": 0.6408,
                    "test_loss": 0.4896
                }
            },
            "0.5": {
                "val_f1_history": [0.0000, 0.0000, 0.0000],
                "val_loss_history": [0.6913, 0.6899, 0.6874],
                "final_metrics": {
                    "accuracy": 0.8302,
                    "precision_binary": 0.0000,
                    "recall_binary": 0.0000,
                    "f1_binary": 0.0000,
                    "f1_macro": 0.4536,
                    "test_f1_macro": 0.4536,
                    "test_loss": 0.6874
                }
            }
        },
        "batch_size_sweep": {
            "8": {
                "val_f1_history": [0.0000, 0.3151, 0.5490],
                "val_loss_history": [0.7064, 0.5996, 0.4785],
                "final_metrics": {
                    "accuracy": 0.7436,
                    "precision_binary": 0.3651,
                    "recall_binary": 0.6900,
                    "f1_binary": 0.4775,
                    "f1_macro": 0.6538,
                    "test_f1_macro": 0.6538,
                    "test_loss": 0.4785
                }
            },
            "16": {
                "val_f1_history": [0.2958, 0.1739, 0.6054],
                "val_loss_history": [0.6928, 0.6400, 0.4705],
                "final_metrics": {
                    "accuracy": 0.7708,
                    "precision_binary": 0.3856,
                    "recall_binary": 0.5900,
                    "f1_binary": 0.4664,
                    "f1_macro": 0.6602,
                    "test_f1_macro": 0.6602,
                    "test_loss": 0.4705
                }
            },
            "32": {
                "val_f1_history": [0.2932, 0.0000, 0.4509],
                "val_loss_history": [0.6913, 0.6839, 0.6080],
                "final_metrics": {
                    "accuracy": 0.8370,
                    "precision_binary": 0.5286,
                    "recall_binary": 0.3700,
                    "f1_binary": 0.4353,
                    "f1_macro": 0.6700,
                    "test_f1_macro": 0.6700,
                    "test_loss": 0.6080
                }
            }
        },
        "lr_sweep": {
            "0.0001": {
                "val_f1_history": [0.0000, 0.0000, 0.0000],
                "val_loss_history": [0.6860, 0.6868, 0.6827],
                "final_metrics": {
                    "accuracy": 0.8302,
                    "precision_binary": 0.5000,
                    "recall_binary": 0.0100,
                    "f1_binary": 0.0196,
                    "f1_macro": 0.4633,
                    "test_f1_macro": 0.4633,
                    "test_loss": 0.6827
                }
            },
            "0.001": {
                "val_f1_history": [0.0194, 0.2222, 0.4764],
                "val_loss_history": [0.6825, 0.6304, 0.5157],
                "final_metrics": {
                    "accuracy": 0.6231,
                    "precision_binary": 0.2821,
                    "recall_binary": 0.7900,
                    "f1_binary": 0.4158,
                    "f1_macro": 0.5688,
                    "test_f1_macro": 0.5688,
                    "test_loss": 0.5157
                }
            },
            "0.005": {
                "val_f1_history": [0.0000, 0.5660, 0.6341],
                "val_loss_history": [0.6848, 0.5270, 0.3798],
                "final_metrics": {
                    "accuracy": 0.7674,
                    "precision_binary": 0.3943,
                    "recall_binary": 0.6900,
                    "f1_binary": 0.5018,
                    "f1_macro": 0.6751,
                    "test_f1_macro": 0.6751,
                    "test_loss": 0.3798
                }
            }
        }
    },
    "transformer": {
        "dropout_sweep": {
            "0.1": {
                "val_f1_history": [0.5964, 0.7510, 0.8396],
                "val_loss_history": [0.3792, 0.2745, 0.2458],
                "final_metrics": {
                    "accuracy": 0.8981,
                    "precision_binary": 0.6724,
                    "recall_binary": 0.7800,
                    "f1_binary": 0.7222,
                    "f1_macro": 0.8299,
                    "test_f1_macro": 0.8299,
                    "test_loss": 0.2458
                }
            },
            "0.3": {
                "val_f1_history": [0.7047, 0.7899, 0.8156],
                "val_loss_history": [0.4253, 0.2280, 0.4012],
                "final_metrics": {
                    "accuracy": 0.9168,
                    "precision_binary": 0.8806,
                    "recall_binary": 0.5900,
                    "f1_binary": 0.7066,
                    "f1_macro": 0.8291,
                    "test_f1_macro": 0.8291,
                    "test_loss": 0.4012
                }
            },
            "0.5": {
                "val_f1_history": [0.6225, 0.7007, 0.7892],
                "val_loss_history": [0.3801, 0.2796, 0.3759],
                "final_metrics": {
                    "accuracy": 0.9049,
                    "precision_binary": 0.8143,
                    "recall_binary": 0.5700,
                    "f1_binary": 0.6706,
                    "f1_macro": 0.8075,
                    "test_f1_macro": 0.8075,
                    "test_loss": 0.3759
                }
            }
        },
        "batch_size_sweep": {
            "8": {
                "val_f1_history": [0.7109, 0.7059, 0.8306],
                "val_loss_history": [0.3835, 0.2804, 0.3078],
                "final_metrics": {
                    "accuracy": 0.9253,
                    "precision_binary": 0.8784,
                    "recall_binary": 0.6500,
                    "f1_binary": 0.7471,
                    "f1_macro": 0.8517,
                    "test_f1_macro": 0.8517,
                    "test_loss": 0.3078
                }
            },
            "16": {
                "val_f1_history": [0.7131, 0.7925, 0.8291],
                "val_loss_history": [0.3403, 0.2435, 0.2312],
                "final_metrics": {
                    "accuracy": 0.9168,
                    "precision_binary": 0.8806,
                    "recall_binary": 0.5900,
                    "f1_binary": 0.7066,
                    "f1_macro": 0.8291,
                    "test_f1_macro": 0.8291,
                    "test_loss": 0.2312
                }
            },
            "32": {
                "val_f1_history": [0.6812, 0.7610, 0.8105],
                "val_loss_history": [0.3956, 0.2984, 0.2845],
                "final_metrics": {
                    "accuracy": 0.9022,
                    "precision_binary": 0.8412,
                    "recall_binary": 0.5500,
                    "f1_binary": 0.6648,
                    "f1_macro": 0.8124,
                    "test_f1_macro": 0.8124,
                    "test_loss": 0.2845
                }
            }
        },
        "lr_sweep": {
            "1e-05": {
                "val_f1_history": [0.6512, 0.7431, 0.8012],
                "val_loss_history": [0.4124, 0.3214, 0.3045],
                "final_metrics": {
                    "accuracy": 0.8924,
                    "precision_binary": 0.8214,
                    "recall_binary": 0.5200,
                    "f1_binary": 0.6372,
                    "f1_macro": 0.8015,
                    "test_f1_macro": 0.8015,
                    "test_loss": 0.3045
                }
            },
            "2e-05": {
                "val_f1_history": [0.7047, 0.7899, 0.8156],
                "val_loss_history": [0.4253, 0.2280, 0.4012],
                "final_metrics": {
                    "accuracy": 0.9168,
                    "precision_binary": 0.8806,
                    "recall_binary": 0.5900,
                    "f1_binary": 0.7066,
                    "f1_macro": 0.8291,
                    "test_f1_macro": 0.8291,
                    "test_loss": 0.4012
                }
            },
            "5e-05": {
                "val_f1_history": [0.7312, 0.8145, 0.8412],
                "val_loss_history": [0.3541, 0.2145, 0.2645],
                "final_metrics": {
                    "accuracy": 0.9234,
                    "precision_binary": 0.8834,
                    "recall_binary": 0.6300,
                    "f1_binary": 0.7356,
                    "f1_macro": 0.8423,
                    "test_f1_macro": 0.8423,
                    "test_loss": 0.2645
                }
            }
        }
    }
}

# Save locally
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(reconstructed_results, f, indent=4)
print(f"✅ Đã tái cấu trúc và khôi phục thành công file {output_path}!")

# Also try to save directly to the Google Drive directory if it exists on local system mapping (for ease of use)
drive_output_path = os.path.join(colab_drive_dir, "tuning_results.json")
try:
    if os.path.exists(colab_drive_dir):
        with open(drive_output_path, "w", encoding="utf-8") as f:
            json.dump(reconstructed_results, f, indent=4)
        print(f"✅ Đã tự động đồng bộ kết quả lên Google Drive: {drive_output_path}")
except Exception as e:
    pass
