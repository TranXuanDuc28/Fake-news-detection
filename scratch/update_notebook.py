import pathlib

notebook_path = pathlib.Path('Fake_News_Detection_Colab.ipynb')
if not notebook_path.exists():
    print("Notebook not found!")
    exit(1)

content = notebook_path.read_text(encoding='utf-8')

# Search strings
old_lstm = '"!python src/train.py --model lstm --epochs 15 --patience 5 --save_dir {DRIVE_MODELS} --data_dir data --history_file {DRIVE_DATA}/lstm_history.json --no_oversample --use_class_weights --additional_dataset both\\n"'
new_lstm = '"!python src/train.py --model lstm --epochs 15 --patience 5 --save_dir {DRIVE_MODELS} --data_dir data --history_file {DRIVE_DATA}/lstm_history.json --no_oversample --use_class_weights --additional_dataset both --lr 1e-3 --dropout 0.3 --batch_size 16\\n"'

old_transformer = '"!python src/train.py --model transformer --epochs 15 --patience 5 --save_dir {DRIVE_MODELS} --data_dir data --history_file {DRIVE_DATA}/transformer_history.json --no_oversample --use_class_weights --additional_dataset both\\n"'
new_transformer = '"!python src/train.py --model transformer --epochs 15 --patience 5 --save_dir {DRIVE_MODELS} --data_dir data --history_file {DRIVE_DATA}/transformer_history.json --no_oversample --use_class_weights --additional_dataset both --lr 2e-5 --dropout 0.3 --batch_size 16\\n"'

if old_lstm in content:
    content = content.replace(old_lstm, new_lstm)
    print("Updated LSTM train command.")
else:
    print("LSTM train command already updated or not found.")

if old_transformer in content:
    content = content.replace(old_transformer, new_transformer)
    print("Updated Transformer train command.")
else:
    print("Transformer train command already updated or not found.")

notebook_path.write_text(content, encoding='utf-8')
print("Done!")
