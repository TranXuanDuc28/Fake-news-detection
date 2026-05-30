from pyvi import ViTokenizer, ViPosTagger

text = "TIN NÓNG: Mỹ chính thức ngừng cấp toàn bộ visa du học cho sinh viên Việt Nam và các nước Đông Nam Á vô thời hạn."
tokenized = ViTokenizer.tokenize(text)
tokens, tags = ViPosTagger.postagging(tokenized)

with open("scratch/pos_output.txt", "w", encoding="utf-8") as f:
    for tok, tag in zip(tokens, tags):
        f.write(f"{tok} -> {tag}\n")
print("Done writing POS tags to scratch/pos_output.txt")
