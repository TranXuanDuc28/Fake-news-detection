import pandas as pd
import random
import argparse
import os

# --- Từ điển đồng nghĩa cơ bản (Tiếng Việt) ---
SYNONYMS = {
    "công an": ["cảnh sát", "lực lượng chức năng", "cơ quan điều tra"],
    "bắt giữ": ["tạm giữ", "bắt giam", "khởi tố", "mời làm việc"],
    "đối tượng": ["kẻ gian", "thủ phạm", "nghi phạm", "tên trộm"],
    "lừa đảo": ["chiếm đoạt", "móc túi", "dụ dỗ", "lừa gạt", "lừa tiền"],
    "ngân hàng": ["tổ chức tín dụng", "nhà băng"],
    "chuyển tiền": ["gửi tiền", "chuyển khoản", "bắn tiền", "chuyển khoản ngân hàng"],
    "thông báo": ["cảnh báo", "tin khẩn", "khuyến cáo", "đưa tin"],
    "virus": ["vi rút", "mầm bệnh", "vi khuẩn"],
    "nguy hiểm": ["nghiêm trọng", "rất nặng", "chết người"],
    "điều trị": ["chữa trị", "cứu chữa", "khám chữa"],
    "thuốc": ["thảo dược", "phương thuốc", "thần dược", "bài thuốc"],
    "ung thư": ["bệnh hiểm nghèo", "bệnh nan y"],
    "miễn phí": ["không mất tiền", "tặng không", "0 đồng"],
    "công nghệ": ["kỹ thuật", "công nghệ cao", "phần mềm"],
    "trí tuệ nhân tạo": ["AI", "trí thông minh nhân tạo"],
    "người dân": ["người tiêu dùng", "quần chúng", "bà con"],
    "tuyệt đối": ["chắc chắn", "hoàn toàn"],
    "phát hiện": ["tìm ra", "bắt quả tang", "khám phá"],
    "đóng cửa": ["phong tỏa", "ngừng hoạt động", "cách ly"],
    "xử phạt": ["phạt tiền", "phạt hành chính", "đình chỉ"]
}

def get_synonyms(word):
    for k, v in SYNONYMS.items():
        if word.lower() == k:
            return v
        if word.lower() in v:
            return [k] + [w for w in v if w != word.lower()]
    return []

# EDA Techniques
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if get_synonyms(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            # Find exact case mapping is hard, just replace lowercase match
            new_words = [synonym if word.lower() == random_word.lower() else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

def random_deletion(words, p):
    if len(words) == 1:
        return words
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]
    return new_words

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

def augment_sentence(sentence, num_aug=10):
    words = sentence.split()
    augmented_sentences = []
    num_words = len(words)
    
    if num_words == 0:
        return [sentence]
        
    n_sr = max(1, int(0.1 * num_words))
    n_rs = max(1, int(0.1 * num_words))
    p_rd = 0.1
    
    for _ in range(num_aug):
        a_words = synonym_replacement(words, n_sr)
        a_words = random_swap(a_words, n_rs)
        a_words = random_deletion(a_words, p_rd)
        augmented_sentences.append(" ".join(a_words))
        
    # Remove duplicates and original
    augmented_sentences = list(set(augmented_sentences))
    if sentence in augmented_sentences:
        augmented_sentences.remove(sentence)
        
    return augmented_sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment Fake News Data using EDA")
    parser.add_argument("--input", type=str, default="data/tingia_crawled.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default="data/augmented_fake_news.csv", help="Output CSV file")
    parser.add_argument("--num_aug", type=int, default=15, help="Number of augmented sentences per original")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading {args.input}: {e}")
        exit(1)
        
    # Tách tin giả
    fake_df = df[df['label'] == 1].copy()
    real_df = df[df['label'] == 0].copy()
    
    print(f"Original Fake News count: {len(fake_df)}")
    print(f"Original Real News count: {len(real_df)}")
    
    print(f"Generating {args.num_aug} augmented variants per fake news item...")
    augmented_data = []
    
    for _, row in fake_df.iterrows():
        text = str(row['post_message'])
        augmented_texts = augment_sentence(text, num_aug=args.num_aug)
        for aug_text in augmented_texts:
            augmented_data.append({
                'post_message': aug_text,
                'label': 1
            })
            
    aug_df = pd.DataFrame(augmented_data)
    print(f"Generated {len(aug_df)} new fake news samples.")
    
    # Gộp lại
    final_df = pd.concat([fake_df, aug_df, real_df], ignore_index=True)
    # Shuffle
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    
    final_df.to_csv(args.output, index=False)
    print(f"Total dataset saved to {args.output} with {len(final_df)} rows.")
