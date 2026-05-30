import os
import re
import sys
import time
import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

# Base Directory (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to path to allow imports from src
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "app"))

# Word segmentation helper
try:
    from pyvi import ViTokenizer, ViPosTagger
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

# Clean Vietnamese text function
def clean_vietnamese_text(text, segment_words=False):
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove usernames
    text = re.sub(r'@\w+', '', text)
    # Keep alphanumeric (including Vietnamese chars), spaces, basic punctuation
    text = re.sub(r'[^\w\s,\.\?\!\-\:\#]', '', text)
    # Standardize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    if segment_words:
        if HAS_PYVI:
            text = ViTokenizer.tokenize(text)
    return text

# Vocab Helper for LSTM
class VocabHelper:
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.unk_idx = word2idx.get("<unk>", 1)
        self.pad_idx = word2idx.get("<pad>", 0)
        
    def encode(self, text, max_len=128, segment_words=False):
        cleaned_text = clean_vietnamese_text(text, segment_words=segment_words)
        tokens = cleaned_text.split()
        idxs = [self.word2idx.get(tok, self.unk_idx) for tok in tokens]
        if len(idxs) < max_len:
            idxs = idxs + [self.pad_idx] * (max_len - len(idxs))
        else:
            idxs = idxs[:max_len]
        return torch.tensor([idxs], dtype=torch.long)

def get_lstm_token_attribution(model, vocab_helper, text, segment_words, pred_class, orig_prob, device):
    cleaned_text = clean_vietnamese_text(text, segment_words=True)
    words = []
    tags = []
    if HAS_PYVI:
        try:
            tokens, tags_list = ViPosTagger.postagging(cleaned_text)
            words = tokens
            tags = tags_list
        except Exception as ex:
            print(f"LSTM POS Tagging failed: {ex}")
            words = cleaned_text.split()
            tags = ["N"] * len(words)
    else:
        words = cleaned_text.split()
        tags = ["N"] * len(words)
        
    allowed_tags = {"N", "V", "A", "R", "Ny", "X"}
    
    # Identify valid unigram indices
    valid_unigram_indices = []
    for idx, (w, tag) in enumerate(zip(words, tags)):
        clean_w = w.replace("_", "").lower()
        if tag in allowed_tags and clean_w.isalnum() and len(clean_w) > 1:
            valid_unigram_indices.append(idx)
            
    # Generate candidates
    candidates = []
    # Add unigrams
    for idx in valid_unigram_indices:
        candidates.append({
            "type": "unigram",
            "indices": [idx],
            "text": words[idx]
        })
    # Add bigrams of adjacent valid content words
    for idx in range(len(words) - 1):
        if idx in valid_unigram_indices and (idx + 1) in valid_unigram_indices:
            candidates.append({
                "type": "bigram",
                "indices": [idx, idx+1],
                "text": f"{words[idx]}_{words[idx+1]}"
            })
            
    if not candidates:
        return []
        
    word2idx = vocab_helper.word2idx
    unk_idx = vocab_helper.unk_idx
    pad_idx = vocab_helper.pad_idx
    max_len = 128
    
    batch_list = []
    for cand in candidates:
        mutated_words = list(words)
        for idx in cand["indices"]:
            mutated_words[idx] = "<pad>"
            
        idxs = [pad_idx if tok == "<pad>" else word2idx.get(tok, unk_idx) for tok in mutated_words]
        if len(idxs) < max_len:
            idxs = idxs + [pad_idx] * (max_len - len(idxs))
        else:
            idxs = idxs[:max_len]
        batch_list.append(idxs)
        
    if not batch_list:
        return []
        
    batch_tensor = torch.tensor(batch_list, dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1)
        mutated_probs = probs[:, pred_class].cpu().numpy()
        
    # Calculate candidate attributions
    scored_candidates = []
    for i, cand in enumerate(candidates):
        score = orig_prob - mutated_probs[i]
        scored_candidates.append({
            "text": cand["text"].replace("_", " "),
            "score": float(score),
            "indices": cand["indices"]
        })
        
    # Sort candidates by score descending
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Non-Maximum Suppression (NMS) to prevent overlapping words (e.g. "bay", "màu" and "bay màu")
    selected_candidates = []
    covered_indices = set()
    for cand in scored_candidates:
        if any(idx in covered_indices for idx in cand["indices"]):
            continue
        if cand["score"] > 0.0:
            selected_candidates.append({
                "word": cand["text"],
                "score": cand["score"]
            })
            for idx in cand["indices"]:
                covered_indices.add(idx)
                
    return selected_candidates

def get_trans_token_attribution(model, tokenizer, text, segment_words, pred_class, orig_prob, device):
    cleaned_text = clean_vietnamese_text(text, segment_words=True)
    words = []
    tags = []
    if HAS_PYVI:
        try:
            tokens, tags_list = ViPosTagger.postagging(cleaned_text)
            words = tokens
            tags = tags_list
        except Exception as ex:
            print(f"Transformer POS Tagging failed: {ex}")
            words = cleaned_text.split()
            tags = ["N"] * len(words)
    else:
        words = cleaned_text.split()
        tags = ["N"] * len(words)
        
    allowed_tags = {"N", "V", "A", "R", "Ny", "X"}
    
    # Identify valid unigram indices
    valid_unigram_indices = []
    for idx, (w, tag) in enumerate(zip(words, tags)):
        clean_w = w.replace("_", "").lower()
        if tag in allowed_tags and clean_w.isalnum() and len(clean_w) > 1:
            valid_unigram_indices.append(idx)
            
    # Generate candidates
    candidates = []
    # Add unigrams
    for idx in valid_unigram_indices:
        candidates.append({
            "type": "unigram",
            "indices": [idx],
            "text": words[idx]
        })
    # Add bigrams of adjacent valid content words
    for idx in range(len(words) - 1):
        if idx in valid_unigram_indices and (idx + 1) in valid_unigram_indices:
            candidates.append({
                "type": "bigram",
                "indices": [idx, idx+1],
                "text": f"{words[idx]}_{words[idx+1]}"
            })
            
    if not candidates:
        return []
        
    batch_input_ids = []
    batch_attention_mask = []
    
    pad_token = tokenizer.pad_token if tokenizer.pad_token else "<pad>"
    for cand in candidates:
        mutated_words = list(words)
        for idx in cand["indices"]:
            mutated_words[idx] = pad_token
        mutated_text = " ".join(mutated_words)
        
        inputs = tokenizer(mutated_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        batch_input_ids.append(inputs["input_ids"][0].tolist())
        batch_attention_mask.append(inputs["attention_mask"][0].tolist())
        
    if not batch_input_ids:
        return []
        
    t_input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
    t_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits = model(input_ids=t_input_ids, attention_mask=t_attention_mask)
        probs = torch.softmax(logits, dim=1)
        mutated_probs = probs[:, pred_class].cpu().numpy()
        
    # Calculate candidate attributions
    scored_candidates = []
    for i, cand in enumerate(candidates):
        score = orig_prob - mutated_probs[i]
        scored_candidates.append({
            "text": cand["text"].replace("_", " "),
            "score": float(score),
            "indices": cand["indices"]
        })
        
    # Sort candidates by score descending
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Non-Maximum Suppression (NMS) to prevent overlapping words (e.g. "bay", "màu" and "bay màu")
    selected_candidates = []
    covered_indices = set()
    for cand in scored_candidates:
        if any(idx in covered_indices for idx in cand["indices"]):
            continue
        if cand["score"] > 0.0:
            selected_candidates.append({
                "word": cand["text"],
                "score": cand["score"]
            })
            for idx in cand["indices"]:
                covered_indices.add(idx)
                
    return selected_candidates

app = FastAPI(title="Vietnamese Fake News Detection System", version="1.0.0")

# Global variables for models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = None
lstm_vocab = None
lstm_segment = False

trans_model = None
trans_tokenizer = None
trans_name = None
trans_segment = False

# Load models on startup
@app.on_event("startup")
def startup_event():
    global lstm_model, lstm_vocab, lstm_segment
    global trans_model, trans_tokenizer, trans_name, trans_segment
    
    # 1. Load LSTM
    lstm_path = os.path.join(BASE_DIR, "models", "best_lstm.pt")
    if os.path.exists(lstm_path):
        try:
            print(f"Loading BiLSTM model from {lstm_path}...")
            checkpoint = torch.load(lstm_path, map_location=device)
            vocab_w2i = checkpoint["vocab_word2idx"]
            lstm_vocab = VocabHelper(vocab_w2i)
            
            hyperparams = checkpoint.get("hyperparameters", {})
            lstm_segment = hyperparams.get("segment_words", False)
            embedding_dim = hyperparams.get("embedding_dim", 128)
            hidden_dim = hyperparams.get("hidden_dim", 128)
            
            from src.lstm_model import BiLSTMClassifier
            lstm_model = BiLSTMClassifier(
                vocab_size=len(vocab_w2i),
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=hyperparams.get("dropout", 0.3)
            )
            lstm_model.load_state_dict(checkpoint["model_state_dict"])
            lstm_model.to(device)
            lstm_model.eval()
            print("Successfully loaded BiLSTM model.")
        except Exception as e:
            print(f"Error loading BiLSTM model: {e}")
    else:
        print(f"BiLSTM model not found at {lstm_path}.")

    # 2. Load Transformer
    trans_path = os.path.join(BASE_DIR, "models", "best_transformer_phobert.pt")
    if os.path.exists(trans_path):
        try:
            print(f"Loading Transformer model from {trans_path}...")
            checkpoint = torch.load(trans_path, map_location=device)
            
            trans_name = checkpoint.get("hyperparameters", {}).get("transformer_model_name", "vinai/phobert-base")
            print(f"Using Transformer backbone: {trans_name}")
            trans_tokenizer = AutoTokenizer.from_pretrained(trans_name)
            
            trans_segment = checkpoint.get("hyperparameters", {}).get("segment_words", None)
            if trans_segment is None:
                trans_segment = ("phobert" in trans_name.lower())
                
            from src.transformer_model import TransformerClassifier
            trans_model = TransformerClassifier(
                model_name=trans_name,
                dropout=checkpoint["hyperparameters"].get("dropout", 0.3),
                freeze_backbone=True
            )
            trans_model.load_state_dict(checkpoint["model_state_dict"])
            trans_model.to(device)
            trans_model.eval()
            print("Successfully loaded Transformer model.")
        except Exception as e:
            print(f"Error loading Transformer model: {e}")
    else:
        print(f"Transformer model not found at {trans_path}.")

class PredictionRequest(BaseModel):
    text: str
    lstm_threshold: float = 0.63
    trans_threshold: float = 0.50

class CrawlRequest(BaseModel):
    url: str

@app.post("/api/predict")
async def predict_news(req: PredictionRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Văn bản không được để trống")
        
    results = {}
    
    # Run prediction for LSTM
    if lstm_model is not None:
        try:
            t0 = time.time()
            inputs = lstm_vocab.encode(req.text, max_len=128, segment_words=lstm_segment).to(device)
            with torch.no_grad():
                logits = lstm_model(inputs)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                prob_fake = probs[1].item()
                prob_real = probs[0].item()
            latency = (time.time() - t0) * 1000  # ms
            
            is_fake = prob_fake >= req.lstm_threshold
            
            # Explainability
            keywords = []
            try:
                pred_class = 1 if is_fake else 0
                orig_prob = prob_fake if is_fake else prob_real
                attributions = get_lstm_token_attribution(
                    lstm_model, lstm_vocab, req.text, lstm_segment, pred_class, orig_prob, device
                )
                keywords = [attr for attr in attributions if attr["score"] > 0.0][:8]
            except Exception as ex:
                print(f"Error computing BiLSTM attributions: {ex}")
                
            results["lstm"] = {
                "success": True,
                "label": "TIN GIẢ (FAKE NEWS)" if is_fake else "TIN THẬT (REAL NEWS)",
                "is_fake": is_fake,
                "prob_fake": prob_fake,
                "prob_real": prob_real,
                "threshold": req.lstm_threshold,
                "latency_ms": latency,
                "segment_words": lstm_segment,
                "keywords": keywords
            }
        except Exception as e:
            results["lstm"] = {
                "success": False,
                "error": str(e)
            }
    else:
        results["lstm"] = {
            "success": False,
            "error": "Mô hình BiLSTM chưa được tải."
        }
        
    # Run prediction for Transformer
    if trans_model is not None:
        try:
            t0 = time.time()
            cleaned = clean_vietnamese_text(req.text, segment_words=trans_segment)
            inputs = trans_tokenizer(cleaned, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            with torch.no_grad():
                logits = trans_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                prob_fake = probs[1].item()
                prob_real = probs[0].item()
            latency = (time.time() - t0) * 1000  # ms
            
            is_fake = prob_fake >= req.trans_threshold
            
            # Explainability
            keywords = []
            try:
                pred_class = 1 if is_fake else 0
                orig_prob = prob_fake if is_fake else prob_real
                attributions = get_trans_token_attribution(
                    trans_model, trans_tokenizer, req.text, trans_segment, pred_class, orig_prob, device
                )
                keywords = [attr for attr in attributions if attr["score"] > 0.0][:8]
            except Exception as ex:
                print(f"Error computing Transformer attributions: {ex}")
                
            results["transformer"] = {
                "success": True,
                "label": "TIN GIẢ (FAKE NEWS)" if is_fake else "TIN THẬT (REAL NEWS)",
                "is_fake": is_fake,
                "prob_fake": prob_fake,
                "prob_real": prob_real,
                "threshold": req.trans_threshold,
                "latency_ms": latency,
                "segment_words": trans_segment,
                "backbone": trans_name,
                "keywords": keywords
            }
        except Exception as e:
            results["transformer"] = {
                "success": False,
                "error": str(e)
            }
    else:
        results["transformer"] = {
            "success": False,
            "error": "Mô hình Transformer/phoBERT chưa được tải."
        }
        
    # Clickbait keyword detector (heuristics)
    clickbait_words = ["tin sốc", "cực sốc", "tin nóng", "tin khẩn", "sự thật kinh hoàng", "chia sẻ khẩn cấp", "bí mật quốc gia", "chữa bách bệnh", "thần dược", "không thể tin nổi", "cực nóng", "khẩn cấp", "giật gân", "lộ clip", "vạch trần"]
    text_lower = req.text.lower()
    detected_keywords = [w for w in clickbait_words if w in text_lower]
    results["heuristics"] = {
        "detected_keywords": detected_keywords,
        "is_clickbait": len(detected_keywords) > 0
    }
    
    return results

@app.post("/api/crawl")
async def crawl_news(req: CrawlRequest):
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="URL không được để trống")
    
    url = req.url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="Định dạng URL không hợp lệ (phải bắt đầu bằng http:// hoặc https://)")
        
    try:
        from bs4 import BeautifulSoup
        import requests
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Không thể truy cập liên kết này (Mã lỗi: {response.status_code})")
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted boilerplate tags
        for el in soup(["script", "style", "iframe", "noscript", "nav", "footer", "header"]):
            el.extract()
            
        # Get title
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text().strip()
        else:
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text().strip()
                if " - " in title:
                    title = title.split(" - ")[0]
                elif " | " in title:
                    title = title.split(" | ")[0]
                    
        # Extract body text from paragraphs
        paragraphs = []
        article = soup.find("article")
        search_root = article if article else soup
        
        for p in search_root.find_all("p"):
            p_text = p.get_text().strip()
            if len(p_text) > 20:
                paragraphs.append(p_text)
                
        content = "\n".join(paragraphs).strip()
        
        if not content and not title:
            raise HTTPException(status_code=400, detail="Không tìm thấy nội dung văn bản trong liên kết này. Trang web có thể dùng Javascript để tải động hoặc chặn crawl.")
            
        return {
            "success": True,
            "url": url,
            "title": title,
            "content": content,
            "full_text": f"{title}\n\n{content}".strip()
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi tải trang web: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")

@app.get("/api/metrics")
async def get_metrics():
    # Load LSTM metrics
    lstm_metrics = {}
    lstm_res_path = os.path.join(BASE_DIR, "models", "test_lstm_results.json")
    if os.path.exists(lstm_res_path):
        try:
            with open(lstm_res_path) as f:
                lstm_metrics = json.load(f)
        except Exception:
            pass
            
    # Load Transformer metrics
    trans_metrics = {}
    trans_res_path = os.path.join(BASE_DIR, "models", "test_transformer_results.json")
    if os.path.exists(trans_res_path):
        try:
            with open(trans_res_path) as f:
                trans_metrics = json.load(f)
        except Exception:
            pass

    # Load Tuning Results
    tuning_results = {}
    tuning_path = os.path.join(BASE_DIR, "data", "tuning_results.json")
    if os.path.exists(tuning_path):
        try:
            with open(tuning_path) as f:
                tuning_results = json.load(f)
        except Exception:
            pass

    # Load Training History
    training_history = {}
    history_path = os.path.join(BASE_DIR, "data", "final_training_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path) as f:
                training_history = json.load(f)
        except Exception:
            pass

    return {
        "lstm_test_metrics": lstm_metrics,
        "transformer_test_metrics": trans_metrics,
        "tuning_results": tuning_results,
        "training_history": training_history,
        "has_pyvi": HAS_PYVI
    }

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    index_path = os.path.join(templates_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise HTTPException(status_code=404, detail=f"Frontend file templates/index.html not found at {index_path}.")

if __name__ == "__main__":
    import uvicorn
    current_dir = os.path.dirname(os.path.abspath(__file__))
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, app_dir=current_dir)
