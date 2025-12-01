import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Hugging Face Transformers [web:42][web:43][web:46][web:52][web:58]

_FINBERT_MODEL_NAME = "ProsusAI/finbert"  # financial sentiment model on Hugging Face [web:42][web:43][web:52][web:58]

class FinBertSentiment:
    def __init__(self, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL_NAME)  # [web:42][web:43][web:52]
        self.model = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL_NAME)  # [web:42][web:43][web:52]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {v: k for k, v in self.model.config.label2id.items()}

    def score_texts(self, texts, batch_size=16):
        scores = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                for j, p in enumerate(preds):
                    label = self.id2label[int(p)]
                    prob = float(probs[j, p].cpu().item())
                    # Map labels to numeric sentiment
                    if label.lower().startswith("positive"):
                        score = prob
                    elif label.lower().startswith("negative"):
                        score = -prob
                    else:
                        score = 0.0
                    scores.append(
                        {
                            "label": label,
                            "confidence": prob,
                            "sentiment_score": score,
                        }
                    )
        return scores

def add_sentiment_to_news(df_news):
    if df_news.empty:
        return df_news
    finbert = FinBertSentiment()
    texts = (df_news["title"].fillna("") + ". " + df_news["description"].fillna("")).tolist()
    results = finbert.score_texts(texts)
    df_sent = df_news.copy()
    df_sent["sentiment_label"] = [r["label"] for r in results]
    df_sent["sentiment_conf"] = [r["confidence"] for r in results]
    df_sent["sentiment_score"] = [r["sentiment_score"] for r in results]
    return df_sent
