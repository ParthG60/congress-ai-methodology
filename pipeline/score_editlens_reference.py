"""
Reference: EditLens scoring logic for congressional blocks.

This replicates the exact inference from Pangram EditLens paper
(arXiv:2510.03154) for the Llama-3.2-3B model.
Not required for replication — provided for
transparency on how raw scores are computed.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "pangram/editlens_Llama-3.2-3B"
N_BUCKETS = 4  # EditLens uses a 4-way softmax head


def clean_text(text: str) -> str:
    """Text normalization matching Pangram's inference.py."""
    import re
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


@torch.no_grad()
def score_batch(texts: list[str],
                model: AutoModelForSequenceClassification,
                tokenizer: AutoTokenizer,
                device: str = "cpu") -> list[float]:
    """Score a batch of cleaned texts. Returns list of scores in [0, 1]."""
    cleaned = [clean_text(t) for t in texts]
    enc = tokenizer(cleaned, padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(device)
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    bucket_ids = torch.arange(N_BUCKETS).numpy()
    scores = (probs @ bucket_ids) / (N_BUCKETS - 1)  # in [0, 1]
    return scores.tolist()


def main():
    """Load model and tokenizer (requires HF_TOKEN with license acceptance)."""
    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN required for gated model access")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, token=token, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, token=token)
    model.eval()

    # Example: score a single block
    text = "Whereas the Congress finds that artificial intelligence..."
    score = score_batch([text], model, tokenizer)
    print(f"Score: {score[0]:.4f}")


if __name__ == "__main__":
    main()