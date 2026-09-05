#!/usr/bin/env python3
"""Reference: local EditLens 3B inference (CPU or CUDA).

Loads meta-llama/Llama-3.2-3B base + pangram/editlens_Llama-3.2-3B QLoRA adapter
with a NormedLinear 4-way ordinal classification head, matching Modal GPU inference.

Requirements: torch, transformers, peft, safetensors, huggingface_hub
Memory: ~6 GB RAM for fp16 on GPU, ~16 GB RAM for fp32 on CPU (~8 hours for 1k chunks)

Usage:
    export HF_TOKEN="hf_..."
    python pipeline/score_editlens_reference.py --text "Whereas the Congress finds..."
    python pipeline/score_editlens_reference.py --input pipeline/data/sample_bills.jsonl
"""
import argparse
import json
import os
import re
import sys

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors.torch import load_file
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_NAME = "pangram/editlens_Llama-3.2-3B"
BASE_NAME = "meta-llama/Llama-3.2-3B"


class NormedLinear(torch.nn.Module):
    """Custom classification head: LayerNorm → Linear (no bias)."""
    def __init__(self, h, nl, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(h, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(h, nl, bias=False, device=device, dtype=dtype)
    def forward(self, x):
        return self.linear(self.norm(x.to(self.norm.weight.dtype)))


def load_model(device="cpu"):
    """Load base model + NormedLinear head + PEFT LoRA adapter."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN required for gated model access. "
                         "Set export HF_TOKEN='hf_...'")

    # Read adapter weights to determine num_labels
    head = load_file(hf_hub_download(MODEL_NAME, "adapter_model.safetensors", token=token))
    nb = next(v.shape[0] for k, v in head.items()
              if ("score" in k or "classifier" in k) and v.ndim == 2)
    print(f"num_labels = {nb}", flush=True)

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_NAME, num_labels=nb, torch_dtype=torch_dtype,
        device_map=device, token=token)
    tok = AutoTokenizer.from_pretrained(BASE_NAME, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base.config.pad_token_id = tok.pad_token_id

    if hasattr(base, "score") and isinstance(base.score, torch.nn.Linear):
        dev = next(base.parameters()).device
        dtype = next(base.parameters()).dtype
        base.score = NormedLinear(base.config.hidden_size, nb, device=dev, dtype=dtype)

    model = PeftModel.from_pretrained(base, MODEL_NAME, token=token).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params/1e6:.0f}M params on {device}", flush=True)
    return model, tok, nb


def clean_text(text: str) -> str:
    """Text normalization matching Pangram's inference pipeline."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


@torch.no_grad()
def score_batch(texts, model, tokenizer, nb, device="cpu"):
    """Score a batch of cleaned texts. Returns list of scores in [0, 1]."""
    cleaned = [clean_text(t) for t in texts]
    enc = tokenizer(cleaned, padding=True, truncation=True,
                    max_length=1024, return_tensors="pt").to(device)
    logits = model(**enc).logits.float().cpu().numpy()
    # Softmax + ordinal expectation
    import numpy as np
    e = np.exp(logits - logits.max(1, keepdims=True))
    probs = e / e.sum(1, keepdims=True)
    bucket_ids = np.arange(nb)
    scores = (probs @ bucket_ids) / (nb - 1)
    return scores.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="Single text string to score")
    ap.add_argument("--input", help="JSONL file with bill blocks (each line needs 'text' field)")
    ap.add_argument("--device", default="cpu", help="Device: 'cpu' or 'cuda'")
    a = ap.parse_args()

    if not a.text and not a.input:
        ap.print_help()
        sys.exit(1)

    device = a.device if torch.cuda.is_available() and a.device == "cuda" else "cpu"
    model, tok, nb = load_model(device)

    if a.text:
        score = score_batch([a.text], model, tok, nb, device)
        print(f"Score: {score[0]:.4f}")

    if a.input:
        texts, meta = [], []
        for line in open(a.input, encoding="utf-8"):
            obj = json.loads(line)
            texts.append(obj.get("text", ""))
            meta.append(obj)

        batch_size = 8
        all_scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            scores = score_batch(batch, model, tok, nb, device)
            all_scores.extend(scores)
            pid = meta[i].get("package_id", meta[i].get("id", "?"))
            print(f"  [{i+1}-{i+len(batch)}/{len(texts)}] {pid}: {scores[0]:.4f}", flush=True)

        for m, sc in zip(meta, all_scores):
            m["editlens_score"] = round(sc, 5)
            print(json.dumps({"id": m.get("package_id", m.get("id")),
                              "score": round(sc, 5)}))


if __name__ == "__main__":
    main()