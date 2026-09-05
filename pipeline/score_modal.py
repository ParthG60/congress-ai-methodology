#!/usr/bin/env python3
"""Score bill text chunks with EditLens 3B on a Modal L4 GPU.

Loads meta-llama/Llama-3.2-3B base + pangram/editlens_Llama-3.2-3B QLoRA adapter
with a NormedLinear 4-way ordinal classification head.

Setup (once):
    pip install modal
    modal setup
    modal secret create huggingface HF_TOKEN=<your-hf-token>
    modal volume create editlens

Upload payload:
    modal volume put editlens pipeline/data/gpu/in/all.jsonl.gz /in/all.jsonl.gz

Validate (small batch, ~$0.02):
    modal run pipeline/score_modal.py::validate

Full run (detached, survives disconnect, ~$3.50):
    modal run --detach pipeline/score_modal.py::full

Pull results:
    modal volume get editlens /out/scores_gpu_3b.jsonl pipeline/data/gpu/out/scores_gpu_3b.jsonl

Resume keys on (sha, chunk_ix) with a real score — killed runs retry in-flight work.
"""
import modal

app = modal.App("editlens-score")

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers", "peft", "bitsandbytes", "accelerate",
                      "huggingface_hub", "safetensors", "numpy"))
vol = modal.Volume.from_name("editlens", create_if_missing=True)
SECRET = modal.Secret.from_name("huggingface")

MODEL = "pangram/editlens_Llama-3.2-3B"
BASE = "meta-llama/Llama-3.2-3B"

GPU = "L4"        # 24GB Ada; fp16 3B is ~6GB — no need for 4-bit
BATCH = 32
MAXLEN = 1024
SYNC = 2000       # flush + vol.commit every N chunks


def _load_model():
    """Load base Llama-3.2-3B + NormedLinear head + PEFT LoRA adapter."""
    import os
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)
    from peft import PeftModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    token = os.environ["HF_TOKEN"]

    class NormedLinear(torch.nn.Module):
        def __init__(self, h, nl, device=None, dtype=None):
            super().__init__()
            self.norm = torch.nn.LayerNorm(h, device=device, dtype=dtype)
            self.linear = torch.nn.Linear(h, nl, bias=False, device=device, dtype=dtype)
        def forward(self, x):
            return self.linear(self.norm(x.to(self.norm.weight.dtype)))

    head = load_file(hf_hub_download(MODEL, "adapter_model.safetensors", token=token))
    nb = next(v.shape[0] for k, v in head.items()
              if ("score" in k or "classifier" in k) and v.ndim == 2)
    print(f"num_labels = {nb}", flush=True)

    base = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=nb, torch_dtype=torch.float16, device_map={"": 0}, token=token)
    tok = AutoTokenizer.from_pretrained(BASE, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base.config.pad_token_id = tok.pad_token_id

    if hasattr(base, "score") and isinstance(base.score, torch.nn.Linear):
        dev = next(base.parameters()).device
        base.score = NormedLinear(base.config.hidden_size, nb, device=dev)

    model = PeftModel.from_pretrained(base, MODEL, token=token).eval()
    print(torch.cuda.get_device_name(0), "| params",
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)
    return model, tok, nb


def _run(limit=None):
    import os, json, gzip, time
    import numpy as np
    import torch

    in_path = "/vol/in/all.jsonl.gz"
    out_path = "/vol/out/scores_gpu_3b.jsonl"
    os.makedirs("/vol/out", exist_ok=True)

    # Resume: load already-scored keys
    done = set()
    if not limit and os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("score") is not None:
                done.add((r["text_sha"], r["chunk_ix"]))

    # Read payload, skipping done and deduped
    opener = gzip.open if in_path.endswith(".gz") else open
    seen, todo = set(), []
    for line in opener(in_path, "rt", encoding="utf-8"):
        r = json.loads(line)
        k = (r["sha"], r["ix"])
        if k in done or k in seen:
            continue
        seen.add(k)
        todo.append(r)
        if limit and len(todo) >= limit:
            break
    print(f"{len(done):,} already scored | {len(todo):,} to score", flush=True)
    if not todo:
        return {"scored": 0, "already": len(done)}

    model, tok, nb = _load_model()
    ladder = np.arange(nb)
    queue = sorted(todo, key=lambda r: len(r["text"]))
    t0, n, last = time.time(), 0, 0

    out_mode = "w" if limit else "a"
    with open(out_path, out_mode, encoding="utf-8") as f:
        for s in range(0, len(queue), BATCH):
            b = queue[s:s + BATCH]
            enc = tok([r["text"] for r in b], truncation=True, max_length=MAXLEN,
                      padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**enc).logits.float().cpu().numpy()
            e = np.exp(logits - logits.max(1, keepdims=True))
            probs = e / e.sum(1, keepdims=True)
            scores = (probs @ ladder) / (nb - 1)
            buckets = probs.argmax(1)

            for r, sc, bk in zip(b, scores, buckets):
                f.write(json.dumps({
                    "text_sha": r["sha"],
                    "chunk_ix": int(r["ix"]),
                    "n_chunks": int(r["n"]),
                    "chunk_words": len(r["text"].split()),
                    "score": round(float(sc), 5),
                    "bucket": int(bk),
                }) + "\n")
            n += len(b)
            if n - last >= SYNC:
                f.flush()
                os.fsync(f.fileno())
                vol.commit()
                last = n
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n:,}/{len(queue):,}  {rate:.0f} chunk/s  "
                      f"eta {(len(queue)-n)/max(rate,1e-9)/60:.1f} min", flush=True)

    vol.commit()
    scs = [json.loads(l)["score"] for l in open(out_path, encoding="utf-8")]
    summary = {"scored": n, "mean": round(float(np.mean(scs)), 4),
               "median": round(float(np.median(scs)), 4),
               "share_gt_0.5": round(float(np.mean(np.array(scs) > 0.5)), 4),
               "minutes": round((time.time() - t0) / 60, 1)}
    print("SUMMARY", summary, flush=True)
    return summary


@app.function(image=image, gpu=GPU, timeout=600, secrets=[SECRET], volumes={"/vol": vol})
def validate():
    """Score 24 chunks (~$0.02, ~1 min) to check auth, CUDA, model loading."""
    return _run(limit=24)


@app.function(image=image, gpu=GPU, timeout=14400, secrets=[SECRET], volumes={"/vol": vol})
def full():
    """Score the full payload (~83k chunks, ~$3.50, ~3.5 hours)."""
    return _run()


@app.local_entrypoint()
def main(mode: str = "validate"):
    """Usage: modal run pipeline/score_modal.py [mode=validate|full]"""
    fn = validate if mode == "validate" else full
    print(fn.remote())