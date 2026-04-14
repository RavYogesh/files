import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import random

# =============================
# CONFIG
# =============================
MODEL_PATH = "/path/to/your/local/gemma"  # ← CHANGE THIS
NUM_SAMPLES = 500
BITS = 2
PQ_M = 8
SEED = 42

np.random.seed(SEED)
random.seed(SEED)

# =============================
# LOAD MODEL
# =============================
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model

# =============================
# GET EMBEDDINGS
# =============================
def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.model(**inputs)  # important for Gemma

    emb = outputs.last_hidden_state.mean(dim=1)
    emb = emb / emb.norm(dim=1, keepdim=True)

    return emb.cpu().numpy().astype('float32')

# =============================
# FAISS PQ
# =============================
def pq_quantize(x):
    d = x.shape[1]
    pq = faiss.ProductQuantizer(d, PQ_M, BITS)
    pq.train(x)
    codes = pq.compute_codes(x)
    return pq.decode(codes)

# =============================
# TURBOQUANT
# =============================
def turboquant(x):
    d = x.shape[1]

    # random rotation
    R = np.linalg.qr(np.random.randn(d, d))[0].astype('float32')

    x_rot = x @ R
    xq_rot = pq_quantize(x_rot)

    return xq_rot @ R.T

# =============================
# RECALL@K
# =============================
def recall(x, xq, k=10):
    d = x.shape[1]

    index_gt = faiss.IndexFlatIP(d)
    index_gt.add(x)
    _, gt = index_gt.search(x, k)

    index_q = faiss.IndexFlatIP(d)
    index_q.add(xq)
    _, pred = index_q.search(xq, k)

    return np.mean([
        len(set(gt[i]) & set(pred[i])) / k
        for i in range(len(x))
    ])

# =============================
# DATA (dummy – replace if needed)
# =============================
def load_data(n):
    return [f"Sample query {i}" for i in range(n)]

# =============================
# MAIN
# =============================
def run():
    tokenizer, model = load_model()

    texts = load_data(NUM_SAMPLES)
    print("Generating embeddings...")
    X = get_embeddings(texts, tokenizer, model)

    print("Running PQ...")
    X_pq = pq_quantize(X)

    print("Running TurboQuant...")
    X_tq = turboquant(X)

    r_pq = recall(X, X_pq)
    r_tq = recall(X, X_tq)

    print("\n===== RESULTS =====")
    print(f"PQ Recall@10 : {r_pq:.4f}")
    print(f"TQ Recall@10 : {r_tq:.4f}")
    print(f"Gain         : {r_tq - r_pq:.4f}")

# =============================
if __name__ == "__main__":
    run()
