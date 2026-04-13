import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.cluster import KMeans
import faiss
import random
import ir_datasets
import os

# -----------------------------
# CONFIG (GEMMA3 ENABLED)
# -----------------------------
SEEDS = [0, 1, 2]

# 👇 point this to your LOCAL Gemma3 path
MODELS = {
    "gemma3": "/path/to/your/local/gemma3"  # <-- CHANGE THIS
}

BEIR_DATASETS = ["beir/fiqa"]

BIT_WIDTHS = [2, 4]  # focused bits (faster + meaningful)
TOP_K = 10
NUM_SAMPLES = 500
PQ_M = 4

CACHE_DIR = "./cache_embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------
# SEED
# -----------------------------
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# -----------------------------
# LOAD MODEL (Gemma-compatible)
# -----------------------------
def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True
    )

    model.eval()
    return tokenizer, model

# -----------------------------
# EMBEDDINGS (Gemma hidden states)
# -----------------------------
def get_embeddings(texts, tokenizer, model, cache_key):
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")

    if os.path.exists(cache_path):
        print(f"Loading cached embeddings: {cache_key}")
        return np.load(cache_path)

    print(f"Computing embeddings: {cache_key}")

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 👇 Use last hidden state (standard trick for LLM embeddings)
    hidden = outputs.hidden_states[-1]

    emb = hidden.mean(dim=1)
    emb = emb / emb.norm(dim=1, keepdim=True)

    emb = emb.cpu().numpy().astype('float32')

    np.save(cache_path, emb)
    return emb

# -----------------------------
# PQ
# -----------------------------
def pq_quantize(x, m, bits):
    subdim = x.shape[1] // m
    k = 2 ** bits
    codebooks, codes = [], []

    for i in range(m):
        subvecs = x[:, i*subdim:(i+1)*subdim]
        kmeans = KMeans(n_clusters=k, n_init=3, random_state=42).fit(subvecs)
        codebooks.append(kmeans.cluster_centers_)
        codes.append(kmeans.labels_)

    return codebooks, np.stack(codes, axis=1)


def pq_dequantize(codebooks, codes):
    return np.hstack([codebooks[i][codes[:, i]] for i in range(len(codebooks))])

# -----------------------------
# METHODS
# -----------------------------
def plain_pq(x, bits):
    cb, cd = pq_quantize(x, PQ_M, bits)
    return pq_dequantize(cb, cd)


def turboquant(x, bits):
    d = x.shape[1]
    R = np.linalg.qr(np.random.randn(d, d))[0]
    x_rot = x @ R
    cb, cd = pq_quantize(x_rot, PQ_M, bits)
    xq = pq_dequantize(cb, cd)
    return xq @ R.T

# -----------------------------
# METRICS
# -----------------------------
def faiss_recall(x, xq, k=10):
    d = x.shape[1]

    index_gt = faiss.IndexFlatIP(d)
    index_gt.add(x)
    _, gt = index_gt.search(x, k)

    index_q = faiss.IndexFlatIP(d)
    index_q.add(xq)
    _, pred = index_q.search(xq, k)

    return np.mean([len(set(gt[i]) & set(pred[i])) / k for i in range(len(x))])

# -----------------------------
# DATA
# -----------------------------
def load_beir_queries(name, n):
    dataset = ir_datasets.load(name)
    queries = [q.text for q in dataset.queries_iter()]
    random.shuffle(queries)
    return queries[:n]

# -----------------------------
# MAIN
# -----------------------------
def run():
    summary = []

    for dataset_name in BEIR_DATASETS:
        print(f"\n===== DATASET: {dataset_name} =====")
        texts = load_beir_queries(dataset_name, NUM_SAMPLES)

        for model_key, model_name in MODELS.items():
            print(f"\nRunning model: {model_key}")

            tokenizer, model = load_model(model_name)
            cache_key = f"{dataset_name.replace('/', '_')}_{model_key}_{NUM_SAMPLES}"

            x = get_embeddings(texts, tokenizer, model, cache_key)

            stats = {bits: {"tq": [], "pq": []} for bits in BIT_WIDTHS}

            for seed in SEEDS:
                set_seed(seed)

                for bits in BIT_WIDTHS:
                    print(f"Seed={seed}, bits={bits}")

                    xq_tq = turboquant(x, bits)
                    xq_pq = plain_pq(x, bits)

                    stats[bits]["tq"].append(faiss_recall(x, xq_tq))
                    stats[bits]["pq"].append(faiss_recall(x, xq_pq))

            for bits in BIT_WIDTHS:
                tq_mean = np.mean(stats[bits]["tq"])
                pq_mean = np.mean(stats[bits]["pq"])

                summary.append({
                    "dataset": dataset_name,
                    "model": model_key,
                    "bits": bits,
                    "tq": tq_mean,
                    "pq": pq_mean,
                    "gain": tq_mean - pq_mean
                })

    print_summary(summary)

# -----------------------------
# SUMMARY
# -----------------------------
def print_summary(summary):
    print("\n===== FINAL SUMMARY =====")
    for row in summary:
        print(f"{row['dataset']} | {row['model']} | bits={row['bits']} | TQ={row['tq']:.3f} | PQ={row['pq']:.3f} | Gain={row['gain']:.3f}")

# -----------------------------
if __name__ == "__main__":
    run()
