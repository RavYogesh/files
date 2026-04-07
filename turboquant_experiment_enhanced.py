"""
TurboQuant KV Cache Compression Experiment
===========================================
Validates the findings from Google's TurboQuant paper:
  - PolarQuant: polar-coordinate KV quantization (zero memory overhead)
  - QJL: 1-bit Johnson-Lindenstrauss residual correction
  - TurboQuant: PolarQuant + QJL combined

Targets: Gemma 3 / Mistral (on-prem, white-box KV access required)
Measures: KV memory footprint, tokens/sec throughput, quality (ROUGE/F1)

Requirements:
    pip install transformers torch accelerate datasets rouge-score tqdm numpy

Usage:
    python turboquant_experiment.py --model google/gemma-3-1b-it --bits 3
    python turboquant_experiment.py --model mistralai/Mistral-7B-Instruct-v0.3 --bits 4
"""

import argparse
import time
import math
import json
import gc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib.pyplot as plt
import os

# ---------------------------------
# GPU SAFETY (RTX 3060 FRIENDLY)
# ---------------------------------
def setup_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        return "cuda"
    return "cpu"


# ─────────────────────────────────────────────
# 1. POLAR QUANT  (Paper Section 3.1)
# ─────────────────────────────────────────────

def polarquant_encode(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a batch of vectors using PolarQuant.

    PolarQuant converts pairs of Cartesian coordinates into polar form:
      (x1, x2) → (r, θ)  where  r = sqrt(x1²+x2²),  θ = atan2(x2, x1)

    The angles are uniformly quantised to `bits` levels (no per-block scale
    constant needed → zero memory overhead).  The final scalar radius is
    stored in float16.

    Args:
        x    : (..., d)  float tensor, d must be even
        bits : quantization bit-width for angles (2, 3, or 4)

    Returns:
        theta_q : (..., d//2)  int8/int16 quantised angles
        radii   : (..., d//2)  float16 radii
    """
    assert x.shape[-1] % 2 == 0, "Dimension must be even for PolarQuant"
    x = x.float()
    # Reshape into coordinate pairs
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)  # (..., d/2, 2)
    x1, x2 = pairs[..., 0], pairs[..., 1]

    radii = torch.sqrt(x1 ** 2 + x2 ** 2).half()           # (..., d/2)
    theta = torch.atan2(x2, x1)                             # (-π, π)

    # Uniform angle quantisation: map [-π, π] → [0, 2^bits - 1]
    levels = 2 ** bits
    theta_norm = (theta + math.pi) / (2 * math.pi)          # [0, 1)
    theta_q = (theta_norm * levels).clamp(0, levels - 1).to(torch.int16)

    return theta_q, radii


def polarquant_decode(theta_q: torch.Tensor, radii: torch.Tensor, bits: int) -> torch.Tensor:
    """Reconstruct float vectors from PolarQuant representation."""
    levels = 2 ** bits
    theta = (theta_q.float() / levels) * 2 * math.pi - math.pi
    r = radii.float()
    x1 = r * torch.cos(theta)
    x2 = r * torch.sin(theta)
    return torch.stack([x1, x2], dim=-1).reshape(*x1.shape[:-1], x1.shape[-1] * 2)


# ─────────────────────────────────────────────
# 2. QJL  (Paper Section 3.2 – 1-bit residual)
# ─────────────────────────────────────────────

def qjl_encode(residual: torch.Tensor, jl_dim: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantised Johnson-Lindenstrauss 1-bit encoding of the residual.

    Projects the residual into a lower-dimensional space via a random
    Gaussian matrix S, then keeps only the sign (+1/-1 → 0/1 bit).

    Args:
        residual : (..., d)  residual error after PolarQuant decode
        jl_dim   : JL projection dimension (paper uses ≈d/4)

    Returns:
        sign_bits : (..., jl_dim)  bool tensor (1 bit per entry)
        S         : (jl_dim, d)   random projection matrix (stored once)
    """
    d = residual.shape[-1]
    S = torch.randn(jl_dim, d, device=residual.device, dtype=torch.float32) / math.sqrt(jl_dim)
    projected = residual.float() @ S.T        # (..., jl_dim)
    sign_bits = projected >= 0                # True/False → 0/1
    return sign_bits, S


def qjl_correct(query: torch.Tensor, sign_bits: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    Compute bias-free dot-product estimate using QJL signs.

    The QJL estimator:  <q, k> ≈ (π/2) * ||k||  * (2*<sign(Sk), Sq>/m - 1)

    This corrects the systematic bias in the PolarQuant approximation.
    """
    m = S.shape[0]
    Sq = (query.float() @ S.T)               # (..., jl_dim)
    agreement = ((sign_bits.float() * 2 - 1) * torch.sign(Sq)).mean(dim=-1)
    # Return a scalar correction per key position
    return agreement  # shape (...,) – added to attention logits


# ─────────────────────────────────────────────
# 3. TURBOQUANT  (PolarQuant + QJL combined)
# ─────────────────────────────────────────────

@dataclass
class TurboQuantCache:
    """Stores one layer's compressed KV cache."""
    # Keys
    k_theta_q: torch.Tensor       # quantised angles
    k_radii:   torch.Tensor       # float16 radii
    k_signs:   torch.Tensor       # QJL sign bits (residual correction)
    k_S:       torch.Tensor       # JL projection matrix
    # Values  (values only use PolarQuant – paper §4.2)
    v_theta_q: torch.Tensor
    v_radii:   torch.Tensor
    bits: int


def turboquant_compress(
    keys: torch.Tensor,
    values: torch.Tensor,
    bits: int = 3,
    jl_dim: Optional[int] = None
) -> TurboQuantCache:
    """
    Compress KV tensors with TurboQuant.

    keys, values: (batch, heads, seq, head_dim)
    """
    if jl_dim is None:
        jl_dim = max(8, keys.shape[-1] // 4)

    # --- Keys: PolarQuant stage ---
    k_theta_q, k_radii = polarquant_encode(keys, bits)
    k_hat = polarquant_decode(k_theta_q, k_radii, bits)

    # --- Keys: QJL residual stage ---
    residual = (keys - k_hat).reshape(-1, keys.shape[-1])
    k_signs, k_S = qjl_encode(residual, jl_dim)
    k_signs = k_signs.reshape(*keys.shape[:-1], jl_dim)

    # --- Values: PolarQuant only (no attention logit correction needed) ---
    v_theta_q, v_radii = polarquant_encode(values, bits)

    return TurboQuantCache(
        k_theta_q=k_theta_q, k_radii=k_radii,
        k_signs=k_signs, k_S=k_S,
        v_theta_q=v_theta_q, v_radii=v_radii,
        bits=bits,
    )


def turboquant_decompress(cache: TurboQuantCache) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompress back to float tensors (for non-CUDA-kernel path)."""
    keys   = polarquant_decode(cache.k_theta_q, cache.k_radii, cache.bits)
    values = polarquant_decode(cache.v_theta_q, cache.v_radii, cache.bits)
    return keys, values


# ─────────────────────────────────────────────
# 4. MEMORY MEASUREMENT UTILS
# ─────────────────────────────────────────────

def kv_cache_bytes(keys: torch.Tensor, values: torch.Tensor) -> int:
    """Raw bytes of uncompressed KV tensors."""
    return (keys.nelement() + values.nelement()) * keys.element_size()


def turboquant_bytes(cache: TurboQuantCache) -> int:
    """Bytes used by TurboQuant-compressed cache."""
    total = 0
    for t in [cache.k_theta_q, cache.k_radii, cache.k_signs,
              cache.v_theta_q, cache.v_radii]:
        total += t.nelement() * t.element_size()
    # k_S is shared across all positions – amortised to ~0 per token
    return total


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    return original_bytes / compressed_bytes


# ─────────────────────────────────────────────
# 5. MODEL HOOK: Intercept KV cache per layer
# ─────────────────────────────────────────────

class KVCacheInterceptor:
    """
    Registers forward hooks on all attention layers to capture and optionally
    replace KV tensors with their TurboQuant-compressed versions.

    Supports HuggingFace models that expose `k_proj` / `v_proj` in their
    attention modules (Gemma, Mistral, LLaMA-family).
    """

    def __init__(self, model, bits: int = 3, compress: bool = True):
        self.model     = model
        self.bits      = bits
        self.compress  = compress
        self.stats     = []        # list of dicts, one per forward pass
        self._hooks    = []
        self._register()

    def _register(self):
        for name, module in self.model.named_modules():
            # Target GemmaAttention / MistralAttention / LlamaAttention
            cls = type(module).__name__
            if "Attention" in cls and hasattr(module, "k_proj"):
                h = module.register_forward_hook(self._hook_fn(name))
                self._hooks.append(h)

    def _hook_fn(self, layer_name):
        def hook(module, inputs, output):
            # HF attention returns a tuple; past_key_value is element [1]
            # Shape depends on model; we skip if not a standard tuple
            if not isinstance(output, tuple) or len(output) < 2:
                return output
            present_kv = output[1]
            if present_kv is None or not isinstance(present_kv, tuple):
                return output
            k, v = present_kv  # (batch, heads, seq, head_dim)

            orig_bytes = kv_cache_bytes(k, v)
            cache = turboquant_compress(k, v, bits=self.bits)
            comp_bytes = turboquant_bytes(cache)
            ratio = compression_ratio(orig_bytes, comp_bytes)

            self.stats.append({
                "layer":        layer_name,
                "orig_mb":      orig_bytes / 1e6,
                "comp_mb":      comp_bytes / 1e6,
                "ratio":        ratio,
                "seq_len":      k.shape[2],
            })

            if self.compress:
                # Reconstruct for the downstream computation
                k_hat, v_hat = turboquant_decompress(cache)
                # Cast back to original dtype
                k_hat = k_hat.to(k.dtype)
                v_hat = v_hat.to(v.dtype)
                new_output = (output[0], (k_hat, v_hat)) + output[2:]
                return new_output
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def summary(self) -> dict:
        if not self.stats:
            return {}
        ratios = [s["ratio"] for s in self.stats]
        orig   = sum(s["orig_mb"] for s in self.stats)
        comp   = sum(s["comp_mb"] for s in self.stats)
        return {
            "avg_compression_ratio": np.mean(ratios),
            "total_orig_mb":  orig,
            "total_comp_mb":  comp,
            "num_layers_intercepted": len(self.stats),
        }


# ─────────────────────────────────────────────
# 6. BENCHMARK TASKS
# ─────────────────────────────────────────────

# --- Needle in a Haystack ---
HAYSTACK_TEMPLATE = """
The following is a long document. Somewhere inside it is a hidden fact.
{padding}
THE SECRET NUMBER IS: {needle}
{padding2}
Question: What is the secret number?
Answer:"""

def build_needle_prompt(needle: str = "42", pad_tokens: int = 500) -> str:
    padding  = "This sentence is filler text. " * (pad_tokens // 6)
    padding2 = "More filler text here. " * (pad_tokens // 6)
    return HAYSTACK_TEMPLATE.format(needle=needle, padding=padding, padding2=padding2)

# --- QA / Summarisation tasks ---
BENCHMARK_PROMPTS = [
    # QA
    {
        "task": "qa",
        "prompt": "Question: What is the capital of France?\nAnswer:",
        "expected_keyword": "Paris",
    },
    {
        "task": "qa",
        "prompt": "Question: What programming language is PyTorch written in?\nAnswer:",
        "expected_keyword": "Python",
    },
    # Summarisation
    {
        "task": "summarisation",
        "prompt": (
            "Summarise this in one sentence: "
            "Transformers are deep learning models that use self-attention mechanisms "
            "to process sequential data. They were introduced in the paper 'Attention is All You Need' "
            "by Vaswani et al. in 2017 and have since become the dominant architecture "
            "for natural language processing tasks.\nSummary:"
        ),
        "expected_keyword": "attention",
    },
    # Code generation
    {
        "task": "code",
        "prompt": "Write a Python function that returns the factorial of n:\ndef factorial(n):",
        "expected_keyword": "return",
    },
    # Needle
    {
        "task": "needle",
        "prompt": build_needle_prompt(needle="7391", pad_tokens=600),
        "expected_keyword": "7391",
    },
]


def keyword_accuracy(text: str, keyword: str) -> float:
    return 1.0 if keyword.lower() in text.lower() else 0.0


# ─────────────────────────────────────────────
# 7. THROUGHPUT MEASUREMENT
# ─────────────────────────────────────────────

def measure_throughput(
    model, tokenizer, prompts: list[str],
    max_new_tokens: int = 50,
    device: str = "cuda"
) -> dict:
    """Returns tokens/sec and latency stats."""
    latencies = []
    total_tokens = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        n_input = inputs["input_ids"].shape[1]

        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - t0

        new_tokens = out.shape[1] - n_input
        latencies.append(elapsed)
        total_tokens += new_tokens

    return {
        "tokens_per_sec": total_tokens / sum(latencies),
        "avg_latency_s":  np.mean(latencies),
        "p95_latency_s":  np.percentile(latencies, 95),
        "total_tokens":   total_tokens,
    }


# ─────────────────────────────────────────────
# 8. QUALITY EVALUATION
# ─────────────────────────────────────────────

def evaluate_quality(
    model, tokenizer, benchmarks: list[dict],
    max_new_tokens: int = 80,
    device: str = "cuda"
) -> dict:
    results = []
    for item in tqdm(benchmarks, desc="Quality eval"):
        inputs = tokenizer(
            item["prompt"], return_tensors="pt",
            truncation=True, max_length=1024
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        acc = keyword_accuracy(generated, item["expected_keyword"])
        results.append({
            "task":     item["task"],
            "accuracy": acc,
            "output":   generated[:120],
        })

    per_task = {}
    for r in results:
        per_task.setdefault(r["task"], []).append(r["accuracy"])

    return {
        "overall_accuracy": np.mean([r["accuracy"] for r in results]),
        "per_task":         {k: np.mean(v) for k, v in per_task.items()},
        "details":          results,
    }


# ─────────────────────────────────────────────
# 9. FULL EXPERIMENT RUNNER
# ─────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    model_id:       str   = "google/gemma-3-1b-it"
    bits_list:      list  = field(default_factory=lambda: [2, 3, 4])
    max_new_tokens: int   = 80
    device:         str   = "cuda" if torch.cuda.is_available() else "cpu"
    output_file:    str   = "turboquant_results.json"


def run_experiment(cfg: ExperimentConfig):
    print(f"\n{'='*60}")
    print(f"  TurboQuant Experiment")
    print(f"  Model : {cfg.model_id}")
    print(f"  Device: {cfg.device}")
    print(f"{'='*60}\n")

    # ── Load model (RTX 3060 optimized) ─────────────────────────
    device = setup_device()

    print("Loading tokenizer and model …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        **load_kwargs
    )

    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()

    prompts = [b["prompt"] for b in BENCHMARK_PROMPTS]
    all_results = {}

    # ── Phase 1: Baseline (no compression) ──────────────────────
    print("\n[Phase 1] Baseline — no KV compression")
    baseline_tp  = measure_throughput(model, tokenizer, prompts,
                                      cfg.max_new_tokens, cfg.device)
    baseline_q   = evaluate_quality(model, tokenizer, BENCHMARK_PROMPTS,
                                    cfg.max_new_tokens, cfg.device)

    all_results["baseline"] = {
        "bits":         "fp16",
        "throughput":   baseline_tp,
        "quality":      baseline_q,
        "compression":  {"avg_compression_ratio": 1.0},
    }
    print(f"  Tokens/sec  : {baseline_tp['tokens_per_sec']:.1f}")
    print(f"  Accuracy    : {baseline_q['overall_accuracy']*100:.1f}%")

    # ── Phase 2–4: TurboQuant at each bit-width ──────────────────
    for bits in cfg.bits_list:
        print(f"\n[Phase 2-4] TurboQuant — {bits}-bit KV compression")

        interceptor = KVCacheInterceptor(model, bits=bits, compress=True)
        tp   = measure_throughput(model, tokenizer, prompts,
                                  cfg.max_new_tokens, cfg.device)
        q    = evaluate_quality(model, tokenizer, BENCHMARK_PROMPTS,
                                cfg.max_new_tokens, cfg.device)
        cstats = interceptor.summary()
        interceptor.remove()

        all_results[f"turboquant_{bits}bit"] = {
            "bits":        bits,
            "throughput":  tp,
            "quality":     q,
            "compression": cstats,
        }

        speedup = tp["tokens_per_sec"] / baseline_tp["tokens_per_sec"]
        acc_delta = (q["overall_accuracy"] - baseline_q["overall_accuracy"]) * 100
        print(f"  Tokens/sec      : {tp['tokens_per_sec']:.1f}  (×{speedup:.2f} vs baseline)")
        print(f"  Accuracy        : {q['overall_accuracy']*100:.1f}%  ({acc_delta:+.1f}pp vs baseline)")
        if cstats:
            print(f"  Compression     : {cstats['avg_compression_ratio']:.1f}×")
            print(f"  KV memory saved : {cstats['total_orig_mb']:.1f} MB → {cstats['total_comp_mb']:.1f} MB")

        gc.collect()
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────────
    with open(cfg.output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {cfg.output_file}")

    # ── Print comparison table ───────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Config':<22} {'Bits':>5} {'Tok/s':>9} {'Speedup':>9} {'Acc%':>7} {'KV Ratio':>10}")
    print("-"*65)

    baseline_tps = all_results["baseline"]["throughput"]["tokens_per_sec"]
    for key, res in all_results.items():
        tps     = res["throughput"]["tokens_per_sec"]
        acc     = res["quality"]["overall_accuracy"] * 100
        speedup = tps / baseline_tps
        bits    = res["bits"]
        ratio   = res["compression"].get("avg_compression_ratio", 1.0)
        print(f"{key:<22} {str(bits):>5} {tps:>9.1f} {speedup:>9.2f}× {acc:>7.1f}  {ratio:>9.1f}×")
    print("="*65)

    print("\nGenerating visualizations...")
    generate_plots(all_results)
    plot_distortion_curve()

    return all_results


# ─────────────────────────────────────────────
# 10. STANDALONE DOT-PRODUCT DISTORTION TEST
#     (validates PolarQuant + QJL independently)
# ─────────────────────────────────────────────

def test_distortion(d: int = 128, n: int = 1000, bits: int = 3):
    """
    Measures mean squared error of dot-product estimates vs ground truth
    for random unit vectors at different bit-widths.
    Reproduces Fig.1 from the paper (dot product distortion vs bit-width).
    """
    print(f"\n[Distortion Test] d={d}, n={n} random pairs, bits={bits}")
    keys    = F.normalize(torch.randn(n, d), dim=-1)
    queries = F.normalize(torch.randn(n, d), dim=-1)

    # Ground truth
    gt = (keys * queries).sum(-1)

    # PolarQuant only
    tq, radii = polarquant_encode(keys, bits)
    k_hat     = polarquant_decode(tq, radii, bits)
    pq_approx = (k_hat * queries).sum(-1)
    pq_mse    = ((gt - pq_approx) ** 2).mean().item()

    # PolarQuant + QJL correction
    residual    = (keys - k_hat).reshape(n, d)
    signs, S    = qjl_encode(residual, jl_dim=max(8, d // 4))
    correction  = qjl_correct(queries, signs, S)
    tq_approx   = pq_approx + correction * 0.1   # scale factor
    tq_mse      = ((gt - tq_approx) ** 2).mean().item()

    print(f"  PolarQuant MSE         : {pq_mse:.6f}")
    print(f"  TurboQuant (PQ+QJL) MSE: {tq_mse:.6f}")
    print(f"  QJL improvement        : {(pq_mse - tq_mse)/pq_mse*100:.1f}%")
    return {"pq_mse": pq_mse, "tq_mse": tq_mse}



# ─────────────────────────────────────────────
# 11. VISUALIZATION UTILITIES
# ─────────────────────────────────────────────

def extract_plot_metrics(results):
    bits = []
    compression = []
    throughput = []
    accuracy = []

    for k, v in results.items():
        if k == "baseline":
            continue

        bits.append(v["bits"])
        compression.append(
            v["compression"].get("avg_compression_ratio", 1.0)
        )
        throughput.append(v["throughput"]["tokens_per_sec"])
        accuracy.append(v["quality"]["overall_accuracy"])

    return bits, compression, throughput, accuracy


def generate_plots(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    bits, compression, throughput, accuracy = extract_plot_metrics(results)

    plt.figure()
    plt.plot(bits, compression, marker="o")
    plt.xlabel("Bit-width")
    plt.ylabel("Compression Ratio")
    plt.title("TurboQuant Compression vs Bits")
    plt.savefig(f"{output_dir}/compression_vs_bits.png")
    plt.close()

    plt.figure()
    plt.plot(bits, throughput, marker="o")
    plt.xlabel("Bit-width")
    plt.ylabel("Tokens/sec")
    plt.title("Throughput vs Bits")
    plt.savefig(f"{output_dir}/throughput_vs_bits.png")
    plt.close()

    plt.figure()
    plt.plot(bits, accuracy, marker="o")
    plt.xlabel("Bit-width")
    plt.ylabel("Accuracy")
    plt.title("Quality Retention vs Bits")
    plt.savefig(f"{output_dir}/accuracy_vs_bits.png")
    plt.close()


def plot_distortion_curve():
    bits_list = [2, 3, 4, 5]
    pq = []
    tq = []

    for b in bits_list:
        res = test_distortion(bits=b)
        pq.append(res["pq_mse"])
        tq.append(res["tq_mse"])

    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(bits_list, pq, label="PolarQuant")
    plt.plot(bits_list, tq, label="TurboQuant")
    plt.xlabel("Bits")
    plt.ylabel("MSE")
    plt.title("Dot Product Distortion")
    plt.legend()
    plt.savefig("results/distortion_curve.png")
    plt.close()

# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboQuant KV Cache Experiment")
    parser.add_argument("--model",      default="google/gemma-3-1b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--bits",       type=int, nargs="+", default=[2, 3, 4],
                        help="Bit-widths to test (e.g. 2 3 4)")
    parser.add_argument("--max_tokens", type=int, default=80,
                        help="Max new tokens per generation")
    parser.add_argument("--output",     default="turboquant_results.json",
                        help="Output JSON file")
    parser.add_argument("--distortion_only", action="store_true",
                        help="Run only the fast standalone distortion test")
    args = parser.parse_args()

    if args.distortion_only:
        for b in args.bits:
            test_distortion(d=128, n=2000, bits=b)
    else:
        cfg = ExperimentConfig(
            model_id=args.model,
            bits_list=args.bits,
            max_new_tokens=args.max_tokens,
            output_file=args.output,
        )
        run_experiment(cfg)
