#!/usr/bin/env python
"""Full-prefill block selection timings in milliseconds."""

import argparse
import csv
from functools import partial
import json
import math
from pathlib import Path
import sys


PROVIDERS = {
    "fused": "Direct-LSE fused",
    "q1": "Direct-LSE Q1",
    "q2": "Direct-LSE Q2",
    "q4": "Direct-LSE Q4",
    "bsa": "BSA Top-K",
}
LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]


def make_call(q, k, provider):
    from pisa.ops.flash_sparse_attn.bsa import bsa_fn
    from pisa.ops.flash_sparse_attn.top_k_pyramid import top_k_pyramid_fn

    if provider == "bsa":
        return partial(bsa_fn, q, k, block_size=64, topk=8, backend="triton")
    return partial(
        top_k_pyramid_fn, q, k, method="pisa", block_size=64, topk=8,
        backend="triton", force_first=True, force_recent=2,
        routing_schedule="fused" if provider == "fused" else "grouped",
        leaf_tile=4 if provider == "fused" else int(provider[1:]),
    )


def length_label(length):
    return f"{length // 1024}K" if length % 1024 == 0 else str(length)


def print_header(dimension, providers):
    print(f"\nblock-selection-d{dimension} (ms):", flush=True)
    print(f"{'n':>10}" + "".join(f"  {PROVIDERS[p]:>16}" for p in providers), flush=True)


def print_row(length, values):
    cells = [f"{v:.6f}" if isinstance(v, float) else v for v in values]
    print(f"{length:10d}" + "".join(f"  {v:>16}" for v in cells), flush=True)


def save_results(directory, metadata, rows, providers):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "selection.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n"
    )
    with (directory / "selection.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["head_dim", "length"] + [PROVIDERS[p] + " (ms)" for p in providers])
        for row in rows:
            writer.writerow([row["head_dim"], row["length"]] + [row["timings_ms"][p] for p in providers])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--head-dims", type=int, nargs="+", choices=[64, 128], default=[64])
    parser.add_argument("--providers", nargs="+", choices=list(PROVIDERS), default=list(PROVIDERS))
    parser.add_argument("--warmup-ms", type=float, default=10)
    parser.add_argument("--rep-ms", type=float, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("stat/block-selection"),
                        help="CSV and JSON destination (default: stat/block-selection).")
    args = parser.parse_args()
    if min(args.lengths) < 1 or not math.isfinite(args.warmup_ms) or args.warmup_ms < 0:
        parser.error("lengths must be positive and warmup-ms must be finite and nonnegative")
    if not math.isfinite(args.rep_ms) or args.rep_ms <= 0:
        parser.error("rep-ms must be finite and positive")

    import torch
    import triton
    import triton.testing

    if not torch.cuda.is_available():
        parser.error("an allocated CUDA GPU is required")
    metadata = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "triton": triton.__version__,
        "batch_size": 1, "query_heads": 32, "key_heads": 2,
        "dtype": "bfloat16", "block_size": 64, "topk": 8, "seed": 2024,
        "force_first": True, "force_recent": 2,
        "lengths": args.lengths, "head_dims": args.head_dims, "providers": args.providers,
        "warmup_ms": args.warmup_ms, "rep_ms": args.rep_ms,
        "timer": "triton.testing.do_bench, mean, L2 flush between samples",
        "paper": "https://arxiv.org/abs/2609.31093, Table 4",
        "scope": "fresh summaries and full-prefill selection; no attention or backward",
    }
    print(f"GPU: {metadata['gpu']} | BF16 | B=1, Hq=32, Hkv=2, C=64, K=8", flush=True)
    print(f"Torch={torch.__version__} | Triton={triton.__version__} | CUDA={torch.version.cuda}", flush=True)
    print("Seed=2024 | forced blocks: first, previous, current | summaries included", flush=True)
    print(f"Mean CUDA time | warmup={args.warmup_ms:g} ms, measurement={args.rep_ms:g} ms", flush=True)
    rows = []
    failed = False
    case = 0
    total = len(args.head_dims) * len(args.lengths) * len(args.providers)
    with torch.no_grad():
        for dimension in args.head_dims:
            print_header(dimension, args.providers)
            for length in args.lengths:
                row = {"head_dim": dimension, "length": length, "timings_ms": {}, "errors": {}}
                q = k = None
                try:
                    torch.manual_seed(2024)
                    q = torch.randn(1, length, 32, dimension, device="cuda", dtype=torch.bfloat16)
                    k = torch.randn(1, length, 2, dimension, device="cuda", dtype=torch.bfloat16)
                except torch.OutOfMemoryError as error:
                    row["errors"]["inputs"] = str(error)
                for provider in args.providers:
                    case += 1
                    print(f"[{case}/{total}] D={dimension} N={length_label(length)} {PROVIDERS[provider]}",
                          file=sys.stderr, flush=True)
                    run = None
                    try:
                        if "inputs" in row["errors"]:
                            raise torch.OutOfMemoryError(row["errors"]["inputs"])
                        run = make_call(q, k, provider)
                        # Compile before do_bench estimates the repetition count.
                        run()
                        torch.cuda.synchronize()
                        value = float(triton.testing.do_bench(
                            run, warmup=args.warmup_ms, rep=args.rep_ms, return_mode="mean"
                        ))
                        if not math.isfinite(value) or value <= 0:
                            raise ValueError(f"invalid elapsed time: {value}")
                        row["timings_ms"][provider] = value
                    except Exception as error:
                        failed = True
                        row["timings_ms"][provider] = "OOM" if isinstance(error, torch.OutOfMemoryError) else "ERROR"
                        row["errors"][provider] = str(error)
                        print(f"  {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                    finally:
                        run = None
                    torch.cuda.empty_cache()
                print_row(length, list(row["timings_ms"].values()))
                rows.append(row)
                if args.output_dir:
                    save_results(args.output_dir, metadata, rows, args.providers)
                del q, k
                torch.cuda.empty_cache()
    print(f"\nResults: {args.output_dir.resolve()}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
