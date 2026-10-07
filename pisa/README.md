# PISA

`pisa-attention` provides PISA-1, PISA-2, PISA, BSA and NSA operators.
The Python package is named `pisa`; operator implementations are in `pisa/ops/`.

Run the commands below from the repository root.

## Install

With PyTorch 2.7.0+cu128, Triton 3.3.0 and `fla-core==0.5.2` installed:

```bash
pip install --no-deps --no-build-isolation -e ./pisa
```

For a new training and evaluation environment, follow the root
[installation instructions](../README.md#installation).

## Usage

```python
from pisa.ops.flash_sparse_attn.top_k_pyramid import top_k_pyramid_fn
from pisa.ops.flash_sparse_attn.sparse_attn import sparse_attn_fn

# q, k, v: [batch, tokens, heads, head_dim]
blocks = top_k_pyramid_fn(
    q, k, method="pisa", block_size=64, topk=8,
    backend="auto", routing_schedule="grouped", leaf_tile=4,
)
output = sparse_attn_fn(q, k, v, blocks, block_size=64, backend="auto")
```

PISA methods are `pisa1`, `pisa2` and `pisa`. Block size and Top-K are configurable;
`force_first` and `force_recent` control forced blocks (defaults: first, previous,
current). The example uses grouped Q4 for LSE prefill. `backend="auto"` selects
Triton on CUDA and Torch on CPU; the API default is `backend="torch"`.
`routing_schedule="fused"` or `"grouped"` selects the routing schedule, and
`leaf_tile=1`, `2` or `4` sets the grouped query tile size. Automatic scheduling
uses grouped Q4 for LSE/PISA-2 prefill and fused routing for single-token decode.

A complete tensor example:

```bash
python pisa/examples/flash_sparse_attn/top_k_pyramid/example.py --device cpu
```

## Benchmark

```bash
python pisa/benchmarks/flash_sparse_attn/top_k_pyramid/benchmark.py
```

Measures full-prefill selection including summary construction, excluding attention
and backward: 4K–256K, BF16, B=1, Hq=32, Hkv=2, D=64, block size 64, Top-K=8.
Columns are LSE fused, Q1, Q2, Q4 and BSA. Times are means in milliseconds,
with 10 ms warmup and 30 ms measurement; compilation is excluded.
CSV/JSON results are written to `stat/block-selection` (existing files are replaced).
Use `--help` for lengths, dimensions, timing and output options.
This measures performance, not numerical equivalence to a reference implementation.

## Build

```bash
pip install build "setuptools>=64" wheel
python pisa/scripts/build_release.py
```

The script builds only `pisa-attention`, with wheel and source archives written
to `pisa/dist/`. From an unpacked source archive, run `python scripts/build_release.py`;
its output goes to that package's `dist/` directory. See [LICENSE](LICENSE), [NOTICE](NOTICE)
and the NSA [source attribution](pisa/ops/flash_sparse_attn/nsa_cmp_select/OFFICIAL_SOURCE.md).
