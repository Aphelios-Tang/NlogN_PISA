# Public NSA compression source

Source: https://github.com/XunhaoLai/native-sparse-attention-triton

Fixed commit: `9bea856c911ebf263be88d797fb28458f82f1d94`.
Copyright 2025 Xunhao Lai & Jianqiao Lu. Apache-2.0; the unmodified license
is included as `LICENSE` beside these files.

The following files were obtained from that public commit:

| Local file | Upstream path | Original SHA-256 |
| --- | --- | --- |
| weighted_pool_triton.py | native_sparse_attention/ops/triton/weighted_pool.py | a20d033af6072dfff2898d048d3d2bd46f357da67e5762c77bcd77dfe0374735 |
| compressed_attention_triton.py | native_sparse_attention/ops/triton/compressed_attention.py | c64a8eededfaa797d39850e705f7b22b0a12fefb82973065f9b581608d927607 |
| utils.py | native_sparse_attention/ops/triton/utils.py | 0ab9aff5a1e4fed9ec131bd3b778d22ccb11a49f51c1e63f4340d3b89f6c187e |

Local changes: utility imports are relative, and executable self-test examples
are omitted. `transform_score` constructs its small integer overlap histogram
with CPU `histc` and transfers the resulting FP32 counts to the score device.
For size/stride/block 32/16/64 these counts are exactly `[1, 2, 2, 2, 1]`.
This avoids PyTorch's deterministic-algorithms rejection of CUDA `histc` without
changing attention arithmetic, score transformation, topk, or kernel launches.

The adapter in `nsa_cmp_select_triton.py` packs full prefill inputs and calls these
kernels once across the complete batch. It applies the caller's RoPE to pooled
raw K, then uses the installed FLA selected-attention primitive. Its Dynamo
boundary preserves data-dependent packed compression lengths. FP32 master
compression weights are explicitly cast to the BF16 activation dtype, and
autograd returns gradients to the original parameters. BF16 reference checks
use identical BF16 weights on both sides and require no such cast.

Canonical dispatch covers BF16 CUDA prefill, Hq=32/Hkv=2, D=64 or 128,
compression size/stride 32/16, selection block 64, K=8/16/32, forced first and
current blocks with optional previous block. D128 requires at least 163840
bytes of opt-in shared memory for the unchanged public launch. Unsupported
layouts, lower-resource D128 devices, explicit Torch, and suffix/decode use
the portable implementation; this source does not claim decode alignment.
