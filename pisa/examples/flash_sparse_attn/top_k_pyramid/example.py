#!/usr/bin/env python
"""Select blocks, run selected attention and differentiate Q/K/V."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--method", choices=("pisa1", "pisa2", "pisa"), default="pisa")
    args = parser.parse_args()

    import torch
    from pisa.ops.flash_sparse_attn.sparse_attn import sparse_attn_fn
    from pisa.ops.flash_sparse_attn.top_k_pyramid import top_k_pyramid_fn

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backend = "triton" if device.type == "cuda" else "torch"
    torch.manual_seed(0)
    q = torch.randn(1, 128, 32, 64, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, 128, 2, 64, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    # A model applies its chosen positional encoding to q/k before these calls.
    blocks = top_k_pyramid_fn(q, k, method=args.method, block_size=64,
                              topk=8, backend=backend)
    output = sparse_attn_fn(q, k, v, blocks, block_size=64, backend=backend)
    output.float().square().mean().backward()
    print(f"blocks={tuple(blocks.shape)} {blocks.dtype}")
    print(f"output={tuple(output.shape)} {output.dtype}")
    print(f"gradient shapes: q={tuple(q.grad.shape)}, k={tuple(k.grad.shape)}, v={tuple(v.grad.shape)}")


if __name__ == "__main__":
    main()
