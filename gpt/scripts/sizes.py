#!/usr/bin/env python3
"""Считает параметры пресетов и произвольных конфигураций.

    python scripts/sizes.py
    python scripts/sizes.py --vocab 12288 --n-layer 8 --n-embd 512
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from minigpt.config import GPTConfig, PRESETS
from minigpt.model import GPT


def report(name: str, cfg: GPTConfig) -> None:
    model = GPT(cfg)
    total = model.count_params()
    non_emb = model.count_params(non_embedding=True)
    emb = cfg.vocab_size * cfg.n_embd
    mlp = cfg.mlp_hidden()
    print(f"{name:10s} | d={cfg.n_embd:4d} L={cfg.n_layer:3d} h={cfg.n_head:2d} "
          f"kv={cfg.n_kv_head:2d} dff={mlp:5d} V={cfg.vocab_size:6d} ctx={cfg.block_size:5d} | "
          f"всего {total/1e6:6.2f}M | без эмб. {non_emb/1e6:5.2f}M | эмб. {emb/1e6:5.2f}M")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=0)
    p.add_argument("--n-layer", type=int, default=0)
    p.add_argument("--n-embd", type=int, default=0)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-kv-head", type=int, default=4)
    p.add_argument("--mlp-mult", type=float, default=0.0)
    p.add_argument("--block-size", type=int, default=4096)
    args = p.parse_args()

    print(f"{'пресет':10s} | {'размеры':^58s} | параметры")
    for name, cfg in PRESETS.items():
        report(name, cfg)
    if args.n_layer and args.n_embd and args.vocab:
        cfg = GPTConfig(vocab_size=args.vocab, n_layer=args.n_layer, n_embd=args.n_embd,
                        n_head=args.n_head, n_kv_head=args.n_kv_head,
                        block_size=args.block_size)
        if args.mlp_mult:
            cfg.mlp_mult = args.mlp_mult
        report("custom", cfg)


if __name__ == "__main__":
    main()
