#!/usr/bin/env python3
"""Готовит данные: учит BPE-токенайзер и пакует корпус в train.bin/val.bin.

    python scripts/prepare_data.py --input data/raw --out data/prepared --num-merges 4000

--input может быть файлом или папкой (.txt/.md, рекурсивно). Документы
разделяются пустой строкой или границей файла.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from minigpt.data import token_dtype, write_bin
from minigpt.tokenizer import BPETokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="файл или папка с текстами")
    p.add_argument("--out", default="data/prepared")
    p.add_argument("--num-merges", type=int, default=4000,
                   help="число BPE-мерджей; словарь = 264 + num-merges")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--max-train-chars", type=int, default=3_000_000,
                   help="сколько символов взять на обучение токенайзера (для скорости)")
    p.add_argument("--doc-sep", default="\n\n", help="разделитель документов внутри файла")
    return p.parse_args()


def collect_docs(path: str, doc_sep: str) -> List[str]:
    files = []
    if os.path.isdir(path):
        for root, _, names in os.walk(path):
            for n in sorted(names):
                if n.endswith((".txt", ".md", ".jsonl")) and not n.endswith(".bin"):
                    files.append(os.path.join(root, n))
    elif os.path.isfile(path):
        files = [path]
    else:
        raise SystemExit(f"не найден {path}")
    docs: List[str] = []
    for f in sorted(files):
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        parts = [p.strip() for p in text.split(doc_sep)]
        docs.extend(p for p in parts if p)
    if not docs:
        raise SystemExit("не удалось найти ни одного документа")
    return docs


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    docs = collect_docs(args.input, args.doc_sep)
    total_chars = sum(len(d) for d in docs)
    print(f"документов: {len(docs)}, символов: {total_chars:,}")

    # 1) токенайзер — на случайной выборке, чтобы не ждать на больших корпусах
    sample, taken = [], 0
    order = np.random.RandomState(0).permutation(len(docs))
    for i in order:
        sample.append(docs[int(i)])
        taken += len(docs[int(i)])
        if taken >= args.max_train_chars:
            break
    t0 = time.time()
    print(f"учим токенайзер на {taken:,} символах...")
    tok = BPETokenizer.train(sample, args.num_merges)
    print(f"  время: {time.time()-t0:.1f} c, словарь: {tok.vocab_size}")
    tok.save(os.path.join(args.out, "tokenizer.json"))

    # 2) кодируем всё и режем на train/val
    print("кодируем корпус...")
    t0 = time.time()
    all_ids: List[int] = []
    for d in docs:
        all_ids.extend(tok.encode(d))
        all_ids.append(2)  # EOS_ID
    print(f"  токенов: {len(all_ids):,} ({total_chars/max(1,len(all_ids)):.2f} симв/токен) "
          f"за {time.time()-t0:.1f} c")

    split = int(len(all_ids) * (1 - args.val_frac)) if args.val_frac > 0 else len(all_ids)
    # граница — по документу: ищем ближайший EOS
    write_bin(all_ids[:split], os.path.join(args.out, "train.bin"), tok.vocab_size)
    if split < len(all_ids):
        write_bin(all_ids[split:], os.path.join(args.out, "val.bin"), tok.vocab_size)

    meta = {
        "vocab_size": tok.vocab_size, "num_merges": args.num_merges,
        "train_tokens": split, "val_tokens": len(all_ids) - split,
        "chars": total_chars, "docs": len(docs),
        "chars_per_token": total_chars / max(1, len(all_ids)),
        "dtype": token_dtype(tok.vocab_size).__name__,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"готово -> {args.out}")


if __name__ == "__main__":
    main()
