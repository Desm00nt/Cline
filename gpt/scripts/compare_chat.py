"""Сравнивает несколько чекпойнтов на одних и тех же репликах.

Greedy-режим убирает шум сэмплинга, поэтому разница между чекпойнтами
видна на одних и тех же вопросах. Заодно печатает скорость генерации.

    python scripts/compare_chat.py --tokenizer weights/tokenizer.json \
        weights/small-21k-chat100.fp16.pt out/sft5/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from minigpt.model import load_any
from minigpt.sampling import chat_once
from minigpt.tokenizer import BPETokenizer

QUESTIONS = [
    "Привет!", "Как тебя зовут?", "Что ты умеешь?", "Сколько у тебя параметров?",
    "Что такое токен?", "Ты помнишь прошлый разговор?", "Почему небо голубое?",
    "Спасибо!", "Расскажи шутку", "Что ты читаешь?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="weights/tokenizer.json")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("ckpts", nargs="+")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tok = BPETokenizer.load(args.tokenizer)

    for path in args.ckpts:
        model, payload = load_any(path, "cpu")
        model.eval()
        print(f"\n{'=' * 72}\n{path}   (step {payload.get('step')}, "
              f"val {payload.get('val_loss')})\n{'=' * 72}")
        t0, ntok = time.time(), 0
        for q in QUESTIONS:
            msgs = [{"role": "user", "ids": tok.encode(q)}]
            with torch.no_grad():
                ids = chat_once(model, msgs, max_new_tokens=args.max_new,
                                temperature=0.0, top_k=0, top_p=1.0, greedy=True)
            ntok += len(ids)
            print(f"В: {q}\nО: {tok.decode(ids).strip()[:160]}")
        dt = max(1e-6, time.time() - t0)
        print(f"-- greedy: {ntok / dt:.1f} ток/с ({ntok} токенов за {dt:.1f} c)")


if __name__ == "__main__":
    main()
