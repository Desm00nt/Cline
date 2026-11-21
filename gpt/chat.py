#!/usr/bin/env python3
"""Терминальный чат с обученной моделью (KV-кэш, стриминг).

    python chat.py --ckpt out/sft/model_sft_ema.pt

Команды внутри чата: /reset — очистить историю, /info — про модель, /quit — выход.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from minigpt.model import load_any
from minigpt.sampling import chat_once
from minigpt.tokenizer import BPETokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default="data/prepared/tokenizer.json")
    p.add_argument("--system", default="")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--kb", default="", help="файл или папка с текстами: включать RAG")
    p.add_argument("--kb-k", type=int, default=3, help="сколько кусков подставлять")
    p.add_argument("--kb-budget", type=int, default=1024, help="максимум токенов контекста")
    p.add_argument("--kb-cache", default="", help="кэш индекса (.json)")
    p.add_argument("--no-kb", action="store_true", help="не подставлять найденное в этом запуске")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    device = pick_device(args.device)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if not os.path.exists(args.tokenizer):
        raise SystemExit(f"нет токенайзера {args.tokenizer}")
    tok = BPETokenizer.load(args.tokenizer)
    model, payload = load_any(args.ckpt, device)
    amp = torch.bfloat16 if device == "cuda" else torch.float32

    n_params = model.count_params()
    print(f"модель: {args.ckpt}")
    print(f"параметров: {n_params/1e6:.2f}M | контекст: {model.cfg.block_size} | "
          f"словарь: {model.cfg.vocab_size} | устройство: {device}")
    if args.system:
        print(f"системный промпт: {args.system}")

    index = None
    if args.kb:
        from minigpt.rag import build_index
        print(f"строю индекс по {args.kb} ...")
        index = build_index(args.kb, tok, cache=args.kb_cache or None)
    print("команды: /reset /info /kb /quit")

    dialog: list = []          # только реплики пользователя и ассистента
    kb_ids: list = []          # найденный контекст на текущий ход
    kb_hits: list = []

    while True:
        try:
            user = input("\nВы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "exit", "quit"):
            break
        if user == "/reset":
            dialog = []
            print("[история очищена]")
            continue
        if user == "/kb":
            if not index:
                print("[RAG выключен: нужен --kb]")
            elif kb_hits:
                for i, h in enumerate(kb_hits, 1):
                    print(f"  [{i}] score={h.score:.3f} doc={h.doc} :: "
                          f"{h.text[:90].replace(chr(10), ' ')}...")
            else:
                print("[контекст не подставлялся]")
            continue
        if user == "/info":
            print(f"реплик: {len(dialog)} | токенов: {sum(len(m['ids']) for m in dialog)}"
                  + (f" | RAG-контекст: {len(kb_ids)} токенов" if kb_ids else ""))
            continue

        u_ids = tok.encode(user)
        dialog.append({"role": "user", "ids": u_ids})

        # RAG: ищем по запросу пользователя и кладём найденное системным сообщением
        kb_ids, kb_hits = ([], [])
        if index is not None and not args.no_kb:
            kb_ids, kb_hits = index.context_ids(user, k=args.kb_k, budget=args.kb_budget)

        # держим всё внутри контекста: режем самые старые реплики
        budget = int(model.cfg.block_size * 0.8)

        def assemble() -> list:
            out = []
            if args.system:
                out.append({"role": "system", "ids": tok.encode(args.system)})
            if kb_ids:
                out.append({"role": "system", "ids": kb_ids})
            out.extend(dialog)
            return out

        messages = assemble()
        while len(dialog) > 1 and sum(len(m["ids"]) for m in messages) > budget:
            del dialog[0]
            messages = assemble()

        sys.stdout.write("Модель> ")
        sys.stdout.flush()
        state = {"printed": "", "ids": []}
        t0 = time.time()

        def on_token(tid: int) -> None:
            """Печатаем только дельту: декодируем накопленный список заново,
            чтобы не резать многосимвольные UTF-8 последовательности."""
            state["ids"].append(tid)
            if args.no_stream:
                return
            text = tok.decode(state["ids"])
            if len(text) > len(state["printed"]):
                sys.stdout.write(text[len(state["printed"]):])
                sys.stdout.flush()
            state["printed"] = text

        with torch.autocast(device_type=device.split(":")[0], dtype=amp,
                            enabled=(amp != torch.float32)):
            ans_ids = chat_once(model, messages, max_new_tokens=args.max_new,
                                temperature=args.temperature, top_k=args.top_k,
                                top_p=args.top_p,
                                repetition_penalty=args.repetition_penalty,
                                greedy=args.greedy, on_token=on_token)
        if args.no_stream:
            print(tok.decode(ans_ids), end="")
        print()
        dt = max(1e-6, time.time() - t0)
        print(f"  [{len(ans_ids)} токенов за {dt:.1f} c = {len(ans_ids)/dt:.1f} ток/с]",
              file=sys.stderr)
        dialog.append({"role": "assistant", "ids": ans_ids})


if __name__ == "__main__":
    main()
