"""Разделяет рукописные диалоговые пары на обучающую и отложенную части.

Нужно, чтобы проверить главное: модель учится отвечать или зубрит примеры.
На 82 парах зубрёжка была доказана (train loss падал, валидационный рос).
Чтобы замер имел смысл, валидационные вопросы не должны встречаться
в обучающей части.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True,
                    help="файлы chat_*.jsonl с рукописными парами")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    pairs = []
    for path in args.src:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                pairs.append(json.loads(line))

    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_frac))
    val, train = pairs[:n_val], pairs[n_val:]

    # валидационный вопрос не должен встречаться в обучающей части
    train_q = {p["messages"][0]["content"] for p in train}
    val = [p for p in val if p["messages"][0]["content"] not in train_q]

    for path, chunk in ((args.train, train), (args.val, val)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in chunk),
            encoding="utf-8",
        )
        print(f"{path}: {len(chunk)}")

    overlap = train_q & {p["messages"][0]["content"] for p in val}
    print(f"пересечение вопросов: {len(overlap)}")


if __name__ == "__main__":
    main()
