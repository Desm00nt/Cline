#!/usr/bin/env python3
"""Готовит данные: учит BPE-токенайзер и пакует корпус в train.bin/val.bin.

    python scripts/prepare_data.py --input data/raw --out data/prepared --num-merges 4000

--input может быть файлом или папкой (.txt/.md, рекурсивно). Документы
разделяются пустой строкой или границей файла.

Кодирование потоковое: токены документа сразу пишутся в бинарный файл, а не
копятся в списке Python-интов. На корпусе в 200M токенов список дал бы ~6 ГБ
и закончился бы OOM (exit 137) на машине с 4 ГБ RAM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from array import array
from typing import Iterable, Iterator, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from minigpt.data import token_dtype, write_bin  # noqa: F401  (write_bin — для тестов)
from minigpt.tokenizer import BPETokenizer

TEXT_EXT = (".txt", ".md", ".jsonl")
EOS_ID = 2
COPY_CHUNK = 1 << 22  # 4 МБ на шаг копирования
READ_CHUNK = 1 << 20  # 1M символов на шаг чтения corpus.txt


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
    p.add_argument("--tokenizer", default="",
                   help="взять готовый tokenizer.json вместо обучения. Обязательно "
                        "при подготовке данных под существующий чекпойнт: новый "
                        "словарь не совпадёт с выученными эмбеддингами")
    p.add_argument("--seed", type=int, default=1337, help="зерно разреза train/val")
    p.add_argument("--exclude", default="/docs/",
                   help="подстрока пути, которую пропускаем. По умолчанию /docs/: "
                        "fetch_corpus.py пишет и corpus.txt, и docs/ с теми же "
                        "текстами, иначе корпус закодировался бы дважды")
    return p.parse_args()


def list_files(path: str, exclude: str = "") -> List[Tuple[str, int]]:
    """Список текстовых файлов с размерами (размер нужен для выборки под токенайзер)."""
    files: List[Tuple[str, int]] = []
    if os.path.isdir(path):
        for root, _, names in os.walk(path):
            for n in sorted(names):
                full = os.path.join(root, n)
                if not n.endswith(TEXT_EXT) or n.endswith(".bin"):
                    continue
                if exclude and exclude in full.replace(os.sep, "/"):
                    continue
                files.append((full, os.path.getsize(full)))
    elif os.path.isfile(path):
        files.append((path, os.path.getsize(path)))
    else:
        raise SystemExit(f"не найден {path}")
    if not files:
        raise SystemExit(f"в {path} нет текстовых файлов")
    return sorted(files)


def iter_docs(path: str, doc_sep: str) -> Iterator[str]:
    """Документы одного файла, без загрузки всего файла в память.

    corpus.txt у novaya_gazeta — 590M символов: fh.read() плюс split() дают
    ~2.4 ГБ и падают в MemoryError на машине с 4 ГБ RAM. Читаем кусками, хвост
    переносим в следующий (разделитель может попасть на границу куска).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        buf = ""
        while True:
            chunk = fh.read(READ_CHUNK)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(doc_sep)
            buf = parts.pop()          # последний кусок может быть неполным
            for part in parts:
                part = part.strip()
                if part:
                    yield part
    part = buf.strip()
    if part:
        yield part


def collect_docs(path: str, doc_sep: str, exclude: str = "") -> List[str]:
    """Все документы сразу — только для небольших корпусов и тестов."""
    docs: List[str] = []
    for f, _ in list_files(path, exclude):
        docs.extend(iter_docs(f, doc_sep))
    if not docs:
        raise SystemExit("не удалось найти ни одного документа")
    return docs


def sample_for_tokenizer(files: List[Tuple[str, int]], doc_sep: str,
                         max_chars: int, seed: int = 0) -> Tuple[List[str], int]:
    """Случайные документы суммарно на max_chars символов.

    Берём файлы наугад, а не первые подряд: иначе словарь выучился бы на одном
    жанре (при сортировке по имени новостной репозиторий идёт раньше романов).
    """
    order = np.random.RandomState(seed).permutation(len(files))
    sample: List[str] = []
    taken = 0
    for i in order:
        f, size = files[int(i)]
        for doc in iter_docs(f, doc_sep):
            sample.append(doc)
            taken += len(doc)
            if taken >= max_chars:
                return sample, taken
        if size > 0 and taken >= max_chars:
            break
    return sample, taken


def encode_stream(files: Iterable[str], doc_sep: str, tok: BPETokenizer,
                  train_path: str, val_path: str, dtype,
                  val_frac: float = 0.05, seed: int = 1337) -> Tuple[int, int, int]:
    """Кодирует и сразу режет: каждый документ целиком уходит в train или в val.

    Разрез именно по документу и именно случайный. Прежний вариант резал хвост
    потока, а файлы обходятся по алфавиту, поэтому в val попадал последний
    источник целиком: так валидация состояла только из романов при train на 93%
    из новостей и измеряла смену домена, а не качество модели (ppl 21.2 на
    романах против 12.5 на новостях у одного и того же чекпойнта).

    Возвращает (токенов в train, токенов в val, документов).
    """
    rng = np.random.default_rng(seed)
    n_train = n_val = docs = 0
    with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        buf_tr, buf_va = bytearray(), bytearray()
        for f in files:
            for doc in iter_docs(f, doc_sep):
                ids = tok.encode(doc)
                if not ids:
                    continue
                arr = np.asarray(ids + [EOS_ID], dtype=dtype)
                if val_frac > 0 and rng.random() < val_frac:
                    buf_va += arr.tobytes()
                    n_val += arr.size
                else:
                    buf_tr += arr.tobytes()
                    n_train += arr.size
                docs += 1
            # сбрасываем раз в файл: иначе на корпус в гигабайт буфер не влезет
            ftr.write(buf_tr); fva.write(buf_va)
            buf_tr.clear(); buf_va.clear()
    if n_train == 0:
        raise SystemExit("не удалось закодировать ни одного документа")
    return n_train, n_val, docs


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = list_files(args.input, args.exclude)
    total_bytes = sum(s for _, s in files)
    print(f"файлов: {len(files)} ({total_bytes/1e6:.1f} МБ), пропускаем {args.exclude!r}")

    # 1) токенайзер — переиспользуем готовый либо учим на случайной выборке
    if args.tokenizer:
        tok = BPETokenizer.load(args.tokenizer)
        print(f"взят готовый токенайзер {args.tokenizer}: словарь {tok.vocab_size}")
    else:
        sample, taken = sample_for_tokenizer(files, args.doc_sep, args.max_train_chars)
        t0 = time.time()
        print(f"учим токенайзер на {taken:,} символах ({len(sample)} документов)...")
        tok = BPETokenizer.train(sample, args.num_merges)
        print(f"  время: {time.time()-t0:.1f} c, словарь: {tok.vocab_size}")
    tok.save(os.path.join(args.out, "tokenizer.json"))

    # 2) кодируем потоково, документ целиком уходит в train или в val
    dtype = token_dtype(tok.vocab_size)
    print("кодируем корпус...")
    t0 = time.time()
    n_train, n_val, docs = encode_stream(
        (f for f, _ in files), args.doc_sep, tok,
        os.path.join(args.out, "train.bin"), os.path.join(args.out, "val.bin"),
        dtype, args.val_frac, args.seed)
    dt = time.time() - t0
    total = n_train + n_val
    print(f"  токенов: {total:,} за {dt:.1f} c ({total/max(dt,1e-9):,.0f} ток/с)")

    meta = {
        "vocab_size": tok.vocab_size,
        # не args.num_merges: при --tokenizer словарь взят готовый, и аргумент
        # к нему отношения не имеет
        "num_merges": len(getattr(tok, "merges", [])) or args.num_merges,
        "train_tokens": n_train, "val_tokens": n_val,
        "docs": docs, "val_frac": args.val_frac, "seed": args.seed,
        # именно байт на токен: размер файла в UTF-8, а не число символов
        "bytes_per_token": round(total_bytes / max(1, total), 3),
        "dtype": np.dtype(dtype).name,
        "encode_seconds": round(dt, 1),
        "tokens_per_second": round(total / max(dt, 1e-9)),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"готово -> {args.out}")


if __name__ == "__main__":
    main()
