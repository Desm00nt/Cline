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
                  tmp_path: str, dtype) -> Tuple[int, array]:
    """Пишет токены в tmp_path, возвращает (всего токенов, границы документов)."""
    ends = array("q")
    total = 0
    with open(tmp_path, "wb") as out:
        for f in files:
            for doc in iter_docs(f, doc_sep):
                ids = tok.encode(doc)
                if not ids:
                    continue
                arr = np.asarray(ids + [EOS_ID], dtype=dtype)
                out.write(arr.tobytes())
                total += arr.size
                ends.append(total)
    if total == 0:
        raise SystemExit("не удалось закодировать ни одного документа")
    return total, ends


def copy_range(src: str, dst: str, start: int, stop: int, itemsize: int) -> int:
    """Копирует токены [start, stop) кусками, без чтения всего файла в память."""
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fi.seek(start * itemsize)
        left = (stop - start) * itemsize
        while left > 0:
            buf = fi.read(min(COPY_CHUNK, left))
            if not buf:
                break
            fo.write(buf)
            left -= len(buf)
    return stop - start


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = list_files(args.input, args.exclude)
    total_bytes = sum(s for _, s in files)
    print(f"файлов: {len(files)} ({total_bytes/1e6:.1f} МБ), пропускаем {args.exclude!r}")

    # 1) токенайзер — на случайной выборке, чтобы не ждать на больших корпусах
    sample, taken = sample_for_tokenizer(files, args.doc_sep, args.max_train_chars)
    t0 = time.time()
    print(f"учим токенайзер на {taken:,} символах ({len(sample)} документов)...")
    tok = BPETokenizer.train(sample, args.num_merges)
    print(f"  время: {time.time()-t0:.1f} c, словарь: {tok.vocab_size}")
    tok.save(os.path.join(args.out, "tokenizer.json"))

    # 2) кодируем потоково во временный файл
    dtype = token_dtype(tok.vocab_size)
    tmp_path = os.path.join(args.out, "all_tokens.tmp.bin")
    print("кодируем корпус...")
    t0 = time.time()
    total, ends = encode_stream((f for f, _ in files), args.doc_sep, tok, tmp_path, dtype)
    dt = time.time() - t0
    print(f"  токенов: {total:,} за {dt:.1f} c ({total/max(dt,1e-9):,.0f} ток/с)")

    # 3) режем по границе документа, ближайшей к (1 - val_frac)
    itemsize = np.dtype(dtype).itemsize
    if args.val_frac > 0:
        ends_np = np.frombuffer(ends.tobytes(), dtype=np.int64)
        idx = int(np.searchsorted(ends_np, int(total * (1 - args.val_frac))))
        cut = int(ends[idx]) if idx < len(ends) else total
    else:
        cut = total
    n_train = copy_range(tmp_path, os.path.join(args.out, "train.bin"), 0, cut, itemsize)
    n_val = 0
    if cut < total:
        n_val = copy_range(tmp_path, os.path.join(args.out, "val.bin"), cut, total, itemsize)
    os.remove(tmp_path)

    meta = {
        "vocab_size": tok.vocab_size, "num_merges": args.num_merges,
        "train_tokens": n_train, "val_tokens": n_val,
        "docs": len(ends),
        "chars_per_token": (total_bytes / max(1, total)),
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
