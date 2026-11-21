"""Данные: упаковка корпуса в бинарный mmap-датасет и SFT-разметка диалогов."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import (ASSISTANT_ID, END_ID, EOS_ID, PAD_ID, SYSTEM_ID, USER_ID,
                        BPETokenizer)


def token_dtype(vocab_size: int):
    return np.uint16 if vocab_size < 65536 else np.uint32


def write_bin(ids: Sequence[int], path: str, vocab_size: int) -> None:
    dt = token_dtype(vocab_size)
    hi = max(ids, default=0)
    if hi >= vocab_size:
        raise ValueError(f"токен {hi} вне словаря (vocab_size={vocab_size})")
    if hi >= np.iinfo(dt).max:
        raise ValueError(f"id не помещаются в {dt.__name__}")
    np.asarray(ids, dtype=dt).tofile(path)


class PackedDataset(Dataset):
    """Блоки длины block_size поверх плоского массива токенов.

    stride < block_size даёт перекрывающиеся окна: на маленьком корпусе это
    заметно повышает эффективность использования данных (каждый токен
    participates в нескольких примерах).
    """

    def __init__(self, path: str, block_size: int, stride: Optional[int] = None):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        n = len(self.data)
        self.block_size = block_size
        if n < block_size + 1:
            raise ValueError(
                f"в {path} всего {n} токенов, нужно минимум {block_size + 1}. "
                f"Уменьшите --block-size или добавьте данных.")
        self.stride = stride or block_size
        self.n_windows = (n - 1 - block_size) // self.stride + 1

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = i * self.stride
        chunk = np.asarray(self.data[start:start + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


# ---------------------------------------------------------------------------
# Диалоги для SFT
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """Инструкционные данные в формате jsonl:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", ...}]}
    Обучающий сигнал считается только на ответах ассистента.
    """

    def __init__(self, path: str, tok: BPETokenizer, block_size: int, system: str = ""):
        self.block_size = block_size
        self.examples: List[Tuple[List[int], List[int]]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = list(obj["messages"])
                if system and not any(m["role"] == "system" for m in msgs):
                    msgs = [{"role": "system", "content": system}] + msgs
                ids, assistant = self._render(msgs, tok)
                if len(ids) < 3:
                    continue
                self.examples.append((ids, assistant))
        if not self.examples:
            raise ValueError(f"в {path} нет ни одного примера")

    @staticmethod
    def _render(msgs: List[Dict[str, str]], tok: BPETokenizer) -> Tuple[List[int], List[int]]:
        ids: List[int] = [EOS_ID]
        mask: List[int] = [0]
        for m in msgs:
            role = m["role"]
            header = {"system": SYSTEM_ID, "user": USER_ID, "assistant": ASSISTANT_ID}[role]
            body = tok.encode(m["content"])
            ids += [header] + body + [END_ID]
            learn = 1 if role == "assistant" else 0
            mask += [0] + [learn] * len(body) + [learn]
        return ids, mask

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ids, mask = self.examples[i]
        ids = ids[: self.block_size + 1]
        mask = mask[: self.block_size + 1]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        m = torch.tensor(mask[1:], dtype=torch.float32)
        return {"x": x, "y": y, "mask": m}


def sft_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    n = max(b["x"].shape[0] for b in batch)
    X = torch.full((len(batch), n), PAD_ID, dtype=torch.long)
    Y = torch.full((len(batch), n), -100, dtype=torch.long)
    M = torch.zeros((len(batch), n), dtype=torch.float32)
    for i, b in enumerate(batch):
        L = b["x"].shape[0]
        X[i, :L] = b["x"]
        Y[i, :L] = b["y"]
        M[i, :L] = b["mask"]
    return {"x": X, "y": Y, "mask": M}
