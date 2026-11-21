"""RAG: поиск по корпусу и подстановка найденного в контекст.

Зачем это нужно маленькой модели. Трансформер не «учится» на лету: вне
обучения его веса не меняются, поэтому новые факты он взять неоткуда. Единственное
место, куда можно положить свежую информацию во время разговора, — это контекст.
RAG делает именно это: ищет в базе релевантные куски и подставляет их в промпт,
то есть модель «изучает» документ прямо сейчас, не запоминая его в весах.

Разделение ответственности получается такое:
    веса модели  — язык, стиль, умение рассуждать (что выучено на pretrain/SFT)
    контекст     — факты и документы (RAG)

Поиск — TF-IDF с косинусной близостью по инвертированному индексу. Никаких
зависимостей, для корпуса в десятки тысяч чанков работает за миллисекунды.
Для миллионов чанков имеет смысл заменить на FAISS/HNSW: интерфейс `search`
при этом не меняется.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .tokenizer import BPETokenizer, EOS_ID


_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def terms(text: str) -> Counter:
    """Признаки для поиска: слова + символьные триграммы.

    BPE-токены для поиска не годятся: одно и то же слово в запросе и в документе
    режется по-разному («борщ» -> бор+щ+щ, «борща» -> бо+рщ+а), и релевантный
    документ не находится. Слова дают точные совпадения, а триграммы ловят
    morphology и опечатки («варить»/«Варить»/«сварить»).
    """
    out: Counter = Counter()
    for w in _WORD.findall(text.lower()):
        out[w] += 1
        if len(w) >= 3:
            for i in range(len(w) - 2):
                out["#" + w[i:i + 3]] += 1
    return out


@dataclass
class Hit:
    score: float
    text: str
    ids: List[int]
    doc: str = ""


class TFIDFIndex:
    def __init__(self, tok: BPETokenizer, chunk_tokens: int = 256, overlap: int = 48):
        self.tok = tok
        self.chunk_tokens = chunk_tokens
        self.overlap = overlap
        self.chunks: List[List[int]] = []
        self.docs: List[str] = []
        self.tf: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.norm: List[float] = []
        self.postings: Dict[str, List[Tuple[int, float]]] = {}

    # ---------------- наполнение ----------------
    def add_text(self, text: str, doc: str = "") -> int:
        """Режет текст на чанки с перекрытием. Возвращает число добавленных чанков."""
        ids = self.tok.encode(text)
        if not ids:
            return 0
        step = max(1, self.chunk_tokens - self.overlap)
        added = 0
        for start in range(0, len(ids), step):
            piece = ids[start:start + self.chunk_tokens]
            if len(piece) < self.chunk_tokens // 2:
                break
            self._add_chunk(piece, doc)
            added += 1
            if start + self.chunk_tokens >= len(ids):
                break
        return added

    def _add_chunk(self, ids: List[int], doc: str) -> None:
        idx = len(self.chunks)
        self.chunks.append(ids)
        self.docs.append(doc)
        counts = terms(self.tok.decode(ids))
        self.tf.append(counts)
        for t in counts:
            self.df[t] = self.df.get(t, 0) + 1

    def build(self) -> None:
        """Считает IDF, нормы и строит инвертированный индекс."""
        n = max(1, len(self.chunks))
        self.idf = {t: math.log((n + 1) / (df + 1)) + 1.0 for t, df in self.df.items()}
        self.postings = {t: [] for t in self.idf}
        self.norm = []
        for i, counts in enumerate(self.tf):
            s = 0.0
            for t, c in counts.items():
                w = (1 + math.log(c)) * self.idf.get(t, 0.0)
                self.postings[t].append((i, w))
                s += w * w
            self.norm.append(math.sqrt(s) or 1.0)

    # ---------------- поиск ----------------
    def search(self, query: str, k: int = 4) -> List[Hit]:
        qc = terms(query)
        if not qc or not self.postings:
            return []
        # Отбрасываем признаки, которые есть почти везде (триграммы вроде «#ого»):
        # они топят запрос в шумовых совпадениях. Если после фильтра ничего не
        # осталось — возвращаемся к полному набору.
        n = len(self.chunks)
        selective = {t: c for t, c in qc.items()
                     if t in self.idf and self.df.get(t, 0) <= max(1, int(n * 0.5))}
        if selective:
            qc = selective
        q_vec: Dict[str, float] = {}
        s = 0.0
        for t, c in qc.items():
            if t in self.idf:
                w = (1 + math.log(c)) * self.idf[t]
                q_vec[t] = w
                s += w * w
        if not q_vec:
            return []
        q_norm = math.sqrt(s)

        scores: Dict[int, float] = {}
        for t, w in q_vec.items():
            for idx, dw in self.postings.get(t, ()):
                scores[idx] = scores.get(idx, 0.0) + w * dw
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out = []
        for idx, raw in ranked:
            out.append(Hit(score=raw / (q_norm * self.norm[idx]),
                           text=self.tok.decode(self.chunks[idx]),
                           ids=self.chunks[idx], doc=self.docs[idx]))
        return out

    def context_ids(self, query: str, k: int = 3, budget: int = 1024) -> Tuple[List[int], List[Hit]]:
        """Собирает контекст в виде токенов: нумерованные куски, в рамках бюджета."""
        hits = self.search(query, k=k)
        ids: List[int] = []
        kept: List[Hit] = []
        for i, h in enumerate(hits, 1):
            header = self.tok.encode(f"[{i}] ")
            piece = header + h.ids + [EOS_ID]
            if len(ids) + len(piece) > budget:
                continue
            ids.extend(piece)
            kept.append(h)
        return ids, kept

    # ---------------- сохранение ----------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chunk_tokens": self.chunk_tokens, "overlap": self.overlap,
                       "docs": self.docs, "chunks": self.chunks}, f)

    @classmethod
    def load(cls, path: str, tok: BPETokenizer) -> "TFIDFIndex":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        idx = cls(tok, d["chunk_tokens"], d["overlap"])
        for chunk, doc in zip(d["chunks"], d["docs"]):
            idx._add_chunk([int(t) for t in chunk], doc)
        idx.build()
        return idx

    def __len__(self) -> int:
        return len(self.chunks)


def iter_texts(path: str, min_chars: int = 200, max_docs: int = 0) -> Iterable[Tuple[str, str]]:
    """(имя, текст) для файла или папки с .txt/.md."""
    files = []
    if os.path.isdir(path):
        for root, _, names in os.walk(path):
            files += [os.path.join(root, n) for n in sorted(names)
                      if n.endswith((".txt", ".md"))]
    else:
        files = [path]
    n = 0
    for f in sorted(files):
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for part in text.split("\n\n"):
            part = part.strip()
            if len(part) >= min_chars:
                n += 1
                yield os.path.basename(f), part
                if max_docs and n >= max_docs:
                    return


def build_index(kb_path: str, tok: BPETokenizer, cache: Optional[str] = None,
                chunk_tokens: int = 256, max_docs: int = 0) -> TFIDFIndex:
    if cache and os.path.exists(cache):
        return TFIDFIndex.load(cache, tok)
    idx = TFIDFIndex(tok, chunk_tokens=chunk_tokens)
    total = 0
    for name, text in iter_texts(kb_path, max_docs=max_docs):
        idx.add_text(text, doc=name)
        total += 1
        if total % 200 == 0:
            print(f"  проиндексировано документов: {total}, чанков: {len(idx)}", flush=True)
    idx.build()
    print(f"индекс: {len(idx)} чанков из {total} документов")
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        idx.save(cache)
    return idx
