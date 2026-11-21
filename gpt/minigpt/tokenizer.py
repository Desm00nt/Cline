"""Байт-уровневый BPE-токенайзер без внешних зависимостей.

Байт-уровневый — значит базовый словарь это 256 байт, поэтому кириллица, эмодзи
и опечатки кодируются всегда, без OOV. Обучение идёт по частотам пар с
инкрементальным индексом «пара -> слова», поэтому тренировка 4-8k мерджей
занимает секунды, а не минуты.

Формат файла (JSON):
    {"version": 1, "special": [...], "merges": [[id_a, id_b], ...]}
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set, Tuple

# Специальные токены занимают id 0..7, дальше идут 256 байт, потом мерджи.
SPECIAL_TOKENS = [
    "<|pad|>",        # 0
    "<|bos|>",        # 1
    "<|eos|>",        # 2  — конец документа
    "<|user|>",       # 3
    "<|assistant|>",  # 4
    "<|system|>",     # 5
    "<|end|>",        # 6  — конец реплики
    "<|mask|>",       # 7
]
N_SPECIAL = len(SPECIAL_TOKENS)
N_BYTES = 256
BASE_OFFSET = N_SPECIAL            # id байта b = BASE_OFFSET + b
MERGE_OFFSET = N_SPECIAL + N_BYTES  # id i-го мерджа = MERGE_OFFSET + i

PAD_ID, BOS_ID, EOS_ID, USER_ID, ASSISTANT_ID, SYSTEM_ID, END_ID, MASK_ID = range(N_SPECIAL)

# Пред-токенизация: сокращения / слова / числа / пунктуация / пробелы.
# [^\W\d_] в Python означает «буква любой письменности», т.е. кириллица включена.
_PAT = re.compile(
    r"'[a-zA-Zа-яА-ЯёЁ]{1,3}"
    r"|[^\W\d_]+\d*"
    r"|\d{1,3}(?:[ .]\d{3})*"
    r"|[^\s\w]+"
    r"|\s+",
    re.UNICODE,
)


def pre_tokenize(text: str) -> List[str]:
    return _PAT.findall(text)


class BPETokenizer:
    def __init__(self, merges: Sequence[Tuple[int, int]]):
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.ranks: Dict[Tuple[int, int], int] = {m: i for i, m in enumerate(self.merges)}
        self.id2bytes: List[bytes] = [bytes([b]) for b in range(N_BYTES)]
        for a, b in self.merges:
            self.id2bytes.append(self.id2bytes[a - BASE_OFFSET] + self.id2bytes[b - BASE_OFFSET])
        self.special_ids = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self._special_pat = re.compile(
            "(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")
        self._encode_cache: Dict[str, List[int]] = {}   # слово -> id, не сохраняется

    # ---------------- обучение ----------------
    @classmethod
    def train(cls, texts: Iterable[str], num_merges: int, verbose: bool = True,
              progress_every: int = 256) -> "BPETokenizer":
        # частоты уникальных «слов» (чанков пред-токенизации).
        # ВАЖНО: работаем в том же пространстве id, что и encode(),
        # иначе мерджи никогда не совпадут при кодировании.
        words: Dict[Tuple[int, ...], int] = Counter()
        for text in texts:
            for chunk in pre_tokenize(text):
                key = tuple(BASE_OFFSET + b for b in chunk.encode("utf-8"))
                words[key] += 1
        if not words:
            raise ValueError("пустой корпус — не на чем учить токенайзер")

        def pairs_of(w: Sequence[int]) -> Counter:
            c: Counter = Counter()
            for i in range(len(w) - 1):
                c[(w[i], w[i + 1])] += 1
            return c

        word_pairs: Dict[Tuple[int, ...], Counter] = {w: pairs_of(w) for w in words}
        pair_counts: Counter = Counter()
        pair_words: Dict[Tuple[int, int], Set[Tuple[int, ...]]] = {}
        for w, pc in word_pairs.items():
            for p, n in pc.items():
                pair_counts[p] += n
                pair_words.setdefault(p, set()).add(w)

        heap: List[Tuple[int, Tuple[int, int]]] = [(-n, p) for p, n in pair_counts.items()]
        heapq.heapify(heap)

        merges: List[Tuple[int, int]] = []
        while len(merges) < num_merges:
            pair = None
            while heap:  # ленивое удаление устаревших записей
                neg, cand = heapq.heappop(heap)
                if pair_counts.get(cand, 0) == -neg and -neg > 0:
                    pair = cand
                    break
            if pair is None:
                break
            a, b = pair
            new_id = MERGE_OFFSET + len(merges)
            merges.append((a, b))

            for w in list(pair_words.get(pair, ())):
                cnt = words[w]
                for p, n in word_pairs[w].items():
                    pair_counts[p] -= n
                    pair_words[p].discard(w)
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                del words[w]
                del word_pairs[w]
                # склеиваем все вхождения пары
                out: List[int] = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                nw = tuple(out)
                if nw in words:  # крайне редкое совпадение — объединяем счётчики
                    cnt += words[nw]
                    for p, n in word_pairs[nw].items():
                        pair_counts[p] -= n
                        pair_words[p].discard(nw)
                words[nw] = cnt
                pc = pairs_of(nw)
                word_pairs[nw] = pc
                for p, n in pc.items():
                    pair_counts[p] += n
                    pair_words.setdefault(p, set()).add(nw)
                    heapq.heappush(heap, (-pair_counts[p], p))

            if verbose and len(merges) % progress_every == 0:
                print(f"  мерджей: {len(merges)}/{num_merges}", flush=True)

        if verbose:
            print(f"  готово: словарь {cls(merges).vocab_size} токенов", flush=True)
        return cls(merges)

    # ---------------- кодирование ----------------
    @property
    def vocab_size(self) -> int:
        return MERGE_OFFSET + len(self.merges)

    def _encode_chunk(self, chunk: str) -> List[int]:
        # слова в тексте повторяются, поэтому кэш даёт кратный выигрыш на корпусе
        hit = self._encode_cache.get(chunk)
        if hit is not None:
            return list(hit)
        ids = self._encode_chunk_uncached(chunk)
        self._encode_cache[chunk] = ids
        return list(ids)

    def _encode_chunk_uncached(self, chunk: str) -> List[int]:
        ids = [BASE_OFFSET + b for b in chunk.encode("utf-8")]
        while len(ids) >= 2:
            best_i, best_rank = -1, len(self.merges)
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and r < best_rank:
                    best_i, best_rank = i, r
            if best_i < 0:
                break
            ids[best_i:best_i + 2] = [MERGE_OFFSET + best_rank]
        return ids

    def encode(self, text: str, add_specials: bool = False) -> List[int]:
        """Текст -> id. Специальные токены в тексте распознаются и не кодируются BPE.

        Важно: кодировать надо ПО ЧАНКАМ пред-токенизации, ровно как при обучении.
        Если скормить весь текст одним куском, мерджи смогут склеиваться через
        границы слов, а сложность станет квадратичной (на 30 КБ текста это
        минуты вместо миллисекунд).
        """
        out: List[int] = []
        if add_specials:
            out.append(BOS_ID)
        for part in self._special_pat.split(text):
            if not part:
                continue
            if part in self.special_ids:
                out.append(self.special_ids[part])
                continue
            for chunk in pre_tokenize(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def encode_documents(self, docs: Iterable[str], add_bos: bool = False) -> List[int]:
        """Документы склеиваются через <|eos|> — стандартная упаковка корпуса."""
        out: List[int] = []
        for d in docs:
            if add_bos:
                out.append(BOS_ID)
            out.extend(self.encode(d))
            out.append(EOS_ID)
        return out

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        buf = bytearray()
        last = BASE_OFFSET + len(self.id2bytes)
        for i in ids:
            if i < N_SPECIAL:
                if not skip_specials:
                    buf.extend(SPECIAL_TOKENS[i].encode("utf-8"))
            elif BASE_OFFSET <= i < last:
                buf.extend(self.id2bytes[i - BASE_OFFSET])
            # всё остальное (пад, мусор за пределами словаря) молча пропускаем
        return buf.decode("utf-8", errors="replace")

    # ---------------- сохранение ----------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "special": SPECIAL_TOKENS,
                       "merges": [list(m) for m in self.merges]}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("special") != SPECIAL_TOKENS:
            raise ValueError("несовместимый набор специальных токенов")
        return cls([(int(a), int(b)) for a, b in data["merges"]])
