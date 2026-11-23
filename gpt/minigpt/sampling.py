"""Генерация: семплинг с KV-кэшем, top-k/top-p, штрафом за повторы."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .model import GPT, KVCache
from .tokenizer import EOS_ID, END_ID


def make_cache(model: GPT, device, dtype=torch.float32) -> KVCache:
    return KVCache(model.cfg.cache_slots(), model.cfg.block_size, model.cfg.n_kv_head,
                   model.cfg.head_dim(), device=device, dtype=dtype)


def sample_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                  top_p: float = 1.0) -> torch.Tensor:
    """logits: (V,) или (B,V) -> сэмпл id."""
    logits = logits.float()
    if temperature <= 0:                    # жадный режим
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        thr = logits.topk(k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < thr, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative - probs > top_p        # ядро ядра не трогаем
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(
    model: GPT,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    stop_ids: Iterable[int] = (EOS_ID, END_ID),
    cache: Optional[KVCache] = None,
    on_token: Optional[Callable[[int], None]] = None,
    greedy: bool = False,
) -> List[int]:
    """Промпт -> список сгенерированных токенов (без промпта)."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    stop = set(stop_ids)
    out: List[int] = []
    ids_list = list(prompt_ids)
    if ids_list and isinstance(ids_list[0], (list, tuple)):
        raise TypeError("prompt_ids должен быть плоским списком id, а не [[...]]")

    if cache is None:
        cache = make_cache(model, device, dtype=next(model.parameters()).dtype)
    room = cache.max_len - cache.length - 1
    ids = list(prompt_ids)
    if len(ids) > room:                      # промпт не влезает -> берём хвост
        ids = ids[-room:]
    if not ids:
        return out

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(idx, cache=cache)
    next_logits = logits[0, -1]
    generated = list(ids)

    for _ in range(max_new_tokens):
        lg = next_logits
        if repetition_penalty != 1.0 and generated:
            seen = torch.tensor(sorted(set(generated)), device=lg.device)
            scores = lg[seen]
            lg = lg.clone()
            lg[seen] = torch.where(scores > 0, scores / repetition_penalty,
                                   scores * repetition_penalty)
        if greedy:
            nxt = int(lg.argmax())
        else:
            nxt = int(sample_logits(lg, temperature, top_k, top_p))
        out.append(nxt)
        if on_token is not None:
            on_token(nxt)
        if nxt in stop:
            break
        if cache.length >= cache.max_len:
            break
        step = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits, _ = model(step, cache=cache)
        next_logits = logits[0, -1]
        generated.append(nxt)

    if was_training:
        model.train()
    return out


@torch.no_grad()
def chat_once(model: GPT, history: Sequence[dict], max_new_tokens: int = 256,
              temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9,
              repetition_penalty: float = 1.05, greedy: bool = False,
              on_token=None, cache: Optional[KVCache] = None) -> List[int]:
    """Один ход диалога по истории [{"role": ..., "ids": [...]}]."""
    from .tokenizer import ASSISTANT_ID, SYSTEM_ID, USER_ID
    headers = {"system": SYSTEM_ID, "user": USER_ID, "assistant": ASSISTANT_ID}
    ids: List[int] = [EOS_ID]
    for m in history:
        header = headers.get(m.get("role", "user"), USER_ID)
        ids += [header] + list(m["ids"]) + [END_ID]
    ids += [ASSISTANT_ID]
    return generate(model, ids, max_new_tokens=max_new_tokens, temperature=temperature,
                    top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                    stop_ids=(END_ID, EOS_ID), greedy=greedy, on_token=on_token,
                    cache=cache)
