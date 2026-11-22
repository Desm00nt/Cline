"""GPT-подобный декодер с нуля.

Что здесь есть и зачем (всё это критично именно на маленьких моделях):

* Pre-RMSNorm блоки — устойчивее LayerNorm на bf16 и дешевле.
* RoPE вместо таблицы позиций — длина контекста на инференсе не ограничена тренировкой.
* GQA (grouped-query attention) — меньше параметров в k/v и меньше KV-кэш.
* QK-Norm — нормализация q и k перед RoPE убирает взрыв attention-логитов.
* SwiGLU MLP — стабильно лучше GELU-MLP при том же числе параметров.
* Гибрид скользящего окна и глобального внимания: большинство слоёв смотрят
  только в последние `sliding_window` токенов, каждый `global_every`-й слой —
  глобальный. Это позволяет держать контекст 4k+ без квадратичной цены.
* Связанные веса (embedding == head) — экономит vocab*n_embd параметров,
  что при 16k словаре больше трети бюджета.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


# ---------------------------------------------------------------------------
# Базовые слои
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square нормализация. Считается в fp32 даже под autocast."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32-вход (инференс на CPU, либо нормализация вне autocast) — считаем
        # одним fused-опом: на декодировании каждый токен делает 33 таких
        # нормализации, и шесть отдельных aten-вызовов на каждую стоят дороже,
        # чем сама арифметика.
        if (x.dtype == torch.float32 and self.weight.dtype == torch.float32
                and not torch.jit.is_tracing()):
            # при экспорте в ONNX остаёмся на формуле ниже: aten::rms_norm
            # не поддерживается экспортером
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        # под autocast (bf16) вручную и строго в fp32: иначе на длинном
        # контексте точности не хватает
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dt)


def build_rope_cache(block_size: int, head_dim: int, base: float) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(block_size, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)                       # (block, hd/2)
    emb = torch.cat([freqs, freqs], dim=-1)                  # (block, hd)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, hd); cos/sin: (T, hd). Классический rotate_half вариант.

    Арифметика делается в fp32: на bf16 вращение быстро теряет точность.
    """
    dt = x.dtype
    # приведение к fp32 пропускаем, если оно и так fp32: .float()/.to() на
    # каждом токене — это лишние aten-вызовы, а их на декодировании сотни
    xf = x if dt == torch.float32 else x.float()
    x1 = xf[..., : xf.shape[-1] // 2]
    x2 = xf[..., xf.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    cos = cos if cos.dtype == torch.float32 else cos.float()
    sin = sin if sin.dtype == torch.float32 else sin.float()
    out = xf * cos[None, None] + rotated * sin[None, None]   # (1,1,T,hd)
    return out if dt == torch.float32 else out.to(dt)


_GQA_OK: Optional[bool] = None


def _sdpa_supports_gqa() -> bool:
    """Поддерживает ли установленный torch enable_gqa (появился в 2.5).

    Проверяем один раз реальным вызовом: на старых версиях и на части бэкендов
    параметр либо отсутствует, либо падает, и тогда остаёмся на
    repeat_interleave.
    """
    global _GQA_OK
    if _GQA_OK is None:
        try:
            q = torch.zeros(1, 4, 2, 8)
            k = torch.zeros(1, 2, 2, 8)
            F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
            _GQA_OK = True
        except Exception:
            _GQA_OK = False
    return _GQA_OK


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head
        self.hd = cfg.head_dim()
        self.n_rep = self.n_head // self.n_kv
        self.global_layer = cfg.is_global_layer(layer_idx)
        self.window = 0 if self.global_layer else cfg.sliding_window

        self.c_q = nn.Linear(cfg.n_embd, self.n_head * self.hd, bias=cfg.bias)
        self.c_k = nn.Linear(cfg.n_embd, self.n_kv * self.hd, bias=cfg.bias)
        self.c_v = nn.Linear(cfg.n_embd, self.n_kv * self.hd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.hd)
            self.k_norm = RMSNorm(self.hd)
        else:
            self.q_norm = self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional["KVCache"] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        # QK-Norm применяется ПО ГОЛОВАМ (последняя ось = head_dim),
        # поэтому сначала reshape, потом нормализация.
        q = self.c_q(x).view(B, T, self.n_head, self.hd)
        k = self.c_k(x).view(B, T, self.n_kv, self.hd)
        v = self.c_v(x).view(B, T, self.n_kv, self.hd)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # переводим в (B, H, T, hd) и вращаем q/k
        q = apply_rope(q.transpose(1, 2), cos, sin)
        k = apply_rope(k.transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)

        past = 0
        if cache is not None:
            assert B == 1, "KVCache рассчитан на batch_size=1 (генерация)"
            past = cache.length
            # кэш хранится как (max_len, n_kv, hd) — пишем чанк и берём весь хвост,
            # включая только что добавленные токены (cache.advance будет позже)
            cache.append(self.layer_idx, k.squeeze(0).transpose(0, 1), v.squeeze(0).transpose(0, 1))
            k = cache.get_k(self.layer_idx, past + T).unsqueeze(0).transpose(1, 2)
            v = cache.get_v(self.layer_idx, past + T).unsqueeze(0).transpose(1, 2)
            k = k.expand(B, self.n_kv, past + T, self.hd)
            v = v.expand(B, self.n_kv, past + T, self.hd)

        # GQA: вместо физического повторения k/v (repeat_interleave копирует
        # тензор на каждом токене) просим SDPA самому развернуть группы.
        drop_p = self.cfg.dropout if self.training else 0.0
        gqa = (self.n_rep > 1 and _sdpa_supports_gqa()
               and not torch.jit.is_tracing())   # экспортер ONNX не знает enable_gqa
        if self.n_rep > 1 and not gqa:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        if attn_mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                               dropout_p=drop_p, enable_gqa=gqa)
        elif cache is None:
            # весь фрагмент виден сразу: причинность даёт is_causal (быстрый путь)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=drop_p, enable_gqa=gqa)
        else:
            # шаг генерации: все ключи в кэше уже в прошлом, маска не нужна.
            # is_causal=True здесь использовать НЕЛЬЗЯ: при T_q=1 и T_kv>1
            # он оставляет видимым только самый первый ключ.
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p,
                                               enable_gqa=gqa)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        h = cfg.mlp_hidden()
        self.c_gate = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
        self.c_up = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
        self.c_down = nn.Linear(h, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_down(F.silu(self.c_gate(x)) * self.c_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg, layer_idx)
        self.ln_2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None, attn_mask=None):
        x = x + self.attn(self.ln_1(x), cos, sin, cache, attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class KVCache:
    """Статический KV-кэш: без аллокаций на каждом шаге генерации."""

    def __init__(self, n_layer: int, max_len: int, n_kv_head: int, head_dim: int,
                 device, dtype=torch.float32):
        self.max_len = max_len
        self.length = 0
        self.k: List[torch.Tensor] = [
            torch.empty(max_len, n_kv_head, head_dim, device=device, dtype=dtype)
            for _ in range(n_layer)
        ]
        self.v: List[torch.Tensor] = [
            torch.empty(max_len, n_kv_head, head_dim, device=device, dtype=dtype)
            for _ in range(n_layer)
        ]

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        t = k.shape[0]
        assert self.length + t <= self.max_len, "контекст переполнен"
        self.k[layer][self.length:self.length + t] = k.to(self.k[layer].dtype)
        self.v[layer][self.length:self.length + t] = v.to(self.v[layer].dtype)

    def get_k(self, layer: int, end: Optional[int] = None) -> torch.Tensor:
        return self.k[layer][: self.length if end is None else end]

    def get_v(self, layer: int, end: Optional[int] = None) -> torch.Tensor:
        return self.v[layer][: self.length if end is None else end]

    def advance(self, t: int) -> None:
        self.length += t


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.block_size, cfg.head_dim(), cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        if cfg.residual_scale_init:
            # уменьшенный init выходных проекций: остаточный поток не растёт с глубиной
            scale = 1.0 / math.sqrt(2 * cfg.n_layer)
            for block in self.blocks:
                nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=cfg.init_std * scale)
                nn.init.normal_(block.mlp.c_down.weight, mean=0.0, std=cfg.init_std * scale)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ---------------- прямой проход ----------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        loss_mask: Optional[torch.Tensor] = None,
        z_loss: Optional[float] = None,
    ):
        B, T = idx.shape
        past = cache.length if cache is not None else 0
        assert past + T <= self.cfg.block_size, (
            f"последовательность {past + T} длиннее block_size={self.cfg.block_size}")

        x = self.tok_emb(idx)
        cos = self.rope_cos[past:past + T].to(x.dtype)
        sin = self.rope_sin[past:past + T].to(x.dtype)

        # Маски строятся один раз на весь проход.
        # Без кэша глобальным слоям маска не нужна (is_causal), со кэшем нужна
        # всегда, кроме шага длиной 1 — там причинность выполняется тривиально.
        need_global_mask = cache is not None and T > 1
        global_mask = self._attention_mask(T, past, idx.device, None) if need_global_mask else None
        # Скользящее окно короче блока — единственная причина строить отдельную
        # маску. Если окно не короче блока (T <= sliding_window), ограничение
        # окна не отсекает ни одного токена, маска совпадает с причинной, и её
        # можно не строить: SDPA уходит на fused-путь is_causal вместо медленного
        # math-пути с материализацией attn_bias размером B x heads x T^2.
        # Это главный режим обучения (block_size <= sliding_window).
        window_bites = T > self.cfg.sliding_window if self.cfg.sliding_window > 0 else False
        if any(not b.attn.global_layer for b in self.blocks) and (
                cache is not None or window_bites):
            swa_mask = self._attention_mask(T, past, idx.device, self.cfg.sliding_window)
        else:
            swa_mask = None

        for block in self.blocks:
            mask = global_mask if block.attn.global_layer else swa_mask
            x = block(x, cos, sin, cache, mask)

        if cache is not None:
            cache.advance(T)

        x = self.ln_f(x)
        logits = self.head(x)
        if self.cfg.logit_softcapping > 0:
            cap = self.cfg.logit_softcapping
            logits = cap * torch.tanh(logits / cap)

        loss = None
        if targets is not None:
            loss = self.compute_loss(logits, targets, loss_mask, z_loss)
        return logits, loss

    def _attention_mask(self, T: int, past: int, device, window: Optional[int]) -> torch.Tensor:
        """Булева маска (1,1,T,past+T): True = токен можно видеть.

        window=None -> только причинность; иначе ещё и ограничение окна.
        """
        q_pos = torch.arange(past, past + T, device=device).unsqueeze(1)   # (T,1)
        k_pos = torch.arange(past + T, device=device).unsqueeze(0)        # (1,kv)
        visible = k_pos <= q_pos
        if window:
            visible = visible & (q_pos - k_pos < window)
        return visible.unsqueeze(0).unsqueeze(0)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        z_loss: Optional[float] = None,
    ) -> torch.Tensor:
        B, T, V = logits.shape
        # reshape, а не view: на вход могут прийти срезы (например, logits[:, k:])
        lf = logits.reshape(-1, V).float()
        tf = targets.reshape(-1)
        if loss_mask is None:
            ce = F.cross_entropy(lf, tf, ignore_index=-100)
            n_tokens = (tf != -100).sum().clamp(min=1)
        else:
            per_tok = F.cross_entropy(lf, tf, ignore_index=-100, reduction="none")
            m = loss_mask.reshape(-1).to(per_tok.dtype)
            ce = (per_tok * m).sum() / m.sum().clamp(min=1.0)
            n_tokens = m.sum().clamp(min=1)
        zl = lf.new_zeros(())
        zc = self.cfg.z_loss if z_loss is None else z_loss
        if zc > 0:
            lse = torch.logsumexp(lf, dim=-1)
            keep = tf != -100
            if keep.any():
                zl = zc * (lse[keep] ** 2).mean()
        return ce + zl

    # ---------------- сервис ----------------
    def count_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def estimate_mfu(self, tokens_per_sec: float, device_type: str = "cuda") -> float:
        """Доля теоретического пика FLOPs. Плотность матричных умножений ~ 6ND."""
        n = self.count_params(non_embedding=True)
        flops_per_token = 6 * n + 12 * self.cfg.n_layer * self.cfg.n_embd * self.cfg.block_size
        peak = {
            "cuda": 312e12,   # A100 bf16
            "cpu": 2 * torch.get_num_threads() * 3e9 * 8,  # очень грубо
        }.get(device_type, 1e12)
        return tokens_per_sec * flops_per_token / peak

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "config": self.cfg.to_dict(),
            "model": self.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device="cpu", strict: bool = True):
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = GPTConfig.from_dict(payload["config"])
        model = cls(cfg)
        model.load_state_dict(payload["model"], strict=strict)
        model.eval()
        return model.to(device), payload


def load_any(path: str, device: str = "cpu", prefer_ema: bool = True):
    """Загружает и чекпойнт обучения, и чистый {'config','model'}.

    Какие веса брать — решает `best_source`, который записал train.py по
    результату eval: в начале прогона EMA ещё не разошлась и бывает хуже raw
    (замер на шаге 2000: raw 3.4737 против ema 3.5599). Если `best_source`
    в чекпойнте нет, берём EMA — на длинных прогонах она почти всегда лучше.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPTConfig.from_dict(payload["config"])
    model = GPT(cfg)
    if "model" in payload:
        model.load_state_dict(payload["model"])
    want_ema = prefer_ema and payload.get("best_source", "ema") != "raw"
    ema = payload.get("ema") if want_ema else None
    if ema:
        sd = model.state_dict()
        for k, v in ema["shadow"].items():
            if k in sd:
                sd[k].copy_(v.to(sd[k].dtype))
    model.eval()
    return model.to(device), payload
