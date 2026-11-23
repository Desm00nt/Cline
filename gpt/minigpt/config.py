"""Конфигурации модели и пресеты размеров.

Все пресеты подобраны так, чтобы укладываться в заявленный бюджет параметров.
Точное число параметров считает `count_params` в model.py, его проверяет тест
test_model.py::test_preset_param_budget.
"""

from dataclasses import dataclass, asdict, fields
from typing import Any, Dict


@dataclass
class GPTConfig:
    vocab_size: int = 16384        # размер словаря BPE
    block_size: int = 2048         # максимальная длина контекста
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int = 4             # GQA: меньше 8 -> экономия параметров и KV-кэша
    n_embd: int = 512
    mlp_mult: float = 8 / 3        # SwiGLU: ширина MLP = mlp_mult * n_embd
    rope_base: float = 10000.0
    dropout: float = 0.0
    bias: bool = False

    # --- механизмы, которые реально важны на малых моделях ---
    qk_norm: bool = True           # RMSNorm по q,k до RoPE -> стабильные attention-логиты
    sliding_window: int = 1024     # 0 = все слои глобальные
    global_every: int = 3          # каждый N-й слой с полным вниманием
    tie_weights: bool = True       # входной эмбеддинг == выходная голова
    logit_softcapping: float = 0.0 # >0 включает tanh-ограничение логитов (Gemma-style)
    z_loss: float = 0.0            # штраф на logsumexp логитов (стабильность bf16)
    residual_scale_init: bool = True  # делить init выходных проекций на sqrt(2*L)
    init_std: float = 0.02
    # Рекурсивная глубина: сколько раз прогнать ПО ОДНИМ И ТЕМ ЖЕ блокам.
    # n_layer=2, n_loop=3 даёт 6 применений слоя при параметрах двух блоков:
    # глубина растёт, веса — нет. Плата: во столько же раз больше FLOPs на токен
    # против n_layer=2, и отдельный KV-слот на каждое применение (иначе кэш
    # второго прохода перезапишет первый и генерация поедет).
    n_loop: int = 1

    def cache_slots(self) -> int:
        """Сколько независимых KV-слотов нужно: по одному на применение блока."""
        return self.n_layer * max(1, self.n_loop)

    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, "n_embd должно делиться на n_head"
        return self.n_embd // self.n_head

    def mlp_hidden(self) -> int:
        # округляем до 64 -> красиво ложится на тензорные ядра
        h = int(self.mlp_mult * self.n_embd)
        return ((h + 63) // 64) * 64

    def is_global_layer(self, i: int) -> bool:
        if self.sliding_window <= 0:
            return True
        return (i + 1) % self.global_every == 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPTConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Пресеты
# ---------------------------------------------------------------------------

PRESETS: Dict[str, GPTConfig] = {
    # ~0.6M параметров. Обучается на CPU за минуты — для тестов и демо пайплайна.
    "tiny": GPTConfig(
        vocab_size=4096, block_size=512,
        n_layer=4, n_head=4, n_kv_head=2, n_embd=128,
        sliding_window=256, global_every=2,
    ),
    # ~13M. Маленькая модель, которую уже можно гонять на слабом GPU.
    "small": GPTConfig(
        vocab_size=8192, block_size=1024,
        n_layer=6, n_head=6, n_kv_head=3, n_embd=384,
        sliding_window=512, global_every=3,
    ),
    # Рекурсивная глубина: те же 6 применений слоя, что у small, но блоками
    # по два с тройным повтором. Параметров вдвое меньше (веса 25 МБ вместо
    # 51 МБ) при том же числе FLOPs на токен — на CPU, где декодирование
    # упирается в полосу памяти, это прямое ускорение.
    "small-loop": GPTConfig(
        vocab_size=8192, block_size=1024,
        n_layer=2, n_head=6, n_kv_head=3, n_embd=384,
        sliding_window=512, global_every=3, n_loop=3,
    ),
    # Ещё глубже без роста весов: 12 применений тех же двух блоков.
    # Вдвое больше FLOPs на токен, зато проверяет, окупается ли глубина.
    "small-loop-deep": GPTConfig(
        vocab_size=8192, block_size=1024,
        n_layer=2, n_head=6, n_kv_head=3, n_embd=384,
        sliding_window=512, global_every=3, n_loop=6,
    ),
    # Флагман: ~29.9M параметров, контекст 4k. Основной пресет под задачу.
    # Словарь 12288 — компромисс: больше словарь = лучше сжатие русского
    # текста, но эмбеддинги съедают бюджет (vocab*n_embd параметров).
    "m30": GPTConfig(
        vocab_size=12288, block_size=4096,
        n_layer=8, n_head=8, n_kv_head=4, n_embd=512,
        sliding_window=1024, global_every=3,
    ),
    # Вариант «глубже и уже»: 16 слоёв по 384. Обычно чуть лучше рассуждает
    # на том же бюджете, но чуть хуже сжимает текст и медленнее на шаг.
    "m30-deep": GPTConfig(
        vocab_size=8192, block_size=4096,
        n_layer=16, n_head=8, n_kv_head=4, n_embd=384,
        sliding_window=1024, global_every=3,
    ),
}


def get_preset(name: str) -> GPTConfig:
    if name not in PRESETS:
        raise KeyError(f"неизвестный пресет {name!r}, доступны: {sorted(PRESETS)}")
    return PRESETS[name]
