"""MiniGPT-RU: GPT-подобная модель на ~30M параметров, обучаемая с нуля.

Модули:
    config     — конфигурация и пресеты размеров
    model      — архитектура (RoPE, GQA, SwiGLU, скользящее окно, KV-кэш)
    muon       — оптимизаторы (Muon + AdamW), расписание, EMA
    tokenizer  — байт-уровневый BPE без внешних зависимостей
    data       — упаковка корпуса и SFT-разметка
    sampling   — генерация
"""

from .config import GPTConfig, get_preset, PRESETS

__all__ = ["GPTConfig", "get_preset", "PRESETS"]
__version__ = "0.1.0"
