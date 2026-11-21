"""Сколько будет стоить прогон на вашем железе — до запуска обучения.

Считает по реальной конфигурации пресета (число параметров берётся из модели,
а не из документации):

* сколько токенов увидит модель при заданных steps/batch/grad-accum/block-size;
* сколько это FLOPs (6 * N * tokens — стандартная оценка fwd+bwd);
* сколько это времени при заданной реальной пропускной способности;
* сколько байтов весят веса, градиенты, состояния оптимизатора и EMA
  (активации сюда НЕ входят — их меряем на реальном железе).

Пример:
    python scripts/estimate_run.py --preset m30 --steps 30000 \
        --batch-size 8 --grad-accum 8 --block-size 4096 --tok-per-sec 120000
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from minigpt.config import get_preset  # noqa: E402
from minigpt.model import GPT  # noqa: E402
from minigpt.muon import build_optimizer  # noqa: E402


def static_memory_gb(model: torch.nn.Module, use_muon: bool) -> dict:
    """Память под параметры, градиенты, состояния оптимизатора и EMA (в ГБ).

    Состояния у Muon/AdamW создаются лениво внутри step(), поэтому считаем их
    по типу группы: группы Muon — те, в которых нет weight_decay
    (см. muon.build_optimizer), у Muon один буфер импульса, у AdamW два момента.
    """
    opt = build_optimizer(model, 1e-3, 0.05, (0.9, 0.95), use_muon=use_muon)
    n_matrix = sum(p.numel() for g in opt.param_groups if "weight_decay" not in g
                   for p in g["params"])
    total = model.count_params()
    params_bytes = total * 4          # fp32 мастер-веса
    grad_bytes = total * 4            # градиенты fp32
    if use_muon:
        # Muon: 1 буфер импульса на матрицы, AdamW: 2 состояния на остальное
        states = n_matrix * 4 + (total - n_matrix) * 8
    else:
        states = total * 8
    ema_bytes = total * 4
    gb = 1024 ** 3
    return {
        "params_gb": params_bytes / gb,
        "grads_gb": grad_bytes / gb,
        "states_gb": states / gb,
        "ema_gb": ema_bytes / gb,
        "static_total_gb": (params_bytes + grad_bytes + states + ema_bytes) / gb,
        "matrix_params": n_matrix,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="m30")
    p.add_argument("--vocab-size", type=int, default=0, help="0 = из пресета")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="как в train.py, по умолчанию 1")
    p.add_argument("--block-size", type=int, default=4096)
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--tok-per-sec", type=float, nargs="+",
                   default=[50000, 100000, 200000, 400000],
                   help="реальная пропускная способность, несколько вариантов")
    args = p.parse_args()

    cfg = get_preset(args.preset)
    if args.vocab_size:
        cfg.vocab_size = args.vocab_size
    cfg.block_size = args.block_size

    with torch.device("meta"):
        model = GPT(cfg)
    n = model.count_params()

    tok_per_step = args.batch_size * args.grad_accum * args.block_size
    tokens = tok_per_step * args.steps
    flops = 6 * n * tokens
    mem = static_memory_gb(GPT(cfg), use_muon=(args.optimizer == "muon"))

    print(f"пресет {args.preset}: {n/1e6:.2f}M параметров, словарь {cfg.vocab_size}, "
          f"контекст {cfg.block_size}")
    print(f"токенов за шаг: {tok_per_step:,}   всего за прогон: {tokens/1e9:.2f}B")
    print(f"токенов на параметр: {tokens/n:.0f}x  (Чинчилла-оптимум ~20x)")
    print(f"FLOPs (6ND): {flops/1e18:.1f} EFLOP")
    print()
    print("статическая память (без активаций):")
    print(f"  веса fp32      {mem['params_gb']:.2f} GB")
    print(f"  градиенты      {mem['grads_gb']:.2f} GB")
    print(f"  состояния {args.optimizer:5s} {mem['states_gb']:.2f} GB")
    print(f"  EMA            {mem['ema_gb']:.2f} GB")
    print(f"  ИТОГО          {mem['static_total_gb']:.2f} GB  + активации "
          f"(растут как batch x block_size x n_layer)")
    print()
    print(f"{'ток/с':>10} | {'часов':>8} | {'EFLOP/с':>8} | эквивалент")
    for tps in args.tok_per_sec:
        seconds = tokens / tps
        eff = flops / seconds / 1e12
        print(f"{tps:>10,.0f} | {seconds/3600:>8.1f} | {eff:>8.1f} | "
              f"~{tps*3600/1e9:.2f}B токенов/час")


if __name__ == "__main__":
    main()
