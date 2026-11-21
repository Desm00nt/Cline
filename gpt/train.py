#!/usr/bin/env python3
"""Обучение модели с нуля (pretrain).

Пример для GPU:
    python train.py --preset m30 --data-dir data/prepared --out out/m30 \
        --steps 30000 --batch-size 32 --grad-accum 2 --lr 3e-3

Пример для CPU (крошечная модель, проверка пайплайна):
    python train.py --preset tiny --data-dir data/prepared --out out/tiny \
        --steps 300 --batch-size 8 --block-size 256
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from typing import Optional

# KMP_BLOCKTIME=0 отключает ожидание в spin-wait у потоков Intel OpenMP между
# параллельными областями. На 2 ядрах без этого шаг обучения на ~4% медленнее
# (замер: 2044 -> 2133 ток/с на `small`). Ставим до импорта torch, иначе
# переменную уже никто не прочитает. На не-Intel OpenMP просто игнорируется.
os.environ.setdefault("KMP_BLOCKTIME", "0")

import numpy as np
import torch
from torch.utils.data import DataLoader

from minigpt.config import GPTConfig, get_preset
from minigpt.data import PackedDataset
from minigpt.model import GPT, load_any
from minigpt.muon import EMA, build_optimizer, lr_multiplier
from minigpt.tokenizer import BPETokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="m30", help="tiny|small|m30|m30-deep")
    p.add_argument("--data-dir", default="data/prepared")
    p.add_argument("--out", default="out/run")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--block-size", type=int, default=0, help="0 = взять из пресета")
    p.add_argument("--stride", type=int, default=0, help="шаг окон датасета, 0 = block_size")

    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--muon-lr-scale", type=float, default=5.0,
                   help="множитель lr для Muon относительно AdamW (см. muon.py)")
    p.add_argument("--layer-decay", type=float, default=1.0)
    p.add_argument("--schedule", choices=["wsd", "cosine", "const"], default="wsd")
    p.add_argument("--warmup", type=int, default=0, help="0 = 3%% от steps")
    p.add_argument("--decay-len", type=int, default=0, help="для wsd, 0 = 20%% от steps")
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)

    p.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--z-loss", type=float, default=1e-4)
    p.add_argument("--softcap", type=float, default=0.0)

    p.add_argument("--ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ema-horizon-frac", type=float, default=0.1,
                   help="горизонт EMA как доля от --steps (decay урезается автоматически)")

    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0,
                   help="потоков CPU, 0 = min(4, все ядра); уменьшите, чтобы "
                        "работать за машиной во время обучения")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample-prompt", default="Однажды ")
    return p.parse_args()


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, device, limit: int, amp_dtype) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                            enabled=(amp_dtype != torch.float32)):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    n_threads = args.threads or max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(max(1, n_threads))
    print(f"поток CPU: {torch.get_num_threads()} | устройство: {device}")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_dtype = torch.bfloat16 if args.precision == "bf16" or (
        args.precision == "auto" and device == "cuda") else torch.float32

    # ---------------- данные ----------------
    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise SystemExit(f"нет {tok_path}: сначала запустите scripts/prepare_data.py")
    tok = BPETokenizer.load(tok_path)

    cfg: GPTConfig = get_preset(args.preset)
    cfg.vocab_size = tok.vocab_size
    if args.block_size:
        cfg.block_size = args.block_size
    cfg.dropout = args.dropout
    cfg.z_loss = args.z_loss
    cfg.logit_softcapping = args.softcap

    block = cfg.block_size
    stride = args.stride or block
    train_ds = PackedDataset(os.path.join(args.data_dir, "train.bin"), block, stride)
    val_path = os.path.join(args.data_dir, "val.bin")
    try:
        val_ds = PackedDataset(val_path, block, stride) if os.path.exists(val_path) else train_ds
    except ValueError as e:
        print(f"ВНИМАНИЕ: {e}\n  использую train как val (корпус слишком мал для разделения)")
        val_ds = train_ds

    if len(train_ds) < args.batch_size:
        print(f"ВНИМАНИЕ: примеров ({len(train_ds)}) меньше batch_size ({args.batch_size}), "
              f"уменьшаю batch_size. Добавьте данных или уменьшите --block-size.")
        args.batch_size = max(1, len(train_ds))

    g = torch.Generator()
    g.manual_seed(args.seed)
    nw = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=(device == "cuda"),
                              drop_last=len(train_ds) >= args.batch_size, generator=g,
                              persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=min(args.batch_size, len(val_ds)), shuffle=False,
                            num_workers=0, drop_last=len(val_ds) >= args.batch_size)

    # ---------------- модель ----------------
    model = GPT(cfg).to(device)
    n_params = model.count_params()
    n_nonemb = model.count_params(non_embedding=True)
    print(f"устройство: {device} | точность: {amp_dtype} | пресет: {args.preset}")
    print(f"параметров: {n_params/1e6:.2f}M (без эмбеддинга {n_nonemb/1e6:.2f}M) | "
          f"словарь: {cfg.vocab_size} | контекст: {cfg.block_size}")
    print(f"данных: train={len(train_ds)} окон, val={len(val_ds)} окон, block={block}, stride={stride}")

    tokens_per_step = args.batch_size * args.grad_accum * block
    total_tokens = tokens_per_step * args.steps
    print(f"токенов за прогон: {total_tokens/1e6:.2f}M | шагов: {args.steps} | "
          f"эффективный batch: {args.batch_size*args.grad_accum}")

    opt = build_optimizer(model, args.lr, args.weight_decay, tuple(args.betas),
                          use_muon=(args.optimizer == "muon"),
                          muon_lr_scale=args.muon_lr_scale, layer_decay=args.layer_decay)

    # Горизонт EMA не должен превышать заметную долю длины прогона, иначе
    # среднее так и останется около инициализации (горизонт = 1/(1-decay)).
    ema = None
    if args.ema:
        horizon = max(10.0, args.ema_horizon_frac * args.steps)
        ema_decay = min(args.ema_decay, 1.0 - 1.0 / horizon)
        ema = EMA(model, ema_decay)
        print(f"EMA: decay={ema_decay:.5f} (горизонт ~{1/(1-ema_decay):.0f} шагов)")

    raw_model = model
    if args.compile:
        if sys.platform == "win32":
            print("ВНИМАНИЕ: torch.compile на Windows требует MSVC и Triton, "
                  "который официально не поддерживается. Если дальше упадёт "
                  "CppCompileError — уберите --compile, он даёт 10-20%, не больше.")
        model = torch.compile(model)

    step0, best_val, best_source = 0, float("inf"), "raw"
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["opt"])
        if ema is not None and payload.get("ema"):
            ema.load_state_dict(payload["ema"])
        step0 = payload.get("step", 0)
        best_val = payload.get("best_val", float("inf"))
        print(f"продолжаем с шага {step0}")

    warmup = args.warmup or max(1, int(0.03 * args.steps))
    decay_len = args.decay_len or max(1, args.steps // 5)

    def save(step: int, val_loss: float, tag: str = "ckpt.pt"):
        payload = {
            "config": cfg.to_dict(), "model": raw_model.state_dict(),
            "opt": opt.state_dict(), "step": step, "best_val": best_val,
            "val_loss": val_loss, "best_source": best_source, "args": vars(args),
        }
        if ema is not None:
            payload["ema"] = ema.state_dict()
        torch.save(payload, os.path.join(args.out, tag))
        with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)

    # ---------------- цикл ----------------
    model.train()
    it = infinite(train_loader)
    t0 = time.time()
    running, n_seen = 0.0, 0
    tokens_since_log = 0

    for step in range(step0, args.steps):
        mult = lr_multiplier(step, warmup, args.steps, decay_len, args.min_lr_frac, args.schedule)
        opt.set_lr_scale(mult)
        opt.zero_grad()

        micro_loss = 0.0
        for _ in range(args.grad_accum):
            batch = next(it)
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                enabled=(amp_dtype != torch.float32)):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            loss.backward()
            micro_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.clip_grad)
        if not torch.isfinite(grad_norm):
            print(f"[шаг {step}] взорвались градиенты, шаг пропущен")
            opt.zero_grad()
            continue
        opt.step()
        if ema is not None:
            ema.update(raw_model)

        running += micro_loss
        n_seen += 1
        tokens_since_log += tokens_per_step

        if (step + 1) % args.log_every == 0 or step == step0:
            dt = time.time() - t0
            tps = tokens_since_log / max(1e-9, dt)
            print(f"[шаг {step+1}/{args.steps}] loss={running/n_seen:.4f} "
                  f"lr={args.lr*mult:.2e} |g|={float(grad_norm):.2f} "
                  f"{tps:,.0f} ток/с" + (f" MFU={raw_model.estimate_mfu(tps, 'cuda')*100:.1f}%"
                                        if device == 'cuda' else ""), flush=True)
            running, n_seen, t0, tokens_since_log = 0.0, 0, time.time(), 0

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            val = evaluate(raw_model, val_loader, device, args.eval_batches, amp_dtype)
            ema_val = float("nan")
            if ema is not None:
                backup = ema.copy_to(raw_model)
                ema_val = evaluate(raw_model, val_loader, device, args.eval_batches, amp_dtype)
                EMA.restore(raw_model, backup)
            best = min(v for v in (val, ema_val) if v == v)
            print(f"  eval: val_loss={val:.4f} (ppl={math.exp(min(20, val)):.1f})"
                  + (f" | ema={ema_val:.4f} (ppl={math.exp(min(20, ema_val)):.1f})" if ema else ""),
                  flush=True)
            if best < best_val:
                best_val = best
                best_source = "ema" if (ema_val == ema_val and ema_val < val) else "raw"
                save(step + 1, best, tag="best.pt")
                print(f"  сохранён лучший чекпойнт (val={best:.4f}, веса: {best_source})")
            model.train()

        if (step + 1) % args.save_every == 0:
            save(step + 1, best_val)

    save(args.steps, best_val)
    torch.save({"config": cfg.to_dict(), "model": raw_model.state_dict()},
               os.path.join(args.out, "model_raw.pt"))
    if ema is not None:
        backup = ema.copy_to(raw_model)
        torch.save({"config": cfg.to_dict(), "model": raw_model.state_dict()},
                   os.path.join(args.out, "model_ema.pt"))
        EMA.restore(raw_model, backup)

    # финальная модель = те веса, что дали лучшую val loss
    src = "model_ema.pt" if (best_source == "ema" and ema is not None) else "model_raw.pt"
    shutil.copyfile(os.path.join(args.out, src), os.path.join(args.out, "model_final.pt"))
    print(f"лучшие веса: {best_source} (val={best_val:.4f}) -> model_final.pt")

    # ---------------- пример генерации ----------------
    print(f"\n--- пример генерации (веса: {best_source}) ---")
    final_model, _ = load_any(os.path.join(args.out, "model_final.pt"), device)
    from minigpt.sampling import generate
    ids = tok.encode(args.sample_prompt)
    with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                        enabled=(amp_dtype != torch.float32)):
        out = generate(final_model, ids, max_new_tokens=80, temperature=0.8, top_k=40)
    print(args.sample_prompt + tok.decode(out))
    print(f"\nготово. чекпойнты в {args.out}")


if __name__ == "__main__":
    main()
