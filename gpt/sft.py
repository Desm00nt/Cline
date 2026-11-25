#!/usr/bin/env python3
"""SFT: дообучение на диалогах, чтобы модель отвечала в формате чата.

    python sft.py --ckpt out/m30/best.pt --data data/sft/chat.jsonl \
        --out out/m30-sft --epochs 3 --lr 3e-4

Формат данных (jsonl, по одному диалогу на строку):
    {"messages": [{"role": "user", "content": "Привет"},
                  {"role": "assistant", "content": "Привет!"}]}

Самый эффективный способ сделать маленькую модель «умной» — положить сюда
дистиллированные ответы большой модели на ваши типичные запросы.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from minigpt.config import get_preset
from minigpt.data import SFTDataset, sft_collate
from minigpt.model import GPT, load_any
from minigpt.muon import EMA, build_optimizer, lr_multiplier
from minigpt.sampling import chat_once
from minigpt.tokenizer import BPETokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="", help="чекпойнт pretrain (best.pt / model_ema.pt)")
    p.add_argument("--preset", default="m30", help="используется, если --ckpt не задан")
    p.add_argument("--data", required=True, nargs="+",
                   help="один или несколько jsonl")
    p.add_argument("--repeat", default="",
                   help="файл:сколько — повторить рукописные диалоги, иначе их "
                        "не слышно среди десятков тысяч корпусных пар")
    p.add_argument("--tokenizer", default="")
    p.add_argument("--out", default="out/sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--system", default="")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-data", default="",
                   help="jsonl для валидации: без него переобучение не видно")
    p.add_argument("--eval-every", type=int, default=0,
                   help="шагов между валидациями (0 = только в конце эпохи)")
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--save-every", type=int, default=0,
                   help="шагов между снапшотами лучшего чекпойнта")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    p.add_argument("--sample", default="Расскажи коротко, что такое нейросеть.")
    return p.parse_args()


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" or (
        args.precision == "auto" and device == "cuda") else torch.float32
    dt_name = device.split(":")[0]

    tok_path = args.tokenizer or "data/prepared/tokenizer.json"
    if not os.path.exists(tok_path):
        raise SystemExit(f"нет токенайзера {tok_path}")
    tok = BPETokenizer.load(tok_path)

    if args.ckpt:
        model, payload = load_any(args.ckpt, device)
        print(f"загружен {args.ckpt} (шаг {payload.get('step', '?')})")
    else:
        cfg = get_preset(args.preset)
        cfg.vocab_size = tok.vocab_size
        model = GPT(cfg).to(device)
        print("обучение SFT с нуля (без pretrain — качество будет низким)")

    # Контекст берём из модели, а не из --block-size: у чекпоинта block_size
    # уже зашит в конфиг, и последовательность длиннее него падает в
    # forward() на assert. --block-size остаётся пределом сверху.
    block = min(args.block_size, model.cfg.block_size)
    if block != args.block_size:
        print(f"block_size модели {model.cfg.block_size}: режем примеры по {block}")
    repeat = {}
    for item in filter(None, (x.strip() for x in args.repeat.split(","))):
        if ":" not in item:
            raise SystemExit(f"--repeat: ждал файл:число, получил {item!r}")
        f, _, k = item.rpartition(":")
        repeat[f] = int(k)
    ds = SFTDataset(args.data, tok, block, system=args.system, repeat=repeat)
    total_tokens = sum(len(ids) for ids, _ in ds.examples)
    print(f"примеров: {len(ds)}, токенов: {total_tokens:,}, block={block}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=sft_collate)
    val_loader = None
    if args.val_data:
        val_ds = SFTDataset(args.val_data, tok, block, system=args.system)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=sft_collate)
        print(f"валидация: {len(val_ds)} примеров из {args.val_data}")

    def evaluate() -> float:
        """Средний loss на ответах ассистента. Без этого переобучение не видно:
        train-loss на SFT падает и тогда, когда модель просто запомнила пары."""
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                if j >= args.eval_batches:
                    break
                x = batch["x"].to(device); y = batch["y"].to(device)
                m = batch["mask"].to(device)
                with torch.autocast(device_type=dt_name, dtype=amp_dtype,
                                    enabled=(amp_dtype != torch.float32)):
                    _, loss = model(x, y, loss_mask=m)
                tot += loss.item(); cnt += 1
        model.train()
        return tot / max(1, cnt)
    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))
    print(f"шагов: {total_steps} ({steps_per_epoch}/эпоху), lr={args.lr}")

    opt = build_optimizer(model, args.lr, args.weight_decay, use_muon=(args.optimizer == "muon"))
    # горизонт EMA — доля от общего числа шагов, иначе на коротком прогоне
    # среднее так и останется около инициализации
    horizon = max(10.0, 0.1 * total_steps)
    ema_decay = min(args.ema_decay, 1.0 - 1.0 / horizon)
    ema = EMA(model, ema_decay)
    print(f"EMA: decay={ema_decay:.5f} (горизонт ~{1/(1-ema_decay):.0f} шагов)")
    model.train()

    best_val, best_step = float("inf"), 0
    step, t0 = 0, time.time()
    running, n = 0.0, 0
    for epoch in range(args.epochs):
        opt.zero_grad()
        for i, batch in enumerate(loader):
            mult = lr_multiplier(step, warmup, total_steps, max(1, total_steps // 5),
                                 min_frac=0.05, schedule="wsd")
            opt.set_lr_scale(mult)
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            m = batch["mask"].to(device)
            with torch.autocast(device_type=dt_name, dtype=amp_dtype,
                                enabled=(amp_dtype != torch.float32)):
                _, loss = model(x, y, loss_mask=m)
                (loss / args.grad_accum).backward()
            running += loss.item()
            n += 1
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                opt.step()
                opt.zero_grad()
                ema.update(model)
                step += 1
                if step % args.log_every == 0:
                    print(f"  эпоха {epoch+1} шаг {step}/{total_steps} "
                          f"loss={running/n:.4f} lr={args.lr*mult:.2e} "
                          f"({step/max(1e-9,(time.time()-t0)):.2f} шаг/с)", flush=True)
                    running, n = 0.0, 0
                if val_loader is not None and args.eval_every and step % args.eval_every == 0:
                    vl = evaluate()
                    print(f"    eval: val_loss={vl:.4f} (ppl={np.exp(vl):.1f})", flush=True)
                    if vl < best_val:
                        best_val, best_step = vl, step
                        backup = ema.copy_to(model)
                        torch.save({"config": model.cfg.to_dict(), "model": model.state_dict(),
                                    "step": step, "val_loss": vl, "best_source": "ema"},
                                   os.path.join(args.out, "best.pt"))
                        EMA.restore(model, backup)
                        print(f"    сохранён лучший чекпойнт (val={vl:.4f}, шаг {step})", flush=True)
        print(f"эпоха {epoch+1} завершена, средняя loss={running/max(1,n):.4f}")
        if val_loader is not None:
            vl = evaluate()
            print(f"  eval в конце эпохи: val_loss={vl:.4f} (ppl={np.exp(vl):.1f})", flush=True)
            if vl < best_val:
                best_val, best_step = vl, step
                backup = ema.copy_to(model)
                torch.save({"config": model.cfg.to_dict(), "model": model.state_dict(),
                            "step": step, "val_loss": vl, "best_source": "ema"},
                           os.path.join(args.out, "best.pt"))
                EMA.restore(model, backup)

    # сохраняем и обычные, и EMA-веса
    torch.save({"config": model.cfg.to_dict(), "model": model.state_dict()},
               os.path.join(args.out, "model_sft.pt"))
    backup = ema.copy_to(model)
    torch.save({"config": model.cfg.to_dict(), "model": model.state_dict()},
               os.path.join(args.out, "model_sft_ema.pt"))
    EMA.restore(model, backup)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(model.cfg.to_dict(), f, ensure_ascii=False, indent=2)

    print("\n--- проверка ---")
    backup = ema.copy_to(model)
    for q in [args.sample]:
        ans_ids = chat_once(model, [{"role": "user", "ids": tok.encode(q)}],
                            max_new_tokens=96, temperature=0.7, top_k=40, greedy=False)
        print(f"В: {q}\nО: {tok.decode(ans_ids).strip()}\n")
    EMA.restore(model, backup)
    print(f"готово -> {args.out}/model_sft_ema.pt")


if __name__ == "__main__":
    main()
