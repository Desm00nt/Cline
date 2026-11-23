#!/usr/bin/env python3
"""Кладёт компактный снимок весов туда, куда дотягивается git.

Зачем: gpt/out/ в .gitignore, поэтому откат песочницы стирает все чекпойнты —
так уже пропали 21.5M токенов обучения (14 000 шагов, ppl 15.7). Снимок весов
в fp16 весит 26 МБ (EMA + raw) или 13 МБ (--ema-only) и переживает откат.

    python scripts/snapshot_weights.py --ckpt out/small-v2/ckpt.pt \
        --out weights/small-v2-step4000.fp16.pt

Обратно снимок читает обычный minigpt.model.load_any: copy_ внутри
load_state_dict приводит fp16 к fp32.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def to_fp16(state: dict) -> dict:
    return {k: (v.half() if torch.is_floating_point(v) else v) for k, v in state.items()}


def snapshot(ckpt_path: str, out_path: str, ema_only: bool = False) -> dict:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise SystemExit(f"{ckpt_path}: нет ключа 'model' — это не чекпойнт обучения")
    snap = {
        "config": payload["config"],
        "model": to_fp16(payload["model"]),
        "best_source": payload.get("best_source", "ema"),
        "step": payload.get("step"),
        "val_loss": payload.get("best_val_loss", payload.get("val_loss")),
        "snapshot_of": os.path.basename(ckpt_path),
    }
    ema = payload.get("ema")
    if ema is not None:
        snap["ema"] = {"shadow": to_fp16(ema["shadow"]), "decay": ema.get("decay")}
    elif not ema_only:
        raise SystemExit(f"{ckpt_path}: нет EMA (--ema-only, чтобы снять только raw)")
    if ema_only:
        snap.pop("model")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(snap, out_path)
    info = {
        "out": out_path,
        "bytes": os.path.getsize(out_path),
        "step": snap.get("step"),
        "best_source": snap["best_source"],
        "keys": sorted(snap),
    }
    with open(out_path + ".json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in info.items() if k != "keys"}, f, ensure_ascii=False, indent=2)
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ema-only", action="store_true",
                   help="только EMA: 13 МБ вместо 26 МБ")
    args = p.parse_args()
    info = snapshot(args.ckpt, args.out, args.ema_only)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
