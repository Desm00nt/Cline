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
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def to_fp16(state: dict) -> dict:
    return {k: (v.half() if torch.is_floating_point(v) else v) for k, v in state.items()}


def snapshot(ckpt_path: str, out_path: str, ema_only: bool = False,
             tokenizer: str = "", source: str = "both") -> dict:
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
    # В git влезает ~32 МБ, а model+ema вместе дают 64 МБ, поэтому по
    # умолчанию снимаем только лучший набор весов (best_source из чекпоинта).
    ema = payload.get("ema")
    if source in ("both", "ema"):
        if ema is None:
            if ema_only or source == "ema":
                raise SystemExit(f"{ckpt_path}: просят EMA, но в файле её нет")
        else:
            snap["ema"] = {"shadow": to_fp16(ema["shadow"]), "decay": ema.get("decay")}
    if source == "ema" and ema is not None:
        snap.pop("model")
    elif source == "raw":
        snap.pop("ema", None)
    # чистый экспорт {'config','model'} (model_final.pt) EMA не содержит —
    # это нормально, снимаем только веса
    if ema_only:
        snap.pop("model", None)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(snap, out_path)
    # Токенайзер кладём рядом со снапшотом. Без него веса не читаются: BPE
    # определяет, какой id у какого куска текста, и другой словарь превращает
    # обученные эмбеддинги в шум (замер: ppl 1622 вместо 13.2). Словарь не
    # восстанавливается из весов, а gpt/data/ в .gitignore — так пропал
    # tokenizer.json прогона на 20 000 шагов.
    tok_out = ""
    if tokenizer:
        tok_out = os.path.join(os.path.dirname(out_path) or ".", "tokenizer.json")
        if os.path.abspath(tokenizer) != os.path.abspath(tok_out):
            shutil.copyfile(tokenizer, tok_out)
    info = {
        "out": out_path,
        "bytes": os.path.getsize(out_path),
        "step": snap.get("step"),
        "best_source": snap["best_source"],
        "keys": sorted(snap),
        "tokenizer": tok_out,
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
    p.add_argument("--source", choices=("best", "raw", "ema", "both"), default="best",
                   help="best = только лучший набор весов (32 МБ, влезает в git); "
                        "both = model+ema (64 МБ)")
    p.add_argument("--tokenizer", default="",
                   help="tokenizer.json из каталога данных: копируется рядом со "
                        "снапшотом, иначе веса нечем прочитать после отката среды")
    args = p.parse_args()
    source = args.source
    if source == "best":
        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        source = "ema" if payload.get("best_source", "ema") != "raw" and payload.get("ema") else "raw"
        if source == "ema" and payload.get("ema") is None:
            source = "raw"
    info = snapshot(args.ckpt, args.out, args.ema_only, args.tokenizer, source)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
