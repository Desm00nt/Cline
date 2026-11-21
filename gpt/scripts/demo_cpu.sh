#!/usr/bin/env bash
# Полный цикл на CPU за ~1 минуту: токенайзер -> pretrain -> SFT -> чат.
# Демо-корпус маленький, поэтому модель ЗАПОМИНАЕТ 22 диалога, а не понимает их.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

echo "==> 1/4 токенайзер и упаковка корпуса"
$PY scripts/prepare_data.py --input data/raw --out data/prepared --num-merges 2000

echo "==> 2/4 pretrain (tiny, 1.3M параметров)"
$PY train.py --preset tiny --data-dir data/prepared --out out/demo-tiny \
    --steps 400 --batch-size 4 --stride 64 --block-size 256 \
    --eval-every 100 --log-every 50 --num-workers 0

echo "==> 3/4 SFT на демо-диалогах"
$PY sft.py --ckpt out/demo-tiny/model_final.pt --data data/sft/demo_chat.jsonl \
    --out out/demo-sft --epochs 25 --batch-size 4 --block-size 256 --lr 1e-3 --log-every 25

echo "==> 4/4 чат (Ctrl-D для выхода)"
echo "Попробуйте спросить: Что такое трансформер? / Почему небо голубое?"
$PY chat.py --ckpt out/demo-sft/model_sft_ema.pt --max-new 80
