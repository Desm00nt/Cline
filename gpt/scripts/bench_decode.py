"""Скорость декодирования и физический потолок по памяти.

Декодирование с batch=1 упирается не в вычисления, а в чтение весов из памяти:
на каждый сгенерированный токен нужно прочитать ВСЕ веса модели. Поэтому

    максимум ток/с ≈ пропускная способность памяти / (параметров × байт на вес)

Скрипт измеряет пропускную способность на вашей машине, затем реальную скорость
декодирования в fp32 и int8, и показывает, какую долю потолка мы используем.

    python scripts/bench_decode.py --preset m30 --vocab 8192
    python scripts/bench_decode.py --preset m30 --vocab 8192 --quant int8 --threads 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from minigpt.config import get_preset  # noqa: E402
from minigpt.model import GPT  # noqa: E402
from minigpt.sampling import generate, make_cache  # noqa: E402


def measure_bandwidth_gbs(n_floats: int = 32 * 1024 * 1024, repeats: int = 5) -> float:
    """Пропускная способность памяти на чтение+запись, ГБ/с."""
    x = torch.randn(n_floats)
    y = torch.empty_like(x)
    best = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        y.copy_(x)
        dt = time.perf_counter() - t0
        best = max(best, 2 * x.numel() * x.element_size() / dt / 1e9)
    return best


def decode_tokps(model, cfg, n_tokens: int, prompt: int = 16) -> float:
    ids = list(range(prompt, prompt + 8))
    cache = make_cache(model, torch.device("cpu"))
    generate(model, ids, max_new_tokens=8, cache=cache, greedy=True)  # прогрев
    t0 = time.perf_counter()
    cache = make_cache(model, torch.device("cpu"))
    out = generate(model, ids, max_new_tokens=n_tokens, cache=cache, greedy=True)
    dt = time.perf_counter() - t0
    return (len(out) - len(ids)) / dt


def bench_ort(model, cfg, args, bw: float, n: int) -> None:
    """ONNX Runtime: граф целиком компилируется в несколько fused-ядер.

    Шаг без KV-кэша (T=1, past=0) — это честная мера накладных расходов на
    токене: чтение прошлых K/V добавляет к нему ещё пару десятых миллисекунды
    на длинном контексте.
    """
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    class Step(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, idx):
            return self.m(idx)[0]

    x = torch.randint(0, cfg.vocab_size, (1, 1))
    path = f"/tmp/{args.preset}_step.onnx"
    torch.onnx.export(Step(model), (x,), path, input_names=["idx"],
                      output_names=["logits"],
                      dynamic_axes={"idx": {0: "b", 1: "t"}, "logits": {0: "b", 1: "t"}},
                      opset_version=17)
    so = ort.SessionOptions()
    so.intra_op_num_threads = torch.get_num_threads()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    xi = np.array([[16]], dtype=np.int64)

    for tag, p in [("fp32", path), ("int8", path + ".int8.onnx")]:
        if tag == "int8":
            quantize_dynamic(path, p, weight_type=QuantType.QInt8)
        sess = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        for _ in range(5):
            sess.run(None, {"idx": xi})
        t0 = time.perf_counter()
        for _ in range(100):
            sess.run(None, {"idx": xi})
        tps = 100 / (time.perf_counter() - t0)
        b = 4 if tag == "fp32" else 1
        roof = bw * 1e9 / (n * b)
        print(f"ONNX Runtime {tag:5s}: {tps:7.1f} ток/с ({1000/tps:5.2f} мс/токен) "
              f"| {100*tps/roof:3.0f}% потолка по памяти")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="m30")
    p.add_argument("--vocab", type=int, default=0, help="0 = из пресета")
    p.add_argument("--block-size", type=int, default=0, help="0 = из пресета")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--threads", type=int, default=0, help="0 = min(4, ядра)")
    p.add_argument("--quant", choices=["none", "int8"], default="none")
    p.add_argument("--ort", action="store_true",
                   help="экспортировать шаг декодирования в ONNX и померить "
                        "ONNX Runtime (fused-граф; требует onnx и onnxruntime)")
    p.add_argument("--engine", default="fbgemm", help="fbgemm|x86|onednn|qnnpack")
    args = p.parse_args()

    threads = args.threads or max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)

    bw = measure_bandwidth_gbs()
    print(f"потоков: {threads} | пропускная способность памяти: {bw:.1f} ГБ/с")

    cfg = get_preset(args.preset)
    if args.vocab:
        cfg.vocab_size = args.vocab
    if args.block_size:
        cfg.block_size = args.block_size
    model = GPT(cfg)
    model.eval()
    n = model.count_params()

    if args.quant == "int8":
        try:
            torch.backends.quantized.engine = args.engine
        except Exception as e:
            print(f"движок {args.engine} недоступен ({e}), беру текущий")
        import torch.nn as nn
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        bytes_per_weight = 1
        tag = f"int8 ({torch.backends.quantized.engine})"
    else:
        bytes_per_weight = 4
        tag = "fp32"

    tps = decode_tokps(model, cfg, args.tokens)
    mb_tok = n * bytes_per_weight / 1e6
    roofline = bw * 1e9 / (n * bytes_per_weight)
    print(f"{args.preset} {n/1e6:.2f}M параметров, {tag}: "
          f"{tps:7.1f} ток/с ({1000/tps:5.2f} мс/токен)")
    print(f"  весов на токен: {mb_tok:.1f} МБ | потолок по памяти: "
          f"{roofline:7.1f} ток/с | использовано {100*tps/roofline:.0f}%")

    if args.ort:
        print()
        bench_ort(GPT(cfg), cfg, args, bw, n)

    print("\nпотолок для этой модели при разной точности весов "
          f"(при {bw:.1f} ГБ/с; замер на copy, а чистое чтение весов чуть "
          f"быстрее, поэтому >100%% бывает):")
    for name, b in [("fp32", 4), ("int8", 1), ("int4", 0.5)]:
        print(f"  {name:5s} ({b} Б/вес): {bw*1e9/(n*b):7.0f} ток/с")


if __name__ == "__main__":
    main()
