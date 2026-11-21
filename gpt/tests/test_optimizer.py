import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minigpt.config import GPTConfig
from minigpt.model import GPT
from minigpt.muon import (EMA, HybridOptimizer, Muon, build_optimizer,
                          lr_multiplier, zeropower_via_newtonschulz5)

torch.manual_seed(0)


def tiny_model():
    cfg = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=4, n_kv_head=2,
                    n_embd=64, sliding_window=16, global_every=2)
    return GPT(cfg)


def test_newtonschulz_orthogonalizes():
    G = torch.randn(48, 32)
    O = zeropower_via_newtonschulz5(G, steps=5)
    assert O.shape == G.shape
    sv = torch.linalg.svdvals(O.float())
    # у ортогонализованной матрицы все сингулярные числа ~ 1
    assert sv.min() > 0.6, sv.min().item()
    assert sv.max() < 1.4, sv.max().item()


def test_newtonschulz_nonsquare_and_dtypes():
    for shape in [(16, 64), (64, 16), (7, 7)]:
        G = torch.randn(*shape)
        O = zeropower_via_newtonschulz5(G, steps=5)
        assert O.shape == shape and torch.isfinite(O).all()


def test_muon_reduces_loss():
    """На выпуклой квадратичной задаче Muon обязан устойчиво снижать loss.

    Точка здесь именно «устойчиво»: у нормализованного шага (RMS ~= 0.2*lr)
    есть шумовой пол, поэтому без затухания lr он не уходит в ноль.
    """
    torch.manual_seed(0)
    x = torch.randn(64, 32)
    target = x @ torch.randn(32, 16)          # точно представимая цель
    lin = nn.Linear(32, 16, bias=False)
    lr = 0.05
    opt = Muon([lin.weight], lr=lr)
    first = last = None
    steps = 300
    for i in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * (1 - i / steps)     # затухание, как в реальном прогоне
        opt.zero_grad()
        loss = ((lin(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.4, (first, last)


def test_param_partition():
    model = tiny_model()
    opt = build_optimizer(model, lr=1e-3, use_muon=True)
    muon_names, adam_names = [], []
    for o in opt.opts:
        for g in o.param_groups:
            (muon_names if isinstance(o, Muon) else adam_names).append(g.get("name"))
    # все 2D-матрицы блоков — в Muon
    for name, p in model.named_parameters():
        if "blocks." in name and p.ndim >= 2:
            assert name in muon_names, name
        else:
            assert name in adam_names, name
    assert len(muon_names) > 0 and len(adam_names) > 0


def test_adamw_only_mode():
    model = tiny_model()
    opt = build_optimizer(model, lr=1e-3, use_muon=False)
    assert len(opt.opts) == 1
    assert not isinstance(opt.opts[0], Muon)


def test_set_lr_scale():
    model = tiny_model()
    opt = build_optimizer(model, lr=1e-3)
    base = [g["base_lr"] for g in opt.param_groups]
    opt.set_lr_scale(0.5)
    now = [g["lr"] for g in opt.param_groups]
    for b, n in zip(base, now):
        assert abs(n - 0.5 * b) < 1e-12
    opt.set_lr_scale(1.0)
    for b, g in zip(base, opt.param_groups):
        assert abs(g["lr"] - b) < 1e-12


def test_lr_multiplier_shapes():
    # warmup растёт от 0 до 1
    assert lr_multiplier(0, warmup=10, total=100) < 0.2
    assert abs(lr_multiplier(9, warmup=10, total=100) - 1.0) < 1e-6
    # cosine заканчивается ровно на min_frac на последнем шаге
    assert abs(lr_multiplier(100, warmup=10, total=100, schedule="cosine", min_frac=0.1) - 0.1) < 1e-9
    assert lr_multiplier(99, warmup=10, total=100, schedule="cosine", min_frac=0.1) < 0.11
    # wsd: стабильно в середине, минимум в конце
    assert abs(lr_multiplier(50, warmup=10, total=100, decay_len=20, schedule="wsd") - 1.0) < 1e-9
    assert abs(lr_multiplier(100, warmup=10, total=100, decay_len=20, schedule="wsd",
                             min_frac=0.1) - 0.1) < 1e-9
    # const
    assert abs(lr_multiplier(50, warmup=1, total=100, schedule="const") - 1.0) < 1e-9


def test_ema_moves_and_restores():
    model = tiny_model()
    ema = EMA(model, decay=0.5)
    before = model.tok_emb.weight.detach().clone()
    with torch.no_grad():
        model.tok_emb.weight.add_(1.0)
    ema.update(model)
    backup = ema.copy_to(model)
    # EMA = 0.5*старое + 0.5*новое
    expected = before + 0.5
    assert torch.allclose(model.tok_emb.weight, expected, atol=1e-6)
    EMA.restore(model, backup)
    assert torch.allclose(model.tok_emb.weight, before + 1.0, atol=1e-6)


def test_model_can_overfit_a_batch():
    """Главный тест «оно вообще учится»: запоминаем один батч до почти нулевой loss."""
    torch.manual_seed(0)
    model = tiny_model()
    x = torch.randint(0, 256, (4, 32))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for i in range(250):
        opt.zero_grad()
        _, loss = model(x, x)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < 0.1, (first, loss.item())
    assert loss.item() < first * 0.05


def test_muon_trains_the_model():
    """Тот же сценарий, но Muon+AdamW: loss должна падать и здесь."""
    torch.manual_seed(0)
    model = tiny_model()
    x = torch.randint(0, 256, (4, 32))
    opt = build_optimizer(model, lr=2e-3, use_muon=True)
    first = None
    for i in range(200):
        opt.set_lr_scale(1.0)
        opt.zero_grad()
        _, loss = model(x, x)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < first * 0.2, (first, loss.item())
    assert loss.item() < 1.0


def test_optimizer_state_roundtrip():
    model = tiny_model()
    opt = build_optimizer(model, lr=1e-3)
    x = torch.randint(0, 256, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    sd = opt.state_dict()
    opt2 = build_optimizer(model, lr=1e-3)
    opt2.load_state_dict(sd)
    # после загрузки состояние момента должно совпадать
    m1 = opt.opts[0].state[opt.opts[0].param_groups[0]["params"][0]]["momentum_buffer"]
    m2 = opt2.opts[0].state[opt2.opts[0].param_groups[0]["params"][0]]["momentum_buffer"]
    assert torch.allclose(m1, m2)
