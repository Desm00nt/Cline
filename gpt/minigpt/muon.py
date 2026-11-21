"""Оптимизация: Muon + AdamW, расписание WSD, EMA и клиппинг.

Главная «тяжёлая артиллерия» здесь — **Muon** (MomentUm Orthogonalized by
Newton-Schulz). Идея: у градиента скрытой матрицы почти вся энергия сидит в
нескольких главных направлениях, и обычный SGD/Adam делает шаги в основном
вдоль них. Muon заменяет буфер момента на его ортогонализацию (все сингулярные
числа -> 1), поэтому за шаг обновляются *все* направления одинаково. На
практике это заметно ускоряет сходимость матричных слоёв при том же бюджете
шагов — именно то, что нужно модели на 30M параметров и ограниченном объёме
данных.

Эмбеддинги, финальная голова и все одномерные параметры (RMSNorm) остаются на
AdamW: для них ортогонализация не имеет смысла.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Ортогонализация через квинтичную итерацию Ньютона-Шульца
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Приближённо заменяет G на U V^T из SVD (G = U S V^T).

    Итерация не сходится к точной ортогонализации — она намеренно «недоходит»,
    потому что точность не нужна, а 5 шагов без SVD дают большой выигрыш.
    Коэффициенты подобраны так, чтобы сингулярные числа быстро подтягивались к 1.

    Замечание: батчевать ортогонализацию по матрицам одинаковой формы (один bmm
    вместо N вызовов) пробовали — на CPU это не быстрее (0.79x на 1.8M, 0.92x на
    30M), потому что копирование в стек съедает выигрыш. Оставлен простой путь.
    """
    assert G.ndim == 2, "ожидается матрица (A, B)"
    a, b, c = (3.4445, -4.7750, 2.0315)
    # bf16 быстрее на GPU; на CPU он медленный, поэтому там считаем в fp32
    dt = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(dt)
    transposed = False
    if X.size(-2) > X.size(-1):          # итерируем по меньшей стороне
        X = X.mT
        transposed = True
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(Optimizer):
    """SGD с моментом, у которого градиент ортогонализуется на каждом шаге."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.015,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                shape = p.shape
                g2 = g.view(shape[0], -1) if g.ndim > 2 else g
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g2)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g2, alpha=1 - momentum)
                # lerp НЕ на месте: иначе мы портим p.grad, и накопление
                # градиента (grad accumulation) даёт неверный результат
                upd = g2.lerp(buf, momentum) if group["nesterov"] else buf
                upd = zeropower_via_newtonschulz5(upd.contiguous(), steps=group["ns_steps"])
                # Moonlight-масштаб: RMS обновления ~= 0.2 независимо от формы
                # матрицы. Поэтому шаг = lr * upd и lr_muon ~= 5 * lr_adam,
                # чтобы размер шага совпадал с AdamW-рецептом.
                adjust = 0.2 * math.sqrt(max(shape[0], math.prod(shape[1:]) or 1))
                if wd != 0.0:
                    p.mul_(1 - lr * wd)
                p.add_(upd.view(shape), alpha=-lr * adjust)
        return loss


# ---------------------------------------------------------------------------
# Сборка оптимизатора
# ---------------------------------------------------------------------------

def param_groups(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.05,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
    muon_lr_scale: float = 5.0,
    muon_momentum: float = 0.95,
    use_muon: bool = True,
    layer_decay: float = 1.0,
) -> List[Dict]:
    """Разбивает параметры: 2D-матрицы блоков -> Muon, остальное -> AdamW."""
    muon_params, adam_params = [], []
    n_blocks = max(1, len([m for m in model.modules() if m.__class__.__name__ == "Block"]))
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.ndim >= 2 and "blocks." in name
        if use_muon and is_matrix:
            # более поздние слои получают чуть больший lr (layer_decay < 1)
            mult = 1.0
            if layer_decay != 1.0:
                idx = int(name.split("blocks.")[1].split(".")[0])
                mult = layer_decay ** (n_blocks - 1 - idx)
            muon_params.append({"params": [p], "lr": lr * muon_lr_scale * mult, "name": name})
        else:
            adam_params.append({
                "params": [p], "name": name,
                "lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps,
            })
    return muon_params + adam_params


class HybridOptimizer:
    """Обёртка, которая гоняет Muon и AdamW как один оптимизатор."""

    def __init__(self, muon_groups: List[Dict], adam_groups: List[Dict]):
        self.opts: List[Optimizer] = []
        if muon_groups:
            self.opts.append(Muon(
                [{"params": g["params"], "lr": g["lr"], "name": g.get("name")} for g in muon_groups],
                lr=muon_groups[0]["lr"], momentum=0.95, weight_decay=0.0,
            ))
        if adam_groups:
            self.opts.append(torch.optim.AdamW(adam_groups, lr=adam_groups[0]["lr"]))
        self.state_dict = self._state_dict
        self.load_state_dict = self._load_state_dict

    @property
    def param_groups(self) -> List[Dict]:
        out = []
        for o in self.opts:
            out.extend(o.param_groups)
        return out

    def _state_dict(self):
        return [o.state_dict() for o in self.opts]

    def _load_state_dict(self, sd):
        for o, s in zip(self.opts, sd):
            o.load_state_dict(s)

    def zero_grad(self, set_to_none: bool = True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def set_lr_scale(self, scale: float):
        """Умножает текущий lr каждого параметра на scale (для расписания)."""
        for o in self.opts:
            for g in o.param_groups:
                g["lr"] = g.get("base_lr", g["lr"]) * scale

    def freeze_base_lrs(self):
        for o in self.opts:
            for g in o.param_groups:
                g["base_lr"] = g["lr"]


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float = 0.05,
                    betas=(0.9, 0.95), use_muon: bool = True, muon_lr_scale: float = 5.0,
                    layer_decay: float = 1.0) -> HybridOptimizer:
    groups = param_groups(model, lr, weight_decay, betas, muon_lr_scale=muon_lr_scale,
                          use_muon=use_muon, layer_decay=layer_decay)
    muon_g = [g for g in groups if "weight_decay" not in g]
    adam_g = [g for g in groups if "weight_decay" in g]
    opt = HybridOptimizer(muon_g, adam_g)
    opt.freeze_base_lrs()
    return opt


# ---------------------------------------------------------------------------
# Расписание и EMA
# ---------------------------------------------------------------------------

def lr_multiplier(step: int, warmup: int, total: int, decay_len: int = 0,
                  min_frac: float = 0.1, schedule: str = "wsd") -> float:
    """Возвращает множитель базового lr.

    wsd   — warmup / stable / линейный спад (лучше всего для коротких прогонов,
            позволяет продолжать обучение из чекпойнта)
    cosine — классический косинусный спад
    const  — постоянный (для отладки)
    """
    if step < warmup:
        return max(1e-8, (step + 1) / max(1, warmup))
    if schedule == "const":
        return 1.0
    if schedule == "wsd":
        decay_len = decay_len or max(1, total // 5)
        decay_start = max(warmup, total - decay_len)
        if step < decay_start:
            return 1.0
        frac = (step - decay_start) / max(1, decay_len)
        return max(min_frac, 1.0 - (1.0 - min_frac) * min(1.0, frac))
    # cosine
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog))


class EMA:
    """Экспоненциальное среднее весов.

    На маленьких моделях с шумным градиентом EMA стабильно даёт минус к
    validation loss — дешёвый способ «выжать» качество без доп. данных.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def copy_to(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Возвращает текущие веса и подставляет EMA-веса (для eval)."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
        return backup

    @staticmethod
    def restore(model: torch.nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = {k: v.clone() for k, v in sd["shadow"].items()}
