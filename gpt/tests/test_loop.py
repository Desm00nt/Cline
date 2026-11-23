"""Рекурсивная глубина (n_loop): те же блоки прогоняются несколько раз.

Главный риск здесь — KV-кэш: при n_loop>1 один блок применяется повторно, и
если слот кэша не различать, второй проход перезапишет ключи первого, и
генерация по токену разойдётся с полным проходом. Это и проверяется первым.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from minigpt.config import GPTConfig  # noqa: E402
from minigpt.model import GPT, KVCache  # noqa: E402


def cfg(n_layer=2, n_loop=3, **kw):
    base = dict(vocab_size=128, block_size=64, n_layer=n_layer, n_head=2,
                n_kv_head=2, n_embd=32, sliding_window=0, global_every=99,
                n_loop=n_loop)
    base.update(kw)
    return GPTConfig(**base)


def test_cache_slots_counts_every_application():
    assert cfg(n_layer=2, n_loop=1).cache_slots() == 2
    assert cfg(n_layer=2, n_loop=3).cache_slots() == 6
    assert cfg(n_layer=6, n_loop=1).cache_slots() == 6


def test_looping_changes_output():
    """Прогон тех же блоков ещё раз — это не тождественное преобразование."""
    torch.manual_seed(0)
    x = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        once, _ = GPT(cfg(n_loop=1)).eval()(x)
        thrice, _ = GPT(cfg(n_loop=3)).eval()(x)
    assert not torch.allclose(once, thrice, atol=1e-3)


def test_cached_decode_matches_full_forward_when_looping():
    """Ключевой тест: слоты кэша не должны пересекаться между проходами."""
    torch.manual_seed(0)
    model = GPT(cfg(n_layer=2, n_loop=3)).eval()
    x = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        full, _ = model(x)
        cache = KVCache(model.cfg.cache_slots(), model.cfg.block_size,
                        model.cfg.n_kv_head, model.cfg.head_dim(), device="cpu")
        l1, _ = model(x[:, :5], cache=cache)
        l2, _ = model(x[:, 5:], cache=cache)
    assert cache.length == 12
    assert len(cache.k) == 6, "по слоту на каждое применение блока"
    assert torch.allclose(full[0, -1], l2[0, -1], atol=2e-4), (
        (full[0, -1] - l2[0, -1]).abs().max().item())


def test_token_by_token_generation_matches_full_forward():
    """Пошаговая генерация (T=1) — режим chat.py."""
    torch.manual_seed(1)
    model = GPT(cfg(n_layer=2, n_loop=3)).eval()
    prompt = torch.randint(0, 128, (1, 6))
    cache = KVCache(model.cfg.cache_slots(), model.cfg.block_size,
                    model.cfg.n_kv_head, model.cfg.head_dim(), device="cpu")
    seq = prompt.clone()
    with torch.no_grad():
        model(prompt, cache=cache)
        for _ in range(5):
            logits, _ = model(seq[:, -1:], cache=cache)
            nxt = logits[0, -1].argmax().item()
            seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)
        full, _ = model(seq)
        # логиты полного прохода на предпоследней позиции = то, что выбрал кэш
        step_logits, _ = model(seq[:, :-1], cache=KVCache(
            model.cfg.cache_slots(), model.cfg.block_size, model.cfg.n_kv_head,
            model.cfg.head_dim(), device="cpu"))
    assert seq[0, -1].item() == step_logits[0, -1].argmax().item()
    assert torch.allclose(full[0, -2], step_logits[0, -1], atol=2e-3)


def test_gradients_reach_the_tied_block():
    torch.manual_seed(0)
    model = GPT(cfg(n_layer=2, n_loop=3))
    x = torch.randint(0, 128, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    for name, p in model.named_parameters():
        if "blocks.0." in name and p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_tied_depth_is_cheaper_in_parameters_than_untied():
    """6 применений слоя: 2 блока x 3 против 6 отдельных блоков."""
    tied = GPT(cfg(n_layer=2, n_loop=3))
    untied = GPT(cfg(n_layer=6, n_loop=1))
    n_tied = sum(p.numel() for p in tied.parameters())
    n_untied = sum(p.numel() for p in untied.parameters())
    assert n_tied < n_untied
    # экономия ровно на четырёх блоках
    one_block = sum(p.numel() for p in GPT(cfg(n_layer=1)).blocks[0].parameters())
    assert n_untied - n_tied == 4 * one_block


def test_n_loop_one_behaves_like_before():
    """Рефакторинг под n_loop не должен менять обычный путь (n_loop=1)."""
    torch.manual_seed(0)
    model = GPT(cfg(n_layer=3, n_loop=1)).eval()
    x = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        got, _ = model(x)
        cache = KVCache(model.cfg.cache_slots(), model.cfg.block_size,
                        model.cfg.n_kv_head, model.cfg.head_dim(), device="cpu")
        a, _ = model(x[:, :3], cache=cache)
        b, _ = model(x[:, 3:], cache=cache)
    assert torch.allclose(got[0, -1], b[0, -1], atol=2e-4)
