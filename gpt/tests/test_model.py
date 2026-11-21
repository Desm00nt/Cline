import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minigpt.config import GPTConfig, get_preset
from minigpt.model import GPT, KVCache, apply_rope, build_rope_cache

torch.manual_seed(0)


def tiny_cfg(**kw):
    base = dict(vocab_size=512, block_size=64, n_layer=2, n_head=4, n_kv_head=2,
                n_embd=64, sliding_window=32, global_every=2, qk_norm=True)
    base.update(kw)
    return GPTConfig(**base)


def test_forward_shapes():
    model = GPT(tiny_cfg())
    x = torch.randint(0, 512, (3, 17))
    y = torch.randint(0, 512, (3, 17))
    logits, loss = model(x, y)
    assert logits.shape == (3, 17, 512)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_loss_mask_ignores_prompt():
    """Маска должна полностью исключать помеченные нулём позиции из loss."""
    model = GPT(tiny_cfg())
    x = torch.randint(0, 512, (2, 10))
    y = torch.randint(0, 512, (2, 10))
    mask = torch.zeros(2, 10)
    mask[:, 5:] = 1.0
    logits, _ = model(x)
    masked = model.compute_loss(logits, y, loss_mask=mask)
    manual = model.compute_loss(logits[:, 5:], y[:, 5:])
    assert torch.allclose(masked, manual, atol=1e-5)


@pytest.mark.parametrize("pos", [1, 5, 11])
def test_causality(pos):
    """Изменение токена в позиции pos не меняет логиты до него."""
    model = GPT(tiny_cfg()).eval()
    x = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        base, _ = model(x)
        x2 = x.clone()
        x2[0, pos] = (x2[0, pos] + 1) % 512
        alt, _ = model(x2)
    assert torch.allclose(base[0, :pos], alt[0, :pos], atol=1e-5)
    assert not torch.allclose(base[0, pos:], alt[0, pos:], atol=1e-4)


def test_kv_cache_matches_full_forward():
    """Пошаговая генерация с кэшем обязана давать те же логиты, что полный проход."""
    model = GPT(tiny_cfg(sliding_window=0)).eval()
    x = torch.randint(0, 512, (1, 12))
    with torch.no_grad():
        full, _ = model(x)
        cache = KVCache(model.cfg.n_layer, model.cfg.block_size, model.cfg.n_kv_head,
                        model.cfg.head_dim(), device="cpu")
        # префикс кусками 5 + 7, проверяем логит последней позиции
        l1, _ = model(x[:, :5], cache=cache)
        l2, _ = model(x[:, 5:], cache=cache)
    assert cache.length == 12
    assert torch.allclose(full[0, -1], l2[0, -1], atol=2e-4), (
        (full[0, -1] - l2[0, -1]).abs().max().item())


def test_kv_cache_with_sliding_window():
    model = GPT(tiny_cfg(sliding_window=8, global_every=99)).eval()  # все слои SWA
    x = torch.randint(0, 512, (1, 20))
    with torch.no_grad():
        full, _ = model(x)
        cache = KVCache(model.cfg.n_layer, model.cfg.block_size, model.cfg.n_kv_head,
                        model.cfg.head_dim(), device="cpu")
        step, _ = model(x[:, :10], cache=cache)
        last, _ = model(x[:, 10:], cache=cache)
    assert torch.allclose(full[0, -1], last[0, -1], atol=2e-4)


def test_sliding_window_actually_limits_context():
    """Если все слои скользящие, токен вне окна не влияет на поздние позиции."""
    model = GPT(tiny_cfg(sliding_window=4, global_every=99, n_layer=3)).eval()
    x = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        base, _ = model(x)
        x2 = x.clone()
        x2[0, 0] = (x2[0, 0] + 1) % 512      # трогаем самую первую позицию
        alt, _ = model(x2)
    # позиция 15 видит только 12..15, поэтому её логиты не должны измениться
    assert torch.allclose(base[0, 15], alt[0, 15], atol=1e-5)
    assert not torch.allclose(base[0, 2], alt[0, 2], atol=1e-4)


def test_rope_preserves_norm():
    cos, sin = build_rope_cache(32, 16, 10000.0)
    q = torch.randn(2, 4, 32, 16)
    out = apply_rope(q, cos, sin)
    assert torch.allclose(q.norm(dim=-1), out.norm(dim=-1), atol=1e-5)
    # разные позиции дают разные повороты
    assert not torch.allclose(out[0, 0, 0], out[0, 0, 1], atol=1e-4)


def test_gradients_finite():
    model = GPT(tiny_cfg())
    x = torch.randint(0, 512, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_save_load_roundtrip(tmp_path):
    model = GPT(tiny_cfg()).eval()
    x = torch.randint(0, 512, (1, 8))
    with torch.no_grad():
        ref, _ = model(x)
    path = str(tmp_path / "m.pt")
    model.save(path)
    loaded, payload = GPT.load(path)
    assert payload["config"]["n_layer"] == model.cfg.n_layer
    with torch.no_grad():
        got, _ = loaded(x)
    assert torch.allclose(ref, got, atol=1e-6)


def test_generate_matches_reforward():
    """Жадная генерация с кэшем == пошаговое полное перевычисление.

    Ловит ошибки в причинности при проходе промпта кусками и в is_causal
    на одно-токенных шагах.
    """
    from minigpt.sampling import generate
    model = GPT(tiny_cfg()).eval()
    ids = torch.randint(0, 512, (12,)).tolist()
    with torch.no_grad():
        got = generate(model, ids, max_new_tokens=8, greedy=True, stop_ids=())
        ctx = list(ids)
        for step in range(8):
            logits, _ = model(torch.tensor([ctx]))
            expected = int(logits[0, -1].argmax())
            assert got[step] == expected, (step, got, expected)
            ctx.append(expected)


def test_m30_param_budget():
    """Флагманский пресет обязан укладываться в 30M параметров."""
    cfg = get_preset("m30")
    model = GPT(cfg)
    n = model.count_params()
    assert n <= 30_000_000, f"m30: {n/1e6:.2f}M > 30M"
    assert n > 20_000_000, f"m30 слишком мал: {n/1e6:.2f}M"


def test_all_presets_sane():
    for name in ("tiny", "small", "m30", "m30-deep"):
        cfg = get_preset(name)
        model = GPT(cfg)
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits, _ = model(x)
        assert logits.shape == (1, 8, cfg.vocab_size)
        assert model.count_params() <= 31_000_000, name


def test_softcap_bounds_logits():
    cfg = tiny_cfg(logit_softcapping=15.0)
    model = GPT(cfg).eval()
    x = torch.randint(0, 512, (1, 8))
    with torch.no_grad():
        logits, _ = model(x)
    assert logits.abs().max().item() <= 15.0 + 1e-4


def test_z_loss_is_nonnegative_and_changes_loss():
    model = GPT(tiny_cfg())
    x = torch.randint(0, 512, (1, 8))
    logits, _ = model(x)
    plain = model.compute_loss(logits, x, z_loss=0.0)
    withz = model.compute_loss(logits, x, z_loss=1e-2)
    assert withz >= plain
