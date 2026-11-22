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


# ---------------------------------------------------------------------------
# Оптимизации инференса: обязаны быть численно эквивалентны исходному пути
# ---------------------------------------------------------------------------

def test_gqa_enable_gqa_matches_repeat_interleave():
    """enable_gqa в SDPA должен давать то же, что физический repeat_interleave."""
    import minigpt.model as M

    torch.manual_seed(3)
    model = GPT(tiny_cfg()).eval()
    x = torch.randint(0, 512, (2, 37))
    y = torch.randint(0, 512, (2, 37))
    with torch.no_grad():
        M._GQA_OK = True
        fast, _ = model(x, y)
        M._GQA_OK = False
        slow, _ = model(x, y)
        M._GQA_OK = None
    assert fast.shape == slow.shape
    # расхождение — только округление разного порядка сложений в ядре
    assert (fast - slow).abs().max().item() < 1e-4


def test_rmsnorm_fused_matches_manual_formula():
    """F.rms_norm на fp32 обязан совпадать с ручной формулой в fp32."""
    from minigpt.model import RMSNorm

    norm = RMSNorm(16)
    norm.weight.data = torch.randn(16)
    x = torch.randn(3, 5, 16) * 3.0
    manual = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight
    assert torch.equal(norm(x), manual)


def test_rmsnorm_still_fp32_under_autocast():
    """Под bf16-autocast нормализация не должна терять точность."""
    from minigpt.model import RMSNorm

    norm = RMSNorm(16)
    x = torch.randn(2, 4, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = norm(x)
    assert out.dtype == torch.float32


def test_apply_rope_skips_noop_cast_but_same_result():
    """Пропуск холостого .float() не должен менять результат ни в fp32, ни в bf16."""
    cos, sin = build_rope_cache(16, 8, 10000.0)
    q32 = torch.randn(1, 2, 16, 8)
    xf = q32.float()
    rot = torch.cat((-xf[..., 4:], xf[..., :4]), dim=-1)
    expected = xf * cos[None, None] + rot * sin[None, None]
    assert torch.equal(apply_rope(q32, cos, sin), expected)

    q16 = q32.to(torch.bfloat16)
    got = apply_rope(q16, cos, sin)
    assert got.dtype == torch.bfloat16
    assert (got.float() - expected).abs().max().item() < 5e-2


def test_swa_mask_equals_causal_when_window_covers_block():
    """Если окно не короче блока, маска окна == причинная маска.

    На этом держится быстрый путь: при T <= sliding_window маска не строится,
    и SDPA уходит на is_causal. Значит результаты обязаны совпадать.
    """
    import torch.nn.functional as F

    model = GPT(tiny_cfg(sliding_window=64, block_size=64))
    T = 32                                  # T <= sliding_window
    m = model._attention_mask(T, 0, torch.device("cpu"), 64)[0, 0]
    assert torch.equal(m, torch.ones(T, T, dtype=torch.bool).tril())

    q = torch.randn(1, 4, T, 16)
    with_mask = F.scaled_dot_product_attention(q, q, q, attn_mask=m)
    with_causal = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    assert (with_mask - with_causal).abs().max().item() < 1e-6


def test_short_block_forward_uses_mask_free_path():
    """Обучающий проход с T <= окном обязан давать тот же результат, что с маской."""
    import torch.nn.functional as F

    torch.manual_seed(11)
    model = GPT(tiny_cfg(sliding_window=64, block_size=64)).eval()
    x = torch.randint(0, 512, (2, 32))
    with torch.no_grad():
        fast, _ = model(x)
        # вручную прогоняем то же самое, но с явной причинной маской
        B, T = x.shape
        mask = torch.ones(T, T, dtype=torch.bool).tril()[None, None]
        h = model.tok_emb(x)
        cos = model.rope_cos[:T].to(h.dtype)
        sin = model.rope_sin[:T].to(h.dtype)
        for blk in model.blocks:
            h = blk(h, cos, sin, None, mask)
        ref = model.head(model.ln_f(h))
    assert (fast - ref).abs().max().item() < 1e-5


def test_load_any_respects_best_source(tmp_path):
    """load_any обязан брать те веса, которые eval признал лучшими.

    В начале прогона EMA ещё не разошлась и хуже raw; брать её «по умолчанию»
    значило бы молча подставить в чат худшую модель.
    """
    from minigpt.model import load_any

    torch.manual_seed(5)
    model = GPT(tiny_cfg())
    raw = {k: v.clone() for k, v in model.state_dict().items()}
    shadow = {k: torch.randn_like(v) for k, v in raw.items()}
    path = str(tmp_path / "ckpt.pt")

    for source, expect_ema in [("raw", False), ("ema", True)]:
        torch.save({"config": model.cfg.to_dict(), "model": raw,
                    "ema": {"shadow": shadow}, "best_source": source}, path)
        got, payload = load_any(path)
        assert payload["best_source"] == source
        # ключ берём несвязанный: tok_emb.weight и head.weight при tie_weights —
        # один тензор, и в state_dict он встречается дважды
        key = "ln_f.weight"
        ref = shadow[key] if expect_ema else raw[key]
        assert torch.equal(got.state_dict()[key], ref), source


def test_load_any_without_best_source_prefers_ema(tmp_path):
    """Старые чекпойнты без best_source — прежнее поведение, берём EMA."""
    from minigpt.model import load_any

    torch.manual_seed(6)
    model = GPT(tiny_cfg())
    raw = {k: v.clone() for k, v in model.state_dict().items()}
    shadow = {k: torch.randn_like(v) for k, v in raw.items()}
    path = str(tmp_path / "old.pt")
    torch.save({"config": model.cfg.to_dict(), "model": raw,
                "ema": {"shadow": shadow}}, path)
    got, _ = load_any(path)
    assert torch.equal(got.state_dict()["ln_f.weight"], shadow["ln_f.weight"])
