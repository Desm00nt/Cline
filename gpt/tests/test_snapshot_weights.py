"""Снимок весов в fp16: переживает откат песочницы и читается обычным load_any."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from minigpt.config import GPTConfig, asdict  # noqa: E402
from minigpt.model import GPT, load_any  # noqa: E402
from snapshot_weights import snapshot  # noqa: E402


def tiny_cfg():
    return GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=2,
                     n_kv_head=2, n_embd=32, sliding_window=0)


def make_ckpt(path, best_source="ema", step=7):
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    ema = {k: v.clone() * 0.5 for k, v in sd.items()}   # заметно отличается от raw
    torch.save({"config": asdict(cfg), "model": sd,
                "ema": {"shadow": ema, "decay": 0.999},
                "best_source": best_source, "step": step}, path)
    return cfg, sd, ema


def test_snapshot_is_smaller_and_fp16(tmp_path):
    ckpt = str(tmp_path / "ckpt.pt")
    make_ckpt(ckpt)
    out = str(tmp_path / "snap.fp16.pt")
    info = snapshot(ckpt, out)
    assert info["step"] == 7
    assert os.path.getsize(out) < os.path.getsize(ckpt)
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert all(v.dtype == torch.float16 for v in payload["model"].values())
    assert os.path.exists(out + ".json")


def test_load_any_reads_fp16_snapshot(tmp_path):
    """fp16 -> fp32 происходит внутри load_state_dict, доп. кода не нужно."""
    ckpt = str(tmp_path / "ckpt.pt")
    cfg, sd, ema = make_ckpt(ckpt)
    out = str(tmp_path / "snap.fp16.pt")
    snapshot(ckpt, out)
    model, payload = load_any(out)
    assert next(model.parameters()).dtype == torch.float32
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        got = model(x)[0]
    ref = GPT(cfg)
    ref.load_state_dict({k: v.half().float() for k, v in ema.items()})
    ref.eval()
    with torch.no_grad():
        want = ref(x)[0]
    assert torch.allclose(got, want, atol=1e-5)


def test_ema_only_snapshot_drops_raw(tmp_path):
    ckpt = str(tmp_path / "ckpt.pt")
    make_ckpt(ckpt)
    out = str(tmp_path / "ema.fp16.pt")
    snapshot(ckpt, out, ema_only=True)
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert "model" not in payload
    assert "ema" in payload
    assert os.path.getsize(out) < os.path.getsize(ckpt)


def test_raw_best_source_is_preserved(tmp_path):
    """best_source решает, какие веса подставит load_any — снимать надо и его."""
    ckpt = str(tmp_path / "ckpt.pt")
    make_ckpt(ckpt, best_source="raw")
    out = str(tmp_path / "snap.fp16.pt")
    info = snapshot(ckpt, out)
    assert info["best_source"] == "raw"
    model, _ = load_any(out)
    cfg, sd, _ = make_ckpt(str(tmp_path / "ref.pt"), best_source="raw")
    ref = GPT(cfg)
    ref.load_state_dict({k: v.half().float() for k, v in sd.items()})
    ref.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.allclose(model(x)[0], ref(x)[0], atol=1e-5)


def test_clean_export_without_ema(tmp_path):
    """model_final.pt — это {'config','model'} без EMA, и это не ошибка."""
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    src = str(tmp_path / "model_final.pt")
    torch.save({"config": asdict(cfg), "model": model.state_dict()}, src)
    out = str(tmp_path / "snap.fp16.pt")
    info = snapshot(src, out)
    assert info["bytes"] < os.path.getsize(src)
    loaded, _ = load_any(out)
    ref = GPT(cfg)
    ref.load_state_dict({k: v.half().float() for k, v in model.state_dict().items()})
    ref.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.allclose(loaded(x)[0], ref(x)[0], atol=1e-5)
    # а вот --ema-only на таком файле — ошибка
    try:
        snapshot(src, str(tmp_path / "bad.pt"), ema_only=True)
        raise AssertionError("ожидали SystemExit")
    except SystemExit:
        pass


def test_snapshot_can_copy_tokenizer(tmp_path):
    """Снапшот весов без словаря не читается: токенайзер кладётся рядом.

    Регрессия: прогон на 20 000 шагов пропал, потому что tokenizer.json лежал
    в gpt/data/prepared_real/ (в .gitignore) и откат среды его стёр.
    """
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    src = str(tmp_path / "ckpt.pt")
    torch.save({"config": asdict(cfg), "model": model.state_dict()}, src)
    tok_src = tmp_path / "tokenizer.json"
    tok_src.write_text('{"version": 1, "special": [], "merges": []}', encoding="utf-8")
    outdir = tmp_path / "weights"
    outdir.mkdir()
    info = snapshot(src, str(outdir / "snap.fp16.pt"), tokenizer=str(tok_src))
    assert info["tokenizer"] == str(outdir / "tokenizer.json")
    assert os.path.exists(info["tokenizer"])
    assert open(info["tokenizer"], encoding="utf-8").read() == tok_src.read_text(encoding="utf-8")
    # без --tokenizer поля нет и файл не создаётся
    info2 = snapshot(src, str(outdir / "snap2.fp16.pt"))
    assert info2["tokenizer"] == ""
