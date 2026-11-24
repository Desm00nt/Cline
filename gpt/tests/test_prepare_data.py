"""Тесты подготовки данных: потоковое кодирование и разрез train/val."""

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import prepare_data as PD  # noqa: E402
from prepare_data import EOS_ID, encode_stream, iter_docs, list_files  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def make_tiny_tokenizer():
    texts = ["мама мыла раму и окно", "рама была большая и светлая",
             "окно смотрело на сад и на реку"]
    return BPETokenizer.train(texts, 20, verbose=False)


def test_list_files_skips_docs_subdir(tmp_path):
    """fetch_corpus пишет и corpus.txt, и docs/ — иначе корпус удваивается."""
    (tmp_path / "corpus.txt").write_text("текст", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "000000.txt").write_text("тот же текст", encoding="utf-8")
    kept = [f for f, _ in list_files(str(tmp_path), "/docs/")]
    assert len(kept) == 1
    assert kept[0].endswith("corpus.txt")
    assert len(list_files(str(tmp_path), "")) == 2


def test_iter_docs_strips_and_drops_empty(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("  первый  \n\n\n\nвторой\n\n\n", encoding="utf-8")
    assert list(iter_docs(str(p), "\n\n")) == ["первый", "второй"]


def test_iter_docs_streams_across_chunk_boundary(tmp_path, monkeypatch):
    """Разделитель может попасть ровно на границу куска чтения."""
    p = tmp_path / "a.txt"
    text = "мама мыла раму\n\n" + ("окно было светлое и большое\n\n" * 50) + "хвост"
    p.write_text(text, encoding="utf-8")
    monkeypatch.setattr(PD, "READ_CHUNK", 7)
    streamed = list(PD.iter_docs(str(p), "\n\n"))
    monkeypatch.setattr(PD, "READ_CHUNK", 1 << 20)
    whole = list(PD.iter_docs(str(p), "\n\n"))
    assert streamed == whole == [t.strip() for t in text.split("\n\n") if t.strip()]
    assert streamed[-1] == "хвост"


def test_encode_stream_splits_documents(tmp_path):
    tok = make_tiny_tokenizer()
    src = tmp_path / "a.txt"
    docs = ["мама мыла раму и окно"] * 200          # много документов -> val непустой
    src.write_text("\n\n".join(docs), encoding="utf-8")
    n_tr, n_va, ndocs = encode_stream([str(src)], "\n\n", tok,
                                      str(tmp_path / "tr.bin"), str(tmp_path / "va.bin"),
                                      np.uint16, val_frac=0.2, seed=7)
    assert ndocs == 200
    assert n_tr > 0 and n_va > 0
    assert n_tr + n_va == ndocs * (len(tok.encode(docs[0])) + 1)
    for name in ("tr.bin", "va.bin"):
        data = np.fromfile(str(tmp_path / name), dtype=np.uint16)
        assert data.size > 0
        assert data[-1] == EOS_ID, "файл должен заканчиваться границей документа"


def test_val_is_not_just_the_tail_of_the_stream(tmp_path):
    """Регрессия: хвостовой разрез отдавал в val последний источник целиком.

    На реальном корпусе это означало val из одних романов при train на 93% из
    новостей, и валидация показывала смену домена вместо качества модели.
    """
    src = tmp_path / "raw"
    for sub, marker in (("a_news", "новости про заседание и отчёт"),
                        ("z_novels", "роман про любовь и дуэль")):
        d = src / sub
        d.mkdir(parents=True)
        for i in range(40):
            (d / f"{i:03d}.txt").write_text(
                "\n\n".join([f"{marker} часть {j} " + "слов побольше " * 8
                             for j in range(3)]), encoding="utf-8")
    out = tmp_path / "prepared"
    r = subprocess.run(
        [PY, os.path.join(HERE, "..", "scripts", "prepare_data.py"),
         "--input", str(src), "--out", str(out), "--num-merges", "20",
         "--val-frac", "0.3", "--max-train-chars", "4000", "--seed", "11"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["val_tokens"] > 0
    assert not (out / "all_tokens.tmp.bin").exists()

    tok = BPETokenizer.load(str(out / "tokenizer.json"))
    val = np.fromfile(str(out / "val.bin"), dtype=np.uint16)
    text = tok.decode(val.tolist())
    assert "новости" in text, "в val должны попасть документы из первого источника"
    assert "роман" in text, "в val должны попасть документы из второго источника"


def test_reused_tokenizer_gives_same_vocab(tmp_path):
    """--tokenizer обязателен при дообучении: новый словарь сломает эмбеддинги."""
    src = tmp_path / "raw"
    src.mkdir()
    (src / "a.txt").write_text("мама мыла раму и окно\n\nокно смотрело на реку",
                               encoding="utf-8")
    tok = make_tiny_tokenizer()
    tok.save(str(tmp_path / "tok.json"))
    out = tmp_path / "prepared"
    r = subprocess.run(
        [PY, os.path.join(HERE, "..", "scripts", "prepare_data.py"),
         "--input", str(src), "--out", str(out), "--tokenizer", str(tmp_path / "tok.json"),
         "--val-frac", "0.0"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["vocab_size"] == tok.vocab_size
    assert "учим токенайзер" not in r.stdout
