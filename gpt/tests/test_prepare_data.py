"""Тесты подготовки данных: потоковое кодирование и граница train/val."""

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from prepare_data import EOS_ID, copy_range, encode_stream, iter_docs, list_files  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def make_tiny_tokenizer():
    texts = ["мама мыла раму и окно", "рама была большая и светлая",
             "окно смотрело на сад и на реку"]
    return BPETokenizer.train(texts, 20, verbose=False)


def write_corpus(root, docs_per_file=2, files=3):
    for i in range(files):
        (root / f"{i:03d}.txt").write_text(
            "\n\n".join([f"документ {i}-{j} про раму и окно" for j in range(docs_per_file)]),
            encoding="utf-8")


def test_list_files_skips_docs_subdir(tmp_path):
    """fetch_corpus пишет и corpus.txt, и docs/ — иначе корпус удваивается."""
    (tmp_path / "corpus.txt").write_text("текст", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "000000.txt").write_text("тот же текст", encoding="utf-8")
    kept = [f for f, _ in list_files(str(tmp_path), "/docs/")]
    assert len(kept) == 1
    assert kept[0].endswith("corpus.txt")
    both = [f for f, _ in list_files(str(tmp_path), "")]
    assert len(both) == 2


def test_iter_docs_strips_and_drops_empty(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("  первый  \n\n\n\nвторой\n\n\n", encoding="utf-8")
    assert list(iter_docs(str(p), "\n\n")) == ["первый", "второй"]


def test_encode_stream_appends_eos_and_tracks_boundaries(tmp_path):
    tok = make_tiny_tokenizer()
    docs = tmp_path / "a.txt"
    docs.write_text("мама мыла раму\n\nокно было светлое", encoding="utf-8")
    tmp_bin = str(tmp_path / "all.bin")
    total, ends = encode_stream([str(docs)], "\n\n", tok, tmp_bin, np.uint16)
    assert len(ends) == 2
    assert ends[-1] == total
    data = np.fromfile(tmp_bin, dtype=np.uint16)
    assert data.size == total
    # каждый документ заканчивается EOS
    assert data[ends[0] - 1] == EOS_ID
    assert data[ends[1] - 1] == EOS_ID


def test_copy_range_preserves_tokens(tmp_path):
    src = tmp_path / "src.bin"
    arr = np.arange(1000, dtype=np.uint16)
    arr.tofile(str(src))
    dst = str(tmp_path / "dst.bin")
    n = copy_range(str(src), dst, 300, 900, 2)
    assert n == 600
    assert np.array_equal(np.fromfile(dst, dtype=np.uint16), arr[300:900])


def test_train_val_split_lands_on_document_boundary(tmp_path):
    """Val не должен начинаться с середины документа: иначе валидация подсматривает."""
    src = tmp_path / "raw"
    src.mkdir()
    write_corpus(src)
    out = tmp_path / "prepared"
    r = subprocess.run(
        [PY, os.path.join(HERE, "..", "scripts", "prepare_data.py"),
         "--input", str(src), "--out", str(out), "--num-merges", "20",
         "--val-frac", "0.3", "--max-train-chars", "2000"],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["train_tokens"] + meta["val_tokens"] >= 2 * meta["docs"]
    train = np.fromfile(str(out / "train.bin"), dtype=np.uint16)
    val = np.fromfile(str(out / "val.bin"), dtype=np.uint16)
    assert train.size == meta["train_tokens"]
    assert val.size == meta["val_tokens"] > 0
    # последний токен train — EOS: разрез ровно по границе документа
    assert train[-1] == EOS_ID
    assert val[-1] == EOS_ID
    # временный файл убран
    assert not (out / "all_tokens.tmp.bin").exists()
