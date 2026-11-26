"""Проверка рукописных диалоговых пар.

Эти файлы пишутся вручную, и в них регулярно попадали посторонние символы
(иероглифы, катакана) и случайно вклеенные английские слова. Тест гоняется
на каждом прогоне pytest, поэтому такая опечатка не доедет до обучения.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import pytest

SFT_DIR = Path(__file__).resolve().parent.parent / "data" / "sft"

# Латинские слова, которые законно встречаются в русском тексте о моделях.
ALLOWED_LATIN = {
    "SFT", "RAG", "Python", "torch", "BPE", "RoPE", "loss", "KV", "top", "k",
    "EMA", "dropout", "Speak", "English", "AdamW", "Muon", "RMSNorm",
    "int8", "int4", "ONNX", "GPU", "CPU",
}

CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def chat_files():
    files = sorted(SFT_DIR.glob("chat_*.jsonl"))
    demo = SFT_DIR / "demo_chat.jsonl"
    if demo.exists():
        files.append(demo)
    return files


def load_pairs():
    pairs = []
    for path in chat_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            pairs.append((path.name, lineno, json.loads(line)))
    return pairs


def test_handwritten_files_exist():
    assert chat_files(), "в data/sft нет ни одного файла chat_*.jsonl"


def test_every_line_is_user_then_assistant():
    for name, lineno, obj in load_pairs():
        assert list(obj) == ["messages"], (name, lineno)
        msgs = obj["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"], (name, lineno)


def test_no_empty_contents():
    for name, lineno, obj in load_pairs():
        for msg in obj["messages"]:
            assert msg["content"].strip(), (name, lineno, msg["role"])


def test_no_cjk_characters():
    for name, lineno, obj in load_pairs():
        for msg in obj["messages"]:
            assert not CJK.search(msg["content"]), (name, lineno, msg["content"][:40])


def test_latin_words_are_known_terms():
    for name, lineno, obj in load_pairs():
        for msg in obj["messages"]:
            for word in LATIN.findall(msg["content"]):
                assert word in ALLOWED_LATIN, (name, lineno, word)


def test_no_duplicate_questions():
    counts = collections.Counter(
        obj["messages"][0]["content"] for _, _, obj in load_pairs()
    )
    dupes = {q: n for q, n in counts.items() if n > 1}
    assert not dupes, f"дубли вопросов: {dupes}"


def test_answers_are_short_replies_not_essays():
    """Рукописные ответы должны быть репликами, а не статьями."""
    for name, lineno, obj in load_pairs():
        assert len(obj["messages"][1]["content"]) <= 400, (name, lineno)


@pytest.mark.parametrize("path", chat_files(), ids=lambda p: p.name)
def test_file_is_not_empty(path):
    assert len(path.read_text(encoding="utf-8").splitlines()) > 0, path.name


def test_there_are_enough_handwritten_pairs():
    """82 пары давали зубрёжку; порог держим заметно выше."""
    assert len(load_pairs()) >= 1000
