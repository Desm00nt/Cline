import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minigpt.tokenizer import (BASE_OFFSET, BOS_ID, EOS_ID, MERGE_OFFSET,
                               BPETokenizer, pre_tokenize)

CORPUS = [
    "Привет! Как дела? Я учу языковые модели.",
    "Машинное обучение — это способ научить компьютер находить закономерности в данных.",
    "Нейросеть состоит из слоёв: каждый слой преобразует вход и передаёт его дальше.",
    "The quick brown fox jumps over the lazy dog. Трансформеры изменили всё.",
    "Раз, два, три, четыре, пять — вышел зайчик погулять.",
    "Внимание (attention) позволяет модели смотреть на разные части текста одновременно.",
    "Он сказал: «Поехали!» — и ракета поднялась в небо.",
    "Градиентный спуск, AdamW, Muon, backprop, dropout, RoPE, SwiGLU.",
]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS * 3, num_merges=200, verbose=False)


def test_pre_tokenize_covers_text():
    text = "Привет, мир! Это тест-2024."
    assert "".join(pre_tokenize(text)) == text


@pytest.mark.parametrize("text", [
    "Привет, мир!",
    "Трансформер — это архитектура нейросети.",
    "Mixed text with English, цифрами 42 и знаками %$#@!",
    "Эмодзи: 🚀✨ и ё-ё-ё",
    "Длинное-длинное слово с дефисами и   пробелами\nи переносами",
])
def test_roundtrip(tok, text):
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_merges_compress(tok):
    text = " ".join(CORPUS)
    ids = tok.encode(text)
    raw = len(text.encode("utf-8"))
    assert len(ids) < raw, "BPE должен сжимать текст"
    assert len(ids) < raw * 0.6


def test_special_tokens_pass_through(tok):
    text = "<|user|>привет<|end|><|assistant|>здравствуй<|end|>"
    ids = tok.encode(text)
    assert ids[0] == 3 and EOS_ID not in ids[:1]
    assert tok.decode(ids, skip_specials=True) == "приветздравствуй"
    assert "<|user|>" in tok.decode(ids, skip_specials=False)


def test_encode_documents_adds_eos(tok):
    ids = tok.encode_documents(["а", "б"])
    assert ids.count(EOS_ID) == 2
    assert ids[-1] == EOS_ID


def test_vocab_size_consistent(tok):
    assert tok.vocab_size == MERGE_OFFSET + len(tok.merges)
    ids = tok.encode(" ".join(CORPUS))
    assert max(ids) < tok.vocab_size
    assert min(ids) >= BASE_OFFSET


def test_save_load(tmp_path, tok):
    path = str(tmp_path / "tok.json")
    tok.save(path)
    loaded = BPETokenizer.load(path)
    text = "Проверка сохранения токенайзера, 123."
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.vocab_size == tok.vocab_size


def test_unknown_bytes_survive():
    """Токенайзер, обученный без редких символов, всё равно их кодирует."""
    tok = BPETokenizer.train(["обычный текст"], num_merges=10, verbose=False)
    text = "иероглифы: 漢字, 日本語"
    assert tok.decode(tok.encode(text)) == text


def test_empty_encode(tok):
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_bos_option(tok):
    ids = tok.encode("текст", add_specials=True)
    assert ids[0] == BOS_ID


def test_encode_matches_per_chunk(tok):
    """encode обязан идти по чанкам пред-токенизации — как при обучении."""
    text = "Машинное обучение — это способ, 42 теста.\nНовая строка и   пробелы."
    expected = []
    for chunk in pre_tokenize(text):
        expected.extend(tok._encode_chunk(chunk))
    assert tok.encode(text) == expected


def test_encode_is_linear_not_quadratic(tok):
    """Регрессия: без пред-токенизации 50 КБ кодируются минутами (O(n^2))."""
    import time
    text = ("Трансформер использует механизм внимания, чтобы связывать слова. " * 900)
    assert len(text) > 50_000
    tok._encode_cache.clear()
    t = time.time()
    ids = tok.encode(text)
    dt = time.time() - t
    assert ids, "ничего не закодировалось"
    assert dt < 5.0, f"50 КБ закодировались за {dt:.1f} c — похоже на квадратичную сложность"
    assert len(ids) / dt > 10_000, f"слишком медленно: {len(ids)/dt:,.0f} ток/с"
