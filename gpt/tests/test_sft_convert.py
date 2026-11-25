"""Тесты на приведение схем датасетов в scripts/fetch_sft.py.

Сама загрузка с HuggingFace здесь не проверяется — из песочницы
huggingface.co недоступен. Проверяется то, что можно проверить офлайн:
правильно ли разные схемы превращаются в наш формат {"messages": [...]}.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from fetch_sft import convert_row  # noqa: E402


def roles(obj):
    return [m["role"] for m in obj["messages"]]


def test_alpaca_schema():
    row = {"instruction": "Объясни, что такое нейросеть", "input": "",
           "output": "Это математическая модель из слоёв."}
    obj = convert_row(row)
    assert roles(obj) == ["user", "assistant"]
    assert obj["messages"][0]["content"] == "Объясни, что такое нейросеть"
    assert obj["messages"][1]["content"].startswith("Это математическая")


def test_alpaca_with_context_gets_joined():
    row = {"instruction": "Переведи", "input": "Hello world", "output": "Привет, мир"}
    obj = convert_row(row)
    assert "Hello world" in obj["messages"][0]["content"]
    assert obj["messages"][0]["content"].startswith("Переведи")


def test_chatml_schema_and_role_aliases():
    row = {"conversations": [
        {"from": "human", "value": "Привет"},
        {"from": "gpt", "value": "Здравствуй!"},
    ]}
    obj = convert_row(row)
    assert roles(obj) == ["user", "assistant"]
    assert obj["messages"][1]["content"] == "Здравствуй!"


def test_messages_schema_passthrough():
    row = {"messages": [
        {"role": "system", "content": "Ты помощник"},
        {"role": "user", "content": "Сколько будет 2+2?"},
        {"role": "assistant", "content": "Четыре"},
    ]}
    obj = convert_row(row)
    assert roles(obj) == ["system", "user", "assistant"]


def test_text_summary_schema():
    row = {"text": "Новость. " * 60, "summary": "Короткое саммари новости"}
    obj = convert_row(row)
    assert roles(obj) == ["user", "assistant"]
    assert obj["messages"][0]["content"].startswith("Кратко перескажи текст.")
    assert obj["messages"][1]["content"] == "Короткое саммари новости"


def test_short_text_is_not_turned_into_task():
    """Короткий text без саммари не должен становиться заданием."""
    assert convert_row({"text": "просто короткий текст"}) is None


def test_unrecognized_schema_returns_none():
    assert convert_row({"foo": "bar", "baz": 42}) is None
    assert convert_row({}) is None


def test_chat_without_assistant_is_rejected():
    row = {"messages": [{"role": "user", "content": "Привет"}]}
    assert convert_row(row) is None


def test_whitespace_is_collapsed_and_long_text_truncated():
    row = {"instruction": "Много\n\n  пробелов\tтут", "output": "о" * 9000}
    obj = convert_row(row, limit=100)
    assert obj["messages"][0]["content"] == "Много пробелов тут"
    assert len(obj["messages"][1]["content"]) == 100


def test_sft_dataset_mixes_files_with_repeat(tmp_path):
    """Рукописных диалогов десятки, корпусных — десятки тысяч.

    Без --repeat модель не замечает первых: замер на 52 533 корпусных парах
    дал ответы новостным текстом на «Привет!».
    """
    from minigpt.data import SFTDataset
    from minigpt.tokenizer import BPETokenizer

    tok = BPETokenizer.train(["привет как дела", "меня зовут аня"], 50, verbose=False)
    a = tmp_path / "hand.jsonl"
    a.write_text('{"messages": [{"role": "user", "content": "привет"},'
                 ' {"role": "assistant", "content": "привет как дела"}]}\n', encoding="utf-8")
    b = tmp_path / "corpus.jsonl"
    b.write_text("\n".join(
        '{"messages": [{"role": "user", "content": "меня зовут аня"},'
        ' {"role": "assistant", "content": "меня зовут аня"}]}' for _ in range(5)),
        encoding="utf-8")

    one = SFTDataset([str(a), str(b)], tok, 64)
    assert len(one) == 6

    rep = SFTDataset([str(a), str(b)], tok, 64, repeat={str(a): 4})
    assert len(rep) == 4 + 5, "файл из repeat должен повториться 4 раза"
