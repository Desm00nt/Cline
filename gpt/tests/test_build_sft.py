"""Тесты на построение SFT-пар из структуры текста (scripts/build_sft_from_corpus.py).

Данные для тестов синтетические: настоящие корпуса большие и в репозиторий
не кладутся.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from build_sft_from_corpus import (  # noqa: E402
    DEFINITION,
    build,
    make_continuation_tasks,
    make_definition_tasks,
    make_heading_tasks,
    make_summary_tasks,
    split_docs,
)

RNG = random.Random(0)


def qa(obj):
    return obj["messages"][0]["content"], obj["messages"][1]["content"]


def test_summary_task_uses_lead_as_answer():
    paras = ["Лид новости про важное событие в городе." * 4]
    paras += [f"Подробность номер {i}. " * 20 for i in range(4)]
    doc = "\n\n".join(paras)
    out = list(make_summary_tasks(doc, RNG))
    assert len(out) == 1
    question, answer = qa(out[0])
    assert answer.startswith("Лид новости")
    assert "Подробность номер 1" in question      # остальной текст — в условии
    assert answer not in question                  # ответ не должен быть в вопросе


def test_summary_skips_short_documents():
    assert list(make_summary_tasks("Слишком короткий текст.", RNG)) == []


def test_heading_task_maps_heading_to_section():
    doc = ("# Заголовок\n\nВступление.\n\n"
           "## Богослужение\n\n" + "Описание порядка богослужения. " * 12 + "\n\n"
           "## Другой раздел\n\nКоротко.\n")
    out = list(make_heading_tasks(doc, RNG))
    assert out, "из markdown с разделами должны получиться пары"
    q, a = qa(out[0])
    assert "богослужение" in q.lower()
    assert "Описание порядка богослужения" in a


def test_definition_task():
    line = "**ЕВХАРИСТИЯ** — центральное христианское таинство, состоящее в освящении хлеба и вина."
    assert DEFINITION.match(line.strip())
    out = list(make_definition_tasks(line + "\n", RNG))
    assert len(out) == 1
    q, a = qa(out[0])
    assert "евхаристия" in q.lower()
    assert a.startswith("центральное христианское таинство")


def test_definition_rejects_too_short():
    assert list(make_definition_tasks("ТЕРМИН — коротко.\n", RNG)) == []


def test_continuation_task_splits_document():
    doc = "\n\n".join([f"Абзац номер {i}, достаточно длинный. " * 8 for i in range(6)])
    out = list(make_continuation_tasks(doc, RNG))
    assert len(out) == 1
    q, a = qa(out[0])
    assert "Абзац номер 0" in q
    assert a.startswith("Абзац номер 3")


def test_split_docs_by_blank_line():
    text = "Первый документ. " * 40 + "\n\n" + "Второй документ. " * 40
    assert len(split_docs(text)) == 2


def test_split_docs_drops_tiny_chunks():
    got = split_docs("коротыш\n\n" + "длинный текст " * 40)
    assert len(got) == 1
    assert got[0].startswith("длинный текст")


def _write_docs(root, paras, n=2):
    """Раскладка как в data/raw_real/*/docs/: один файл = один документ."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / f"{i:06d}.txt").write_text("\n\n".join(paras), encoding="utf-8")


def test_build_writes_valid_jsonl(tmp_path):
    paras = ["Лид про важное событие." * 5] + [f"Деталь {i}. " * 25 for i in range(4)]
    docs = tmp_path / "docs"
    _write_docs(docs, paras, n=2)
    out = tmp_path / "sft.jsonl"
    stats = build(str(docs), str(out), kinds=("summary",))
    assert stats["summary"] >= 1
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == stats["summary"]
    obj = json.loads(lines[0])
    assert obj["messages"][0]["role"] == "user"
    assert obj["messages"][1]["role"] == "assistant"


def test_build_respects_max_rows(tmp_path):
    paras = ["Лид про важное событие." * 5] + [f"Деталь {i}. " * 25 for i in range(4)]
    docs = tmp_path / "docs"
    _write_docs(docs, paras, n=5)
    out = tmp_path / "sft.jsonl"
    stats = build(str(docs), str(out), kinds=("summary",), max_rows=2)
    assert stats["summary"] == 2


def test_per_file_mode_treats_whole_file_as_one_document(tmp_path):
    """Внутри документа пустые строки — это абзацы, а не граница документа."""
    paras = ["Лид про важное событие." * 5] + [f"Деталь {i}. " * 25 for i in range(4)]
    docs = tmp_path / "docs"
    _write_docs(docs, paras, n=1)
    out = tmp_path / "sft.jsonl"
    stats = build(str(docs), str(out), kinds=("summary",), per_file=True)
    assert stats["docs"] == 1
    assert stats["summary"] == 1


def test_answer_is_never_empty(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "000000.txt").write_text("# Раздел\n\n" + "Текст раздела. " * 30,
                                      encoding="utf-8")
    out = tmp_path / "sft.jsonl"
    build(str(docs), str(out), kinds=("heading", "definition", "continuation"))
    for line in out.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        obj = json.loads(line)
        assert obj["messages"][1]["content"].strip()
