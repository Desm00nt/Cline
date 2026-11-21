import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minigpt.rag import TFIDFIndex, iter_texts
from minigpt.tokenizer import BPETokenizer

DOCS = [
    ("Трансформер — это архитектура нейронной сети. Её главное отличие — механизм внимания, "
     "который позволяет модели смотреть на весь фрагмент текста сразу."),
    ("Париж — столица Франции. Город стоит на реке Сене, там находится Эйфелева башня и Лувр."),
    ("Механизм внимания строит для каждого токена запрос, ключ и значение, а затем взвешивает "
     "значения по совпадению запроса с ключами."),
    ("Рецепт борща: свёкла, капуста, картофель, морковь и мясной бульон. Варить полтора часа."),
]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(list(DOCS) * 5, num_merges=300, verbose=False)


@pytest.fixture()
def index(tok):
    idx = TFIDFIndex(tok, chunk_tokens=48, overlap=8)
    for i, d in enumerate(DOCS):
        idx.add_text(d, doc=f"doc{i}")
    idx.build()
    return idx


def test_index_built(index):
    assert len(index) >= len(DOCS)
    assert len(index.norm) == len(index)
    assert all(n > 0 for n in index.norm)


def test_retrieval_finds_relevant(index):
    hits = index.search("как работает механизм внимания", k=2)
    assert hits
    joined = " ".join(h.text for h in hits)
    assert "внимани" in joined.lower()
    assert hits[0].score > 0


def test_retrieval_topic_separation(index):
    food = index.search("как варить борщ", k=1)
    paris = index.search("столица Франции", k=1)
    assert food and paris
    assert "борщ" in food[0].text.lower()
    assert "париж" in paris[0].text.lower()


def test_scores_are_normalized(index):
    for h in index.search("нейронная сеть трансформер", k=4):
        assert -1.01 <= h.score <= 1.01


def test_empty_and_unknown_query(index):
    assert index.search("", k=3) == []
    # символы, которых физически нет в корпусе, -> ни одного совпадения
    assert index.search("\U0001d11e\U0001d11e\U0001d11e", k=3) == []


def test_context_ids_within_budget(index):
    ids, hits = index.context_ids("механизм внимания", k=3, budget=200)
    assert ids and hits
    assert len(ids) <= 200
    assert len(hits) <= 3


def test_save_load_roundtrip(tmp_path, tok, index):
    path = str(tmp_path / "idx.json")
    index.save(path)
    loaded = TFIDFIndex.load(path, tok)
    assert len(loaded) == len(index)
    a = index.search("механизм внимания", k=3)
    b = loaded.search("механизм внимания", k=3)
    assert [h.text for h in a] == [h.text for h in b]
    assert [round(h.score, 6) for h in a] == [round(h.score, 6) for h in b]


def test_iter_texts(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Первый документ про одно.\n\nВторой документ про другое совсем.",
                 encoding="utf-8")
    got = list(iter_texts(str(p), min_chars=10))
    assert len(got) == 2
    assert got[0][0] == "a.txt"
