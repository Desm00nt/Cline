"""Строит SFT-данные из структуры самого корпуса — без большой модели и без HF.

Зачем: HuggingFace из песочницы недоступен, а готовых русских инструкционных
датасетов на GitHub нет (проверено поиском). Но обучающий сигнал для SFT можно
достать из структуры текста, который у нас уже есть:

* **Пересказ.** Первый абзац новости — это её лид, то есть готовое краткое
  содержание. Задача «перескажи» -> лид, остальной текст в условии.
* **Вопрос по заголовку.** В markdown-энциклопедии заголовок раздела — это
  готовый вопрос, а тело раздела — ответ.
* **Определение термина.** Строка вида «ТЕРМИН — определение» в энциклопедии.

Честно про то, что это даёт: такие данные учат модель **формату диалога,
удержанию темы и стилю ответа**. Фактов они не добавляют — факты берём из RAG.
Это ровно тот же приём, которым делают self-instruct-лайт: вопросы шаблонные,
ответы настоящие.

    python scripts/build_sft_from_corpus.py --input data/raw_real --out data/sft/built.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple

ASK_SUMMARY = [
    "Кратко перескажи этот текст.",
    "О чём этот текст? Ответь одним-двумя предложениями.",
    "Сформулируй основную мысль текста.",
    "Перескажи самое главное из текста ниже.",
    "Опиши коротко, о чём идёт речь.",
]
ASK_SECTION = [
    "Что такое {topic}?",
    "Расскажи про {topic}.",
    "Объясни, что имеется в виду: {topic}.",
    "Коротко о {topic} — что это?",
]
ASK_CONTINUE = [
    "Продолжи текст.",
    "Что было дальше? Продолжи.",
]

HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
# Порог длины определения держим в одном месте: он был продублирован в
# теле make_definition_tasks, и порог в 80 символов отбрасывал нормальные
# словарные статьи («центральное христианское таинство, ...» = 71 символ).
MIN_DEFINITION_CHARS = 60
DEFINITION = re.compile(
    r"^\*{0,2}([А-ЯЁA-Z][\w\- ]{2,40})\*{0,2}\s*[—–-]{1,2}\s+(.{%d,})$"
    % MIN_DEFINITION_CHARS, re.MULTILINE)


def split_docs(text: str) -> List[str]:
    """Документы в нашем корпусе разделены пустой строкой."""
    return [d.strip() for d in re.split(r"\n\s*\n", text) if len(d.strip()) > 200]


def split_paragraphs(doc: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n+", doc) if len(p.strip()) > 60]
    return paras


def _msgs(question: str, answer: str) -> Dict:
    return {"messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]}


def make_summary_tasks(doc: str, rng: random.Random, max_per_doc: int = 1
                       ) -> Iterable[Dict]:
    """Лид-абзац как краткое содержание, остальное — как исходный текст."""
    paras = split_paragraphs(doc)
    if len(paras) < 3:
        return
    lead, rest = paras[0], paras[1:]
    if not (80 <= len(lead) <= 400):
        return
    body = "\n\n".join(rest)[:2500]
    if len(body) < 300:
        return
    for _ in range(max_per_doc):
        yield _msgs(f"{rng.choice(ASK_SUMMARY)}\n\n{body}", lead)


def make_continuation_tasks(doc: str, rng: random.Random) -> Iterable[Dict]:
    """«Продолжи текст» — учит держать стиль и связность на длинном ответе."""
    paras = split_paragraphs(doc)
    if len(paras) < 4:
        return
    cut = len(paras) // 2
    head = "\n\n".join(paras[:cut])[:1200]
    tail = "\n\n".join(paras[cut:])[:800]
    if len(head) < 200 or len(tail) < 150:
        return
    yield _msgs(f"{head}\n\n{rng.choice(ASK_CONTINUE)}", tail)


def make_heading_tasks(doc: str, rng: random.Random) -> Iterable[Dict]:
    """Markdown: заголовок раздела -> тело раздела. Работает на энциклопедии."""
    lines = doc.split("\n")
    topic: Optional[str] = None
    buf: List[str] = []
    for line in lines:
        m = HEADING.match(line)
        if m:
            if topic and buf:
                yield from _emit_section(topic, "\n".join(buf).strip(), rng)
            topic, buf = re.sub(r"[*_`]", "", m.group(2)).strip(), []
        elif topic:
            buf.append(line)
    if topic and buf:
        yield from _emit_section(topic, "\n".join(buf).strip(), rng)


def _emit_section(topic: str, body: str, rng: random.Random) -> Iterable[Dict]:
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if not (60 <= len(body) <= 1200) or len(topic) < 4:
        return
    tpl = rng.choice(ASK_SECTION).format(topic=topic.lower())
    yield _msgs(tpl, body[:1200])


# Регулярное выражение DEFINITION ловит любое слово с заглавной буквы, поэтому в
# «термины» попадают слова-связки из художественного текста: «Дальше — ...»,
# «И вот — ...», «Но это — ...». Замер на 27 602 парах: топ-25 терминов на
#половину состоит из таких слов. Отбрасываем их явно.
NON_TERMS = {
    "дальше", "а дальше", "потом", "а потом", "теперь", "а теперь", "и вот",
    "но это", "и это", "а еще", "а ещё", "и главное", "но главное", "итак",
    "однако", "например", "кроме того", "тем не менее", "в том числе",
    "более того", "прежде всего", "в частности", "таким образом",
    "в результате", "при этом", "в то же время", "все это", "всё это",
    "это -", "первый", "первая", "первое", "второй", "вторая", "второе",
    "третье", "третья", "другое дело", "сегодня", "завтра", "вчера",
    "автор", "в этом", "итак,", "итак:",
    "другой пример", "с другой стороны", "с одной стороны", "то же самое",
    "иное дело", "отдельная тема", "среди них", "для начала", "вот -",
    "перед вами", "причина", "почему", "сначала", "здесь", "рассказ",
    "кроме того,", "в общем", "в конце концов", "так или иначе",
}


# Обороты, которые выдают связку, а не термин: «с другой стороны — ...»,
# «для начала — ...». Держим отдельно от NON_TERMS: их видно по началу фразы.
LINK_PREFIXES = (
    "это", "но ", "а ", "и ", "не ", "с ", "для ", "в ", "во ", "к ", "на ",
    "из ", "от ", "до ", "по ", "при ", "без ", "перед ", "после ", "между ",
    "другой", "другое", "иное", "отдельн", "то же", "так ", "тут ", "вот ",
    "здесь", "там ", "потом", "далее", "затем", "сначала", "почему", "потому",
    "если ", "когда ", "хотя ", "чтобы", "каждый", "всякий", "любой", "один",
    "перв", "втор", "треть", "четверт", "последн", "следующ", "например",
)


def is_real_term(term: str) -> bool:
    """Термин годится, если это не слово-связка и не служебная фраза."""
    t = term.lower().strip(" .,;:-")
    if not t or t in NON_TERMS:
        return False
    words = t.split()
    if words[0] in NON_TERMS:
        return False
    # проверяем и первое слово целиком: «Вот — ...» после strip даёт «вот»,
    # а префикс в списке записан с пробелом («вот ») и не срабатывает
    if t.startswith(LINK_PREFIXES) or words[0] in {p.strip() for p in LINK_PREFIXES}:
        return False
    # термин из трёх и более слов почти всегда оборот, а не понятие
    return len(words) <= 3


def make_definition_tasks(doc: str, rng: random.Random) -> Iterable[Dict]:
    """«ТЕРМИН — определение»: самый чистый вид QA-пары."""
    for line in doc.split("\n"):
        m = DEFINITION.match(line.strip())
        if not m:
            continue
        term, definition = m.group(1).strip(), m.group(2).strip()
        if not (4 <= len(term) <= 40) or len(definition) < MIN_DEFINITION_CHARS:
            continue
        if not is_real_term(term):
            continue
        yield _msgs(rng.choice(ASK_SECTION).format(topic=term.lower()),
                    definition[:900])


EXTRACTORS = {
    "summary": make_summary_tasks,
    "continuation": make_continuation_tasks,
    "heading": make_heading_tasks,
    "definition": make_definition_tasks,
}


def iter_files(root: str) -> Iterable[str]:
    if os.path.isfile(root):
        yield root
        return
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if n.endswith((".txt", ".md")):
                yield os.path.join(dirpath, n)


READ_CHUNK = 1 << 20   # 1M символов на шаг чтения corpus.txt


def iter_doc_chunks(path: str, per_file: bool) -> Iterable[str]:
    """Документы файла без загрузки файла целиком.

    corpus.txt у novaya_gazeta — 1023M символов: fh.read() плюс re.split дают
    ~4 ГБ и падают в MemoryError на машине с 4 ГБ RAM. Читаем кусками, хвост
    переносим в следующий шаг (разделитель может попасть на границу куска).
    """
    # per_file=True значит «файл = один документ» (режим docs/), но для
    # corpus.txt на гигабайт это MemoryError: такой файл режем по документам.
    if per_file and os.path.getsize(path) <= 8 * READ_CHUNK:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            yield fh.read().strip()
        return
    buf = ""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        while True:
            chunk = fh.read(READ_CHUNK)
            if not chunk:
                break
            buf += chunk
            parts = re.split(r"\n\s*\n", buf)
            buf = parts.pop()          # последний кусок может быть неполным
            for part in parts:
                part = part.strip()
                if len(part) > 200:
                    yield part
    part = buf.strip()
    if len(part) > 200:
        yield part


def build(root: str, out_path: str, seed: int = 1337, per_doc: int = 1,
          kinds: Tuple[str, ...] = ("summary", "heading", "definition",
                                    "continuation"), max_rows: int = 0,
          per_file: bool = True) -> Dict[str, int]:
    """per_file=True: каждый файл считается одним документом.

    Это режим для data/raw_real/*/docs/, который пишет fetch_corpus.py. Для
    одного большого corpus.txt ставьте per_file=False — тогда документы
    разделяются пустой строкой, но внутри документа пустые строки тоже
    бывают, и границы получаются приблизительными.
    """
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    stats = {k: 0 for k in kinds}
    stats["docs"] = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for path in iter_files(root):
            for doc in iter_doc_chunks(path, per_file):
                stats["docs"] += 1
                for kind in kinds:
                    fn = EXTRACTORS[kind]
                    gen = fn(doc, rng, per_doc) if kind in ("summary",) else fn(doc, rng)
                    for obj in gen:
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        stats[kind] += 1
                        if max_rows and sum(v for k, v in stats.items() if k != "docs") >= max_rows:
                            return stats
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="data/sft/built.jsonl")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--per-doc", type=int, default=1)
    p.add_argument("--kinds", default="summary,heading,definition,continuation")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--whole-file-as-doc", action="store_true", default=True,
                   help="каждый файл = один документ (режим docs/ из fetch_corpus)")
    p.add_argument("--split-docs", dest="whole_file_as_doc", action="store_false",
                   help="для одного большого corpus.txt: документы по пустой строке")
    args = p.parse_args()

    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip() in EXTRACTORS)
    stats = build(args.input, args.out, args.seed, args.per_doc, kinds, args.max_rows,
                  per_file=args.whole_file_as_doc)
    total = sum(v for k, v in stats.items() if k != "docs")
    print(f"документов просмотрено: {stats['docs']}")
    for k in kinds:
        print(f"  {k:13s} {stats[k]:7d} пар")
    print(f"всего: {total} -> {args.out}")


if __name__ == "__main__":
    main()
