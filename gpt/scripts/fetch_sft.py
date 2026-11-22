"""Скачивает инструкционные датасеты с HuggingFace и приводит их к нашему jsonl.

Зачем отдельный скрипт: схемы у датасетов разные (alpaca, chatml, пары
«текст—саммари»), а sft.py ждёт один формат:

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

Скрипт сам определяет схему по именам колонок и пишет готовый jsonl.

ВАЖНО: из песочницы, где разрабатывался проект, huggingface.co недоступен
(HTTP 000), поэтому загрузка здесь не проверялась — проверялась только логика
приведения схем (tests/test_sft_convert.py). Запускайте на своей машине:

    pip install datasets
    python scripts/fetch_sft.py --list                 # что вообще отдаст данные
    python scripts/fetch_sft.py --datasets IlyaGusev/gazeta --max-rows 20000
    python scripts/fetch_sft.py --datasets IlyaGusev/gazeta --split train \
        --out data/sft/gazeta.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Кандидаты. Имена не проверялись загрузкой (HF из песочницы недоступен),
# поэтому --list существует: он честно скажет, что открылось, а что нет.
CANDIDATES = [
    "IlyaGusev/gazeta",                 # 63k новостей с саммари -> «перескажи»
    "IlyaGusev/ru_turbo_alpaca",        # русские инструкционные пары
    "IlyaGusev/instruction_dataset_ru",
    "OpenAssistant/oasst2",             # мультиязычный, есть русская ветка
    "kirill-garanzha/russian-instruction-dataset",
]

# Имена колонок, по которым узнаём схему
ALPACA_IN = ("instruction", "question", "prompt", "input_text", "query")
ALPACA_CTX = ("input", "context", "additional_info")
ALPACA_OUT = ("output", "response", "answer", "completion", "summary", "text_output")
MSG_KEYS = ("messages", "conversations", "dialog", "chat")


def _first(row: Dict[str, Any], names) -> Optional[str]:
    for n in names:
        v = row.get(n)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _clean(s: str, limit: int) -> str:
    s = " ".join(s.split())
    return s[:limit]


def convert_row(row: Dict[str, Any], limit: int = 4000) -> Optional[Dict[str, Any]]:
    """Одна строка датасета -> {"messages": [...]} или None, если не распознали."""
    # 1) уже готовый чат
    for key in MSG_KEYS:
        msgs = row.get(key)
        if isinstance(msgs, (list, tuple)) and msgs:
            out: List[Dict[str, str]] = []
            for m in msgs:
                if not isinstance(m, dict):
                    return None
                role = str(m.get("role") or m.get("from") or "").lower()
                role = {"human": "user", "gpt": "assistant",
                        "bot": "assistant", "system": "system"}.get(role, role)
                content = m.get("content") or m.get("text") or m.get("value")
                if role in ("user", "assistant", "system") and isinstance(content, str):
                    out.append({"role": role, "content": _clean(content, limit)})
            if any(m["role"] == "assistant" for m in out) and \
                    any(m["role"] == "user" for m in out):
                return {"messages": out}
            return None

    # 2) alpaca-подобная схема
    instr = _first(row, ALPACA_IN)
    out_txt = _first(row, ALPACA_OUT)
    if instr and out_txt:
        ctx = _first(row, ALPACA_CTX)
        user = f"{instr}\n\n{ctx}" if ctx else instr
        return {"messages": [{"role": "user", "content": _clean(user, limit)},
                             {"role": "assistant", "content": _clean(out_txt, limit)}]}

    # 3) «текст + саммари» (gazeta и похожие)
    body = row.get("text") if isinstance(row.get("text"), str) else None
    summ = row.get("summary") or row.get("title")
    if body and isinstance(summ, str) and len(body) > 200:
        return {"messages": [
            {"role": "user", "content": _clean("Кратко перескажи текст.\n\n" + body, limit)},
            {"role": "assistant", "content": _clean(summ, limit)}]}
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="", help="через запятую; пусто = список кандидатов")
    p.add_argument("--list", action="store_true", help="только проверить доступность")
    p.add_argument("--split", default="train")
    p.add_argument("--max-rows", type=int, default=0, help="0 = весь сплит")
    p.add_argument("--min-chars", type=int, default=20,
                   help="отбрасывать ответы короче этого")
    p.add_argument("--out", default="", help="файл jsonl; по умолчанию data/sft/<имя>.jsonl")
    p.add_argument("--lang-filter", action="store_true",
                   help="оставить только строки с кириллицей в ответе")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("нужен пакет datasets: pip install datasets")

    names = [s.strip() for s in args.datasets.split(",") if s.strip()] or CANDIDATES
    for name in names:
        try:
            ds = load_dataset(name, split=args.split, trust_remote_code=True)
        except Exception as e:
            print(f"[{name}] НЕ ЗАГРУЗИЛСЯ: {type(e).__name__}: {str(e)[:160]}")
            continue
        n = len(ds) if hasattr(ds, "__len__") else -1
        print(f"[{name}] сплит {args.split}: {n} строк, колонки {ds.column_names}")
        if args.list:
            continue

        out_path = args.out or os.path.join("data", "sft",
                                            name.replace("/", "__") + ".jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        kept = skipped = 0
        limit = args.max_rows or n
        with open(out_path, "w", encoding="utf-8") as f:
            for i, row in enumerate(ds):
                if i >= limit:
                    break
                obj = convert_row(dict(row))
                if obj is None:
                    skipped += 1
                    continue
                ans = [m["content"] for m in obj["messages"] if m["role"] == "assistant"]
                if not ans or min(len(a) for a in ans) < args.min_chars:
                    skipped += 1
                    continue
                if args.lang_filter and not any("\u0400" <= c <= "\u04ff" for c in ans[0]):
                    skipped += 1
                    continue
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
        print(f"  -> {out_path}: записано {kept}, отброшено {skipped}")


if __name__ == "__main__":
    main()
