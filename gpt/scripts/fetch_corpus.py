#!/usr/bin/env python3
"""Скачивает русские корпуса с GitHub и чистит их в data/raw/.

HuggingFace из некоторых окружений недоступен, а GitHub — доступен. Здесь
собраны источники, которые реально отдаются через codeload.github.com:

    russian_novels        108 русских романов, UTF-8 (JoannaBy/RussianNovels)
    novaya_gazeta         архив «Новой газеты», ~57k статей (novaya-gazeta-ru/archive)
    orthodox_encyclopedia «Православная энциклопедия» в Markdown (slavonic/pravenc-md)

    python scripts/fetch_corpus.py --source russian_novels
    python scripts/fetch_corpus.py --source all --out data/raw

Для больших корпусов (taiga и т.п.) качайте на своей машине с HuggingFace —
команды есть в README.md.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import os
import re
import sys
import tarfile
import urllib.request
from typing import Iterable, List

SOURCES = {
    "russian_novels": {
        "url": "https://codeload.github.com/JoannaBy/RussianNovels/tar.gz/refs/heads/master",
        "patterns": ["*/corpus/*.txt"],
    },
    "novaya_gazeta": {
        "url": "https://codeload.github.com/novaya-gazeta-ru/archive/tar.gz/refs/heads/main",
        "patterns": ["*/articles/*.md", "*/*/articles/*.md", "*/*/*/articles/*.md"],
    },
    "orthodox_encyclopedia": {
        "url": "https://codeload.github.com/slavonic/pravenc-md/tar.gz/refs/heads/main",
        "patterns": ["*/*.md", "*/*/*.md"],
    },
}

# мусор из конвертации HTML в текст (lib.ru и похожие)
_WS = re.compile(r"[ \t\u00a0]{2,}")
_BLANK = re.compile(r"\n{3,}")
_NAV = re.compile(r"^\s*(главн(ая|ое)|содержание|оглавление|в начало|назад|далее|"
                  r"предыдущ|следующ|страница \d+|Читать дальше)\b", re.I)
_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_MARK = re.compile(r"^[#>*_\-|`~\s]+", re.M)
_HTML = re.compile(r"<[^>]{1,80}>")
_URL = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    text = _FRONTMATTER.sub("", text, count=1)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u2013", "—").replace("\u2014", "—")
    text = _MD_LINK.sub(r"\1", text)     # [текст](ссылка) -> текст
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    lines = []
    for ln in text.split("\n"):
        ln = _MD_MARK.sub("", ln)
        ln = _WS.sub(" ", ln).strip()
        if not ln or _NAV.match(ln):
            lines.append("")
            continue
        lines.append(ln)
    return _BLANK.sub("\n\n", "\n".join(lines)).strip()


def iter_tar(url: str, patterns: List[str], limit: int = 0,
             local: str = "") -> Iterable[tuple]:
    if local and os.path.exists(local):
        print(f"беру локальный архив {local}", flush=True)
        with open(local, "rb") as f:
            blob = f.read()
    else:
        print(f"качаю {url} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "minigpt-fetch"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            blob = r.read()
    print(f"  получено {len(blob)/1e6:.1f} MB, распаковываю...", flush=True)
    n = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for m in tf:
            if not m.isfile() or m.size == 0 or m.size > 20 * 1024 * 1024:
                continue
            if not any(fnmatch.fnmatch(m.name, p) for p in patterns):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            raw = f.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = raw.decode("cp1251")
                except UnicodeDecodeError:
                    continue
            n += 1
            yield os.path.basename(m.name), text
            if limit and n >= limit:
                return


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="russian_novels",
                   help=f"{', '.join(SOURCES)} или all")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--limit", type=int, default=0, help="максимум файлов (0 = все)")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--local", default="", help="взять уже скачанный .tar.gz вместо загрузки")
    args = p.parse_args()

    names = list(SOURCES) if args.source == "all" else [args.source]
    if args.local and len(names) > 1:
        raise SystemExit("--local работает только с одним --source")
    for name in names:
        if name not in SOURCES:
            raise SystemExit(f"неизвестный источник {name!r}: {sorted(SOURCES)}")
        outdir = os.path.join(args.out, name)
        os.makedirs(outdir, exist_ok=True)
        total_chars = kept = 0
        with open(os.path.join(outdir, "corpus.txt"), "w", encoding="utf-8") as out:
            for fname, text in iter_tar(SOURCES[name]["url"], SOURCES[name]["patterns"],
                                        args.limit, local=args.local):
                cleaned = clean_text(text)
                if len(cleaned) < args.min_chars:
                    continue
                out.write(cleaned)
                out.write("\n\n")           # разделитель документов
                total_chars += len(cleaned)
                kept += 1
                if kept % 500 == 0:
                    print(f"  {kept} файлов, {total_chars/1e6:.1f}M символов", flush=True)
        print(f"[{name}] документов: {kept}, символов: {total_chars:,} "
              f"(~{total_chars/3/1e6:.1f}M токенов) -> {outdir}/corpus.txt")


if __name__ == "__main__":
    main()
