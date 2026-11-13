"""Сборка документации: Markdown -> HTML -> PDF (Chrome или Edge в режиме без окна).

    python docs/build_pdf.py            собрать все документы в docs/pdf/
    python docs/build_pdf.py 04_guide   собрать один

Оформление — по ГОСТ 7.32-2017 (п. 6.1): формат A4, поля 30/15/20/20 мм, шрифт с засечками
12 пт, полуторный интервал, абзацный отступ 1,25 см, номер страницы внизу по центру,
титульный лист без номера, «Содержание» с номерами страниц. Подписи: «Таблица N — …» над
таблицей слева, «Рисунок N — …» под рисунком по центру.

Разметка поверх обычного Markdown:
    шапка файла  ---  title / designation / kind / basis / org / year  ---
    T[метка]: Название таблицы        строка перед таблицей
    F[метка]: Подпись рисунка         строка после рисунка
    {T:метка} {F:метка}               номер таблицы или рисунка в тексте
    <div class="pagebreak"></div>     разрыв страницы

Номера страниц в «Содержании» получаются в два прохода: первый PDF читается через pypdf,
по тексту страниц находятся заголовки, второй проход печатает уже с номерами.
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

DOCS = Path(__file__).parent
OUT = DOCS / "pdf"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

CSS = """
@page { size: A4; margin: 20mm 15mm 20mm 30mm;
        @bottom-center { content: counter(page); font: 12pt "Times New Roman", serif; } }
@page :first { @bottom-center { content: ""; } }
* { box-sizing: border-box; }
body { font: 12pt/1.5 "Times New Roman", "Liberation Serif", serif; color: #000; margin: 0;
       text-align: justify; hyphens: auto; }
p { margin: 0; text-indent: 1.25cm; }
h1, h2, h3 { font-size: 12pt; font-weight: bold; text-align: left; margin: 1.1em 0 .5em; text-indent: 1.25cm;
             break-after: avoid; hyphens: none; }
h1 { font-size: 13pt; margin-top: 1.4em; }
h1.struct { text-align: center; text-indent: 0; text-transform: uppercase; break-before: page; }
ul, ol { margin: 0; padding-left: 1.25cm; list-style-position: inside; }
li { text-indent: 0; padding-left: 0; }
li p { text-indent: 0; display: inline; }
table { border-collapse: collapse; width: 100%; margin: 0 0 .9em; font-size: 10.5pt; line-height: 1.25;
        text-align: left; hyphens: auto; }
th, td { border: 0.5pt solid #000; padding: 2pt 4pt; vertical-align: top; }
th { text-align: center; font-weight: bold; hyphens: none; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
p.tcap { text-indent: 0; text-align: left; margin: .7em 0 .25em; break-after: avoid; }
p.fcap { text-indent: 0; text-align: center; margin: .3em 0 .9em; }
figure { margin: .7em 0 0; text-align: center; break-inside: avoid; }
figure img, figure svg { max-width: 100%; height: auto; }
pre { font: 9.5pt/1.3 Consolas, "DejaVu Sans Mono", monospace; border: 0.5pt solid #777; padding: 5pt 7pt;
      margin: .5em 0 .8em; white-space: pre-wrap; overflow-wrap: anywhere; text-align: left; break-inside: avoid; }
code { font: 10pt Consolas, "DejaVu Sans Mono", monospace; hyphens: none; }
pre code { font-size: 9.5pt; }
blockquote { margin: .5em 0 .5em 1.25cm; font-size: 11pt; }
blockquote p { text-indent: 0; }
a { color: inherit; text-decoration: none; overflow-wrap: anywhere; }
.pagebreak { break-before: page; }
.title { height: 245mm; display: flex; flex-direction: column; align-items: center; text-align: center;
         break-after: page; }
.title p { text-indent: 0; text-align: center; }
.title .org { font-size: 13pt; }
.title .name { margin-top: 62mm; font-size: 16pt; font-weight: bold; text-transform: uppercase; }
.title .kind { margin-top: 8mm; font-size: 14pt; font-weight: bold; }
.title .des { margin-top: 6mm; font-size: 13pt; }
.title .basis { margin-top: 10mm; font-size: 11pt; max-width: 130mm; }
.title .year { margin-top: auto; }
.toc { break-after: page; }
.toc p { text-indent: 0; display: flex; align-items: baseline; text-align: left; }
.toc p.l2 { padding-left: 1.25cm; }
.toc .dots { flex: 1; border-bottom: 1px dotted #000; margin: 0 4pt; transform: translateY(-3pt); }
.small { font-size: 10.5pt; }
"""


def find_browser() -> str:
    for b in BROWSERS:
        if Path(b).exists():
            return b
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome", "msedge"):
        if shutil.which(name):
            return shutil.which(name)
    raise SystemExit("Не найден Chrome/Chromium/Edge — он нужен для печати PDF")


def parse_meta(text: str) -> tuple[dict, str]:
    meta: dict = {}
    if text.startswith("---"):
        head, _, text = text[3:].partition("\n---")
        for line in head.strip().splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, text.lstrip("\n")


def number_captions(text: str) -> str:
    """Пронумеровать таблицы и рисунки и подставить номера в ссылки {T:метка}, {F:метка}."""
    labels: dict[str, int] = {}
    counters = {"T": 0, "F": 0}

    def cap(m):
        kind, label, title = m.group(1), m.group(2), m.group(3).strip()
        counters[kind] += 1
        n = counters[kind]
        if label:
            labels[f"{kind}:{label}"] = n
        word, cls = ("Таблица", "tcap") if kind == "T" else ("Рисунок", "fcap")
        return f'\n<p class="{cls}">{word} {n} — {title}</p>\n'

    text = re.sub(r"^([TF])(?:\[([\w-]+)\])?:\s*(.+)$", cap, text, flags=re.M)

    def ref(m):
        key = f"{m.group(1)}:{m.group(2)}"
        if key not in labels:
            raise SystemExit(f"нет метки {key}")
        return str(labels[key])

    return re.sub(r"\{([TF]):([\w-]+)\}", ref, text)


# невидимая отметка начала основного текста: по ней находится первая страница после «Содержания»
BODY_MARK = '<span style="font-size:1pt;color:#fff">QZBODYSTARTQZ</span>'

STRUCT = ("ВВЕДЕНИЕ", "ЗАКЛЮЧЕНИЕ", "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ", "РЕФЕРАТ", "ПРИЛОЖЕНИЕ", "АННОТАЦИЯ",
          "ТЕРМИНЫ", "ПЕРЕЧЕНЬ СОКРАЩЕНИЙ")


def render(md_path: Path, pages: dict[str, int] | None = None) -> tuple[str, list[tuple[int, str]]]:
    meta, body = parse_meta(md_path.read_text(encoding="utf-8"))
    body = number_captions(body)
    content = markdown.markdown(body, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    heads: list[tuple[int, str]] = []

    def head(m):
        level, title = int(m.group(1)), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if level <= 2:
            heads.append((level, html.unescape(title)))
        cls = ' class="struct"' if level == 1 and title.upper().startswith(STRUCT) else ""
        return f"<h{level}{cls}>{m.group(2)}</h{level}>"

    content = re.sub(r"<h([1-3])>(.*?)</h\1>", head, content, flags=re.S)
    content = re.sub(r"<p>(<img [^>]+>)</p>", r"<figure>\1</figure>", content)
    # неразрывные пробелы в числах («21 500») и перед единицами: число не должно рваться по строкам
    parts = re.split(r"(<pre>.*?</pre>|<svg.*?</svg>|<[^>]+>)", content, flags=re.S)
    for i in range(0, len(parts), 2):
        t = re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "\u00a0", parts[i])
        t = re.sub(r"(\d)–(\d)", "\\1\u2060–\u2060\\2", t)
        parts[i] = re.sub(r"(?<=\d) (?=(?:руб|тыс|млн|млрд|мм|см|м|кг|Вт|В|А|Гц|мс|с|ч|шт|ГБ|МБ|Мп)\b|[%°])", "\u00a0", t)
    content = "".join(parts)
    toc = ['<div class="toc"><h1 class="struct" style="break-before:auto">Содержание</h1>']
    abstract_first = bool(heads) and heads[0][1].upper() == "РЕФЕРАТ"
    if abstract_first:
        heads = heads[1:]          # реферат стоит перед содержанием и в него не входит
    for level, title in heads:
        page = "" if pages is None else str(pages.get(title, ""))
        toc.append(f'<p class="l{level}"><span>{html.escape(title)}</span><span class="dots"></span>'
                   f"<span>{page}</span></p>")
    toc.append("</div>")
    toc_html = "".join(toc)
    # ГОСТ 7.32: реферат стоит перед содержанием
    if abstract_first:
        cut = content.find("<h1", content.find("<h1") + 3)
        content = ('<div style="break-after:page">' + content[:cut].replace('class="struct"', 'class="struct" style="break-before:auto"', 1)
                   + "</div>" + toc_html + re.sub(r"(<h1[^>]*>)", r"\1" + BODY_MARK, content[cut:], count=1))
        toc_html = ""
    else:
        content = re.sub(r"(<h1[^>]*>)", r"\1" + BODY_MARK, content, count=1)
    title = f"""<div class="title">
<p class="org">{html.escape(meta.get("org", "Проектная команда RackScan"))}</p>
<p class="name">{html.escape(meta.get("title", md_path.stem))}</p>
<p class="kind">{html.escape(meta.get("kind", ""))}</p>
<p class="des">{html.escape(meta.get("designation", ""))}</p>
<p class="basis">{html.escape(meta.get("basis", ""))}</p>
<p class="year">{html.escape(meta.get("year", "2026"))}</p></div>"""
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>{html.escape(meta.get("title", ""))}. {html.escape(meta.get("kind", ""))}</title>
<style>{CSS}</style></head><body>{title}{toc_html}{content}</body></html>"""
    return page, heads


def to_pdf(browser: str, html_path: Path, pdf_path: Path) -> None:
    with tempfile.TemporaryDirectory() as profile:
        cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
               "--allow-file-access-from-files", f"--print-to-pdf={pdf_path}", html_path.as_uri()]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not pdf_path.exists():
        raise SystemExit(f"PDF не создан: {r.stderr[-400:]}")


def heading_pages(pdf_path: Path, heads: list[tuple[int, str]]) -> dict[str, int]:
    """На какой странице PDF впервые встречается каждый заголовок (после «Содержания»)."""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    texts = [re.sub(r"\s+", " ", p.extract_text() or "").replace("\u00ad", "") for p in reader.pages]
    first = next((i for i, t in enumerate(texts) if "QZBODYSTARTQZ" in t), 2)
    out, start = {}, first
    for _, title in heads:
        key = re.sub(r"\s+", " ", title)
        for i in range(start, len(texts)):
            if key[:60].upper() in texts[i].upper():
                out[title] = i + 1
                start = i
                break
    return out


def build(md_path: Path, browser: str) -> Path:
    OUT.mkdir(exist_ok=True)
    work = DOCS / "build"
    work.mkdir(exist_ok=True)
    html_path = work / (md_path.stem + ".html")
    pdf_path = OUT / (md_path.stem + ".pdf")
    for link in ("figures",):       # относительные ссылки на рисунки должны работать из build/
        dst = work / link
        if not dst.exists() and (DOCS / link).exists():
            shutil.copytree(DOCS / link, dst)
        elif dst.exists():
            shutil.copytree(DOCS / link, dst, dirs_exist_ok=True)
    page, heads = render(md_path)
    html_path.write_text(page, encoding="utf-8")
    tmp = work / (md_path.stem + "_pass1.pdf")
    to_pdf(browser, html_path, tmp)
    pages = heading_pages(tmp, heads)
    page, _ = render(md_path, pages)
    html_path.write_text(page, encoding="utf-8")
    to_pdf(browser, html_path, pdf_path)
    from pypdf import PdfReader
    n = len(PdfReader(str(pdf_path)).pages)
    missing = [t for _, t in heads if t not in pages]
    print(f"{pdf_path.name}: {n} с." + (f"  (нет номера страницы: {missing})" if missing else ""))
    return pdf_path


def main(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    browser = find_browser()
    files = sorted(DOCS.glob("[0-9][0-9]_*.md"))
    if argv:
        files = [f for f in files if any(f.stem.startswith(a) or f.stem == a for a in argv)]
    if not files:
        raise SystemExit("нет документов для сборки")
    for f in files:
        build(f, browser)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
