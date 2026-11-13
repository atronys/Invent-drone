"""Сверка результата сканирования с учётными данными и сличительная ведомость.

Термины — как в бухгалтерской инвентаризации (ФСБУ 28/2023): излишек, недостача,
пересортица. Ячейка, по которой стенд не смог дать ответ (нет этикетки, код не читается,
занятость не определена), не считается ни совпадением, ни расхождением — она уходит
на ручную проверку с фотографией.
"""
from __future__ import annotations

import csv
import html
import io
from enum import Enum


class ScanStatus(str, Enum):
    READ = "READ"              # код прочитан
    EMPTY = "EMPTY"            # ячейка пуста
    NO_LABEL = "NO_LABEL"      # коробка есть, этикетки с кодом нет
    READ_ERROR = "READ_ERROR"  # этикетка есть, код не читается
    UNCERTAIN = "UNCERTAIN"    # занятость определить не удалось
    SKIPPED = "SKIPPED"        # пропущено оператором или по тайм-ауту


class Verdict(str, Enum):
    MATCH = "MATCH"            # совпало (в том числе «пусто = пусто»)
    SHORTAGE = "SHORTAGE"      # недостача: по учёту есть, фактически пусто
    SURPLUS = "SURPLUS"        # излишек: по учёту пусто, фактически стоит
    MISMATCH = "MISMATCH"      # пересортица: стоит другое
    PARTIAL = "PARTIAL"        # совпало частично
    MANUAL = "MANUAL"          # нужна ручная проверка
    NOT_SCANNED = "NOT_SCANNED"


SCAN_RU = {
    "READ": "Код прочитан", "EMPTY": "Пусто", "NO_LABEL": "Нет этикетки", "READ_ERROR": "Ошибка чтения",
    "UNCERTAIN": "Занятость не определена", "SKIPPED": "Пропущено", "": "Не сканировалась",
}
VERDICT_RU = {
    "MATCH": "Совпадает", "SHORTAGE": "Недостача", "SURPLUS": "Излишек", "MISMATCH": "Пересортица",
    "PARTIAL": "Частичное совпадение", "MANUAL": "Ручная проверка", "NOT_SCANNED": "Не сканировалась",
}
FLAG_RU = {"GAP": "большой зазор между коробками", "UNDERFILLED": "ячейка заполнена слабо",
           "MULTI": "несколько кодов в ячейке", "POSE_AMBIGUOUS": "поза по одной метке",
           "UNREAD_LABEL": "есть непрочитанная этикетка"}
DECIDED = {Verdict.MATCH, Verdict.SHORTAGE, Verdict.SURPLUS, Verdict.MISMATCH, Verdict.PARTIAL}


def verdict_for(scan_status: str | None, found: list[str], expected: list[str]) -> tuple[Verdict, str]:
    """Вердикт по одной ячейке и пояснение."""
    f, e = set(found), set(expected)
    if not scan_status or scan_status == ScanStatus.SKIPPED:
        return Verdict.NOT_SCANNED, ""
    if scan_status == ScanStatus.EMPTY:
        if not e:
            return Verdict.MATCH, "пусто, как в учёте"
        return Verdict.SHORTAGE, "по учёту числится: " + ", ".join(sorted(e))
    if scan_status == ScanStatus.READ:
        if f == e:
            return Verdict.MATCH, ""
        if not e:
            return Verdict.SURPLUS, "по учёту ячейка пуста"
        if f & e:
            parts = []
            if f - e:
                parts.append("лишние: " + ", ".join(sorted(f - e)))
            if e - f:
                parts.append("не найдены: " + ", ".join(sorted(e - f)))
            return Verdict.PARTIAL, "; ".join(parts)
        return Verdict.MISMATCH, "по учёту числится: " + ", ".join(sorted(e))
    reason = {ScanStatus.NO_LABEL: "коробка без этикетки", ScanStatus.READ_ERROR: "этикетка не читается",
              ScanStatus.UNCERTAIN: "занятость не определена"}.get(ScanStatus(scan_status), "")
    exp = ("; по учёту числится: " + ", ".join(sorted(e))) if e else "; по учёту ячейка пуста"
    return Verdict.MANUAL, reason + exp


def summarize(lines: list[dict]) -> dict:
    """Итоги по ведомости. IRA — точность учёта: доля совпавших среди ячеек с определённым результатом."""
    counts = {v.value: 0 for v in Verdict}
    for ln in lines:
        counts[ln.get("verdict") or "NOT_SCANNED"] += 1
    decided = sum(counts[v.value] for v in DECIDED)
    total = len(lines)
    return {
        "total": total, "counts": counts, "decided": decided,
        "scanned": total - counts["NOT_SCANNED"],
        "ira": round(counts["MATCH"] / decided, 4) if decided else None,
        "manual_share": round(counts["MANUAL"] / total, 4) if total else None,
    }


COLUMNS = ["№", "Адрес ячейки", "По учёту (SSCC)", "Номенклатура", "Фактически (код)", "Сканирование",
           "Результат сверки", "Признаки", "Примечание", "Фото", "Время"]


def _row(i: int, ln: dict) -> list[str]:
    exp = ln.get("expected") or []
    return [
        str(i), ln["address"],
        ", ".join(e["sscc"] for e in exp),
        ", ".join(sorted({e.get("name") or e.get("sku", "") for e in exp} - {""})),
        ", ".join(ln.get("found") or []),
        SCAN_RU.get(ln.get("scan_status") or "", ln.get("scan_status") or ""),
        VERDICT_RU.get(ln.get("verdict") or "NOT_SCANNED", ""),
        ", ".join(FLAG_RU.get(f, f) for f in (ln.get("flags") or [])),
        ln.get("note") or "", ln.get("photo") or "", ln.get("ts") or "",
    ]


def to_csv(lines: list[dict]) -> str:
    """Сличительная ведомость в CSV (разделитель «;», BOM — чтобы Excel открыл кириллицу)."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", lineterminator="\n")
    w.writerow(COLUMNS)
    for i, ln in enumerate(lines, 1):
        w.writerow(_row(i, ln))
    return "﻿" + buf.getvalue()


def to_html(report: dict, title: str = "Сличительная ведомость результатов инвентаризации") -> str:
    s = report["summary"]
    ira = "—" if s["ira"] is None else f"{s['ira'] * 100:.1f} %"
    rows = []
    for i, ln in enumerate(report["lines"], 1):
        cells = _row(i, ln)
        photo = ln.get("photo")
        if photo:
            cells[9] = f'<a href="/evidence/{html.escape(photo)}">фото</a>'
        tds = "".join(f"<td>{c if j == 9 and photo else html.escape(c)}</td>" for j, c in enumerate(cells))
        rows.append(f'<tr class="v-{ln.get("verdict") or "NOT_SCANNED"}">{tds}</tr>')
    c = s["counts"]
    return f"""<!doctype html><html lang="ru"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:14px/1.4 Arial,sans-serif;margin:24px;color:#111}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #999;padding:4px 6px;text-align:left;vertical-align:top}}th{{background:#eee}}
.v-MATCH td:nth-child(7){{color:#0a6b2d}}.v-SHORTAGE td:nth-child(7),.v-SURPLUS td:nth-child(7),
.v-MISMATCH td:nth-child(7),.v-PARTIAL td:nth-child(7){{color:#b00020;font-weight:bold}}
.v-MANUAL td:nth-child(7){{color:#8a5a00;font-weight:bold}}</style>
<h1>{html.escape(title)}</h1>
<p>Задание № {html.escape(str(report.get("task_id", "")))} · создано {html.escape(report.get("created", ""))} ·
состояние: {html.escape(report.get("status", ""))}</p>
<p>Ячеек: {s["total"]} · отсканировано: {s["scanned"]} · совпадает: {c["MATCH"]} · недостача: {c["SHORTAGE"]} ·
излишек: {c["SURPLUS"]} · пересортица: {c["MISMATCH"]} · частично: {c["PARTIAL"]} ·
ручная проверка: {c["MANUAL"]} · точность учёта (IRA): {ira}</p>
<table><tr>{"".join(f"<th>{html.escape(h)}</th>" for h in COLUMNS)}</tr>{"".join(rows)}</table></html>"""
