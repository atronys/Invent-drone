"""Эмулятор склада на SQLite.

Таблицы повторяют объекты, которыми оперирует «1С:WMS Логистика. Управление складом»:
    cells        — справочник ячеек (адрес: стеллаж–ярус–место);
    items        — номенклатура;
    containers   — контейнеры (грузовые места) с кодом SSCC;
    stock        — остатки: какой контейнер в какой ячейке;
    tasks, task_lines — документ «Инвентаризация» и его строки с результатом.
Имена объектов настоящей конфигурации могут отличаться — соответствие задаётся в onec.py.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from .base import Expected, InventoryTask, TaskLine
from .reconcile import Verdict, summarize, verdict_for

SCHEMA = """
CREATE TABLE IF NOT EXISTS cells (address TEXT PRIMARY KEY, rack INTEGER, level INTEGER, pos INTEGER,
                                  zone TEXT DEFAULT 'Хранение', blocked INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS items (sku TEXT PRIMARY KEY, name TEXT, gtin TEXT, unit TEXT DEFAULT 'кор');
CREATE TABLE IF NOT EXISTS containers (sscc TEXT PRIMARY KEY, sku TEXT REFERENCES items(sku), qty REAL DEFAULT 1,
                                       batch TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS stock (sscc TEXT PRIMARY KEY REFERENCES containers(sscc),
                                  address TEXT REFERENCES cells(address));
CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, closed TEXT,
                                  status TEXT DEFAULT 'open');
CREATE TABLE IF NOT EXISTS task_lines (task_id INTEGER, address TEXT, expected TEXT, scan_status TEXT,
                                       found TEXT, verdict TEXT, flags TEXT, note TEXT, photo TEXT, ts TEXT,
                                       confidence REAL, PRIMARY KEY (task_id, address));
"""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class EmulatorBackend:
    """Реализация WmsBackend поверх SQLite. Потокобезопасна (один замок на соединение)."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(SCHEMA)

    # --- наполнение ---------------------------------------------------------------
    def add_cell(self, address: str, rack: int, level: int, pos: int) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO cells(address, rack, level, pos) VALUES (?,?,?,?)",
                             (address, rack, level, pos))

    def add_item(self, sku: str, name: str, gtin: str = "", unit: str = "кор") -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO items(sku, name, gtin, unit) VALUES (?,?,?,?)",
                             (sku, name, gtin, unit))

    def put(self, sscc: str, sku: str, address: str, qty: float = 1.0, batch: str = "") -> None:
        """Разместить контейнер в ячейке (по учёту)."""
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO containers(sscc, sku, qty, batch) VALUES (?,?,?,?)",
                             (sscc, sku, qty, batch))
            self._db.execute("INSERT OR REPLACE INTO stock(sscc, address) VALUES (?,?)", (sscc, address))

    def clear(self) -> None:
        with self._lock, self._db:
            for t in ("task_lines", "tasks", "stock", "containers", "items", "cells"):
                self._db.execute(f"DELETE FROM {t}")

    # --- WmsBackend ---------------------------------------------------------------
    def cells(self) -> list[str]:
        with self._lock:
            return [r["address"] for r in self._db.execute("SELECT address FROM cells ORDER BY address")]

    def stock(self, address: str | None = None) -> list[Expected]:
        q = ("SELECT s.address, s.sscc, c.sku, COALESCE(i.name, '') AS name, c.qty FROM stock s "
             "JOIN containers c ON c.sscc = s.sscc LEFT JOIN items i ON i.sku = c.sku")
        args: tuple = ()
        if address is not None:
            q += " WHERE s.address = ?"
            args = (address,)
        with self._lock:
            return [Expected(r["address"], r["sscc"], r["sku"] or "", r["name"], r["qty"])
                    for r in self._db.execute(q + " ORDER BY s.address, s.sscc", args)]

    def where_is(self, sscc: str) -> str | None:
        with self._lock:
            r = self._db.execute("SELECT address FROM stock WHERE sscc = ?", (sscc,)).fetchone()
        return r["address"] if r else None

    def create_task(self, addresses: list[str] | None = None) -> InventoryTask:
        known = set(self.cells())
        addrs = sorted(known) if addresses is None else list(addresses)
        unknown = [a for a in addrs if a not in known]
        if unknown:
            raise KeyError(f"в справочнике ячеек нет адресов: {', '.join(unknown)}")
        by_addr: dict[str, list[Expected]] = {a: [] for a in addrs}
        for e in self.stock():
            if e.address in by_addr:
                by_addr[e.address].append(e)
        created = _now()
        with self._lock, self._db:
            cur = self._db.execute("INSERT INTO tasks(created) VALUES (?)", (created,))
            tid = cur.lastrowid
            self._db.executemany(
                "INSERT INTO task_lines(task_id, address, expected) VALUES (?,?,?)",
                [(tid, a, json.dumps([e.to_dict() for e in by_addr[a]], ensure_ascii=False)) for a in addrs])
        return InventoryTask(str(tid), created, "open", [TaskLine(a, by_addr[a]) for a in addrs])

    def post_result(self, task_id: str, result: dict) -> dict:
        """result: address, scan_status, codes[], flags[], photo, confidence, ts."""
        address = result["address"]
        with self._lock:
            task = self._db.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
            row = self._db.execute("SELECT expected FROM task_lines WHERE task_id = ? AND address = ?",
                                   (task_id, address)).fetchone()
        if task is None or row is None:
            raise KeyError(f"в задании {task_id} нет ячейки {address}")
        if task["status"] != "open":
            raise ValueError(f"задание {task_id} закрыто")
        expected = [e["sscc"] for e in json.loads(row["expected"])]
        found = list(result.get("codes") or [])
        verdict, note = verdict_for(result.get("scan_status"), found, expected)
        # код найден не там, где числится — подсказка, куда смотреть
        moved = [f"{c} по учёту в {w}" for c in found if (w := self.where_is(c)) and w != address]
        note = "; ".join(x for x in [note, *moved] if x)
        with self._lock, self._db:
            self._db.execute(
                "UPDATE task_lines SET scan_status=?, found=?, verdict=?, flags=?, note=?, photo=?, ts=?, "
                "confidence=? WHERE task_id=? AND address=?",
                (result.get("scan_status"), json.dumps(found), verdict.value,
                 json.dumps(result.get("flags") or []), note, result.get("photo") or "",
                 result.get("ts") or _now(), result.get("confidence"), task_id, address))
        return {"address": address, "verdict": verdict.value, "note": note, "expected": expected}

    def _lines(self, task_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM task_lines WHERE task_id = ? ORDER BY address",
                                    (task_id,)).fetchall()
        out = []
        for r in rows:
            out.append({
                "address": r["address"], "expected": json.loads(r["expected"] or "[]"),
                "scan_status": r["scan_status"], "found": json.loads(r["found"] or "[]"),
                "verdict": r["verdict"] or Verdict.NOT_SCANNED.value, "flags": json.loads(r["flags"] or "[]"),
                "note": r["note"] or "", "photo": r["photo"] or "", "ts": r["ts"] or "",
                "confidence": r["confidence"],
            })
        return out

    def report(self, task_id: str) -> dict:
        with self._lock:
            t = self._db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if t is None:
            raise KeyError(f"задания {task_id} нет")
        lines = self._lines(task_id)
        return {"task_id": str(task_id), "created": t["created"], "closed": t["closed"] or "",
                "status": t["status"], "lines": lines, "summary": summarize(lines)}

    def close_task(self, task_id: str) -> dict:
        with self._lock, self._db:
            self._db.execute("UPDATE tasks SET status='closed', closed=? WHERE id=? AND status='open'",
                             (_now(), task_id))
        return self.report(task_id)

    def tasks(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT * FROM tasks ORDER BY id DESC")]

    # --- обмен файлами -------------------------------------------------------------
    def import_stock_csv(self, text: str) -> int:
        """Загрузка остатков из CSV «адрес;sscc;артикул;наименование;количество» (выгрузка из WMS)."""
        import csv
        import io
        from ..layout import Address
        n = 0
        rows = csv.reader(io.StringIO(text.lstrip("﻿")), delimiter=";")
        for row in rows:
            if not row or not row[0].strip() or not row[0][0].isdigit():
                continue   # заголовок или пустая строка
            address = row[0].strip()
            a = Address.parse(address)
            self.add_cell(address, a.rack, a.level, a.pos)
            if len(row) > 1 and row[1].strip():
                sku = row[2].strip() if len(row) > 2 else ""
                if sku:
                    self.add_item(sku, row[3].strip() if len(row) > 3 else sku)
                qty = float(row[4].replace(",", ".")) if len(row) > 4 and row[4].strip() else 1.0
                self.put(row[1].strip(), sku, address, qty)
                n += 1
        return n

    def export_stock_csv(self) -> str:
        lines = ["адрес;sscc;артикул;наименование;количество"]
        stock = {a: [] for a in self.cells()}
        for e in self.stock():
            stock[e.address].append(e)
        for a, items in stock.items():
            if not items:
                lines.append(f"{a};;;;")
            for e in items:
                lines.append(f"{a};{e.sscc};{e.sku};{e.name};{e.qty:g}")
        return "﻿" + "\n".join(lines) + "\n"
