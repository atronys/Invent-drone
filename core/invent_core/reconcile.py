"""Сверка результатов облёта с данными WMS/1С -> отчёт о расхождениях.

Ожидаемое состояние выгружается из WMS/1С в CSV (адрес;код). Пустой код — ячейка
должна быть пустой. Наблюдения дрона — (адрес, код|None, занято?, уверенность).
Занятость может быть неизвестна (None): пока нет датчика занятости (PER-08), ячейку без
прочитанной этикетки нельзя честно назвать пустой — такие случаи идут в UNREADABLE/примечание.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Status(str, Enum):
    OK = "OK"                    # совпало (в т.ч. пусто = пусто)
    MISMATCH = "MISMATCH"        # в ячейке другая паллета
    MISSING = "MISSING"          # ждали паллету — ячейка пуста
    UNEXPECTED = "UNEXPECTED"    # ждали пусто — стоит паллета
    UNREADABLE = "UNREADABLE"    # паллета есть, этикетка не прочитана -> перескан
    NOT_SCANNED = "NOT_SCANNED"  # дрон не видел ячейку


@dataclass(frozen=True)
class Observation:
    address: str
    code: str | None
    occupied: bool | None = True   # None — занятость не определялась
    confidence: float = 1.0


@dataclass(frozen=True)
class LocationResult:
    address: str
    status: Status
    expected: str | None
    found: str | None
    note: str = ""


@dataclass
class Report:
    results: list[LocationResult]

    def count(self, status: Status) -> int:
        return sum(r.status == status for r in self.results)

    @property
    def accuracy(self) -> float:
        """Точность учёта (IRA): доля совпавших среди ячеек с определённым результатом."""
        decided = [r for r in self.results if r.status not in (Status.UNREADABLE, Status.NOT_SCANNED)]
        return sum(r.status == Status.OK for r in decided) / len(decided) if decided else 0.0

    def summary(self) -> dict[str, int]:
        return {s.value: self.count(s) for s in Status}


def aggregate(observations: list[Observation]) -> dict[str, tuple[bool | None, str | None, str]]:
    """Несколько наблюдений на ячейку -> (занято|None, лучший код, примечание о конфликте)."""
    by_addr: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        by_addr[o.address].append(o)
    out = {}
    for addr, obs in by_addr.items():
        votes: dict[str, float] = defaultdict(float)
        for o in obs:
            if o.code:
                votes[o.code] += o.confidence
        code = max(votes, key=votes.get) if votes else None
        note = f"конфликт чтений: {sorted(votes)}" if len(votes) > 1 else ""
        if votes or any(o.occupied for o in obs):
            occupied = True
        elif all(o.occupied is None for o in obs):
            occupied = None
        else:
            occupied = False
        out[addr] = (occupied, code, note)
    return out


def reconcile(expected: dict[str, str | None], observations: list[Observation]) -> Report:
    seen = aggregate(observations)
    home = {code: addr for addr, code in expected.items() if code}   # где код должен быть
    where = {code: addr for addr, (_, code, _) in seen.items() if code}  # где код найден

    results = []
    for addr in sorted(set(expected) | set(seen)):
        exp = expected.get(addr) or None
        if addr not in seen:
            results.append(LocationResult(addr, Status.NOT_SCANNED, exp, None))
            continue
        occupied, found, note = seen[addr]
        notes = [note] if note else []
        if found and found != exp and found in home:
            notes.append(f"по WMS паллета числится в {home[found]}")
        if exp and not found and not occupied and exp in where:
            notes.append(f"паллета найдена в {where[exp]}")
        if occupied is None and not found:
            notes.append("занятость не определялась")
        if exp:
            # Паллету видели в другой ячейке -> здесь её точно нет (одна паллета не бывает в двух местах),
            # даже если занятость этой ячейки не определялась.
            elsewhere = exp in where and where[exp] != addr
            status = (Status.OK if found == exp else Status.MISMATCH if found
                      else Status.MISSING if occupied is False or elsewhere else Status.UNREADABLE)
        else:
            status = Status.UNEXPECTED if occupied else Status.OK
        results.append(LocationResult(addr, status, exp, found, "; ".join(notes)))
    return Report(results)


# --- CSV ----------------------------------------------------------------------

def _read_rows(path: str | Path) -> list[dict[str, str]]:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1251"):  # выгрузки 1С часто в cp1251
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"{path}: не удалось определить кодировку")
    lines = text.splitlines()
    delimiter = max(";,\t", key=lines[0].count)  # «русский» Excel пишет ';'
    reader = csv.DictReader(lines, delimiter=delimiter)
    return [{(k or "").strip().lower(): (v or "").strip() for k, v in row.items()} for row in reader]


def _pick(row: dict[str, str], *names: str) -> str:
    for n in names:
        if n in row:
            return row[n]
    raise KeyError(f"нет ни одного из столбцов {names}: {list(row)}")


def load_expected_csv(path: str | Path) -> dict[str, str | None]:
    """CSV из WMS/1С: столбцы address|адрес и code|код|sscc."""
    return {_pick(r, "address", "адрес"): (_pick(r, "code", "код", "sscc") or None) for r in _read_rows(path)}


def _parse_occupied(value: str | None) -> bool | None:
    if value is None:
        return True          # столбца нет — старый формат: наблюдение = паллета видна
    v = value.strip().lower()
    if v == "":
        return None
    return v not in ("0", "false", "нет", "no")


def load_observations_csv(path: str | Path) -> list[Observation]:
    """CSV наблюдений: address;code;occupied;confidence (occupied: 1/0/пусто = неизвестно)."""
    out = []
    for r in _read_rows(path):
        out.append(Observation(
            address=r["address"],
            code=r.get("code") or None,
            occupied=_parse_occupied(r.get("occupied")),
            confidence=float(r.get("confidence") or 1.0),
        ))
    return out


def write_observations_csv(observations: list[Observation], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["address", "code", "occupied", "confidence"])
        for o in observations:
            occ = "" if o.occupied is None else int(o.occupied)
            w.writerow([o.address, o.code or "", occ, f"{o.confidence:.3f}"])


class SortieRecorder:
    """Сбор наблюдений за вылет (PER-07): прочитанные этикетки + ячейки, которые были в кадре.

    Ячейка из плана без чтения -> Observation(code=None, occupied=None): без датчика занятости
    это «не прочитано», а не «пусто».
    """

    def __init__(self) -> None:
        self.expected: set[str] = set()
        self.reads: list[Observation] = []

    def expect(self, addresses) -> None:
        self.expected.update(a for a in addresses if a)

    def read(self, address: str, code: str, confidence: float = 1.0) -> None:
        if address and code:
            self.reads.append(Observation(address, code, True, confidence))

    def observations(self) -> list[Observation]:
        seen = {o.address for o in self.reads}
        return self.reads + [Observation(a, None, None, 0.0) for a in sorted(self.expected - seen)]


def write_report_csv(report: Report, path: str | Path) -> None:
    """utf-8-sig + ';' — открывается в русском Excel без танцев."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["адрес", "статус", "ожидалось", "найдено", "примечание"])
        for r in report.results:
            w.writerow([r.address, r.status.value, r.expected or "", r.found or "", r.note])
