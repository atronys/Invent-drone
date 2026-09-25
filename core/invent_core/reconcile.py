"""Сверка результатов облёта с данными WMS/1С -> отчёт о расхождениях.

Ожидаемое состояние выгружается из WMS/1С в CSV (адрес;код). Пустой код — ячейка
должна быть пустой. Наблюдения дрона — (адрес, код|None, занято?, уверенность).
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
    occupied: bool = True
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


def aggregate(observations: list[Observation]) -> dict[str, tuple[bool, str | None, str]]:
    """Несколько наблюдений на ячейку -> (занято, лучший код, примечание о конфликте)."""
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
        out[addr] = (bool(votes) or any(o.occupied for o in obs), code, note)
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
        if exp:
            status = (Status.OK if found == exp else Status.MISMATCH if found
                      else Status.UNREADABLE if occupied else Status.MISSING)
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


def load_observations_csv(path: str | Path) -> list[Observation]:
    """CSV наблюдений: address;code;occupied;confidence (occupied: 1/0, по умолчанию 1)."""
    out = []
    for r in _read_rows(path):
        out.append(Observation(
            address=r["address"],
            code=r.get("code") or None,
            occupied=r.get("occupied", "1") not in ("0", "false", "нет", ""),
            confidence=float(r.get("confidence") or 1.0),
        ))
    return out


def write_report_csv(report: Report, path: str | Path) -> None:
    """utf-8-sig + ';' — открывается в русском Excel без танцев."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["адрес", "статус", "ожидалось", "найдено", "примечание"])
        for r in report.results:
            w.writerow([r.address, r.status.value, r.expected or "", r.found or "", r.note])
