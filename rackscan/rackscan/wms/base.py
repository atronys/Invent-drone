"""Интерфейс складской системы, одинаковый для эмулятора и для настоящей 1С."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Expected:
    """Строка остатка: что числится в ячейке по учёту."""

    address: str
    sscc: str
    sku: str = ""
    name: str = ""
    qty: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskLine:
    address: str
    expected: list[Expected] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"address": self.address, "expected": [e.to_dict() for e in self.expected]}


@dataclass
class InventoryTask:
    """Задание на инвентаризацию: перечень ячеек и учётные остатки на момент создания."""

    id: str
    created: str
    status: str = "open"
    lines: list[TaskLine] = field(default_factory=list)

    # Как это называется в «1С:WMS Логистика. Управление складом» (редакция 5): инвентаризация
    # оформляется документом «Плановая инвентаризация», работа по ячейке — задачей на пересчёт
    # типа «Пересчет ячейки». Стенд выполняет именно её: считает контейнеры в ячейке и их коды.
    DOCUMENT = "Плановая инвентаризация"
    TASK_TYPE = "Пересчет ячейки"

    def to_dict(self) -> dict:
        return {"id": self.id, "created": self.created, "status": self.status,
                "document": self.DOCUMENT, "task_type": self.TASK_TYPE,
                "lines": [ln.to_dict() for ln in self.lines]}

    @classmethod
    def from_dict(cls, d: dict) -> "InventoryTask":
        lines = [TaskLine(ln["address"], [Expected(**e) for e in ln.get("expected", [])]) for ln in d["lines"]]
        return cls(str(d["id"]), d.get("created", ""), d.get("status", "open"), lines)


class WmsBackend(Protocol):
    def cells(self) -> list[str]:
        """Адреса всех ячеек."""

    def stock(self, address: str | None = None) -> list[Expected]:
        """Остатки по ячейке или по всему складу."""

    def create_task(self, addresses: list[str] | None = None) -> InventoryTask:
        """Создать задание на инвентаризацию (по умолчанию — все ячейки)."""

    def post_result(self, task_id: str, result: dict) -> dict:
        """Передать результат сканирования ячейки; ответ — вердикт сверки."""

    def report(self, task_id: str) -> dict:
        """Сличительная ведомость задания."""

    def close_task(self, task_id: str) -> dict:
        """Закрыть задание."""
