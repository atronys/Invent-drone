"""Юнит-экономика (BIZ-05): ручная инвентаризация против дрона — затраты в год, экономия, окупаемость.

Все значения по умолчанию — ИЛЛЮСТРАЦИЯ, а не данные рынка: их нужно заменить цифрами из интервью
(docs/custdev/interview_script.md, блок 2–4). Производительность дрона — из планировщика облёта
(ячеек в час с учётом замен АКБ), а не «с потолка».

    python -m invent_core economics --layout my_wh.json --cells 20000 --counts-per-year 12 --manual-sec 20
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Inputs:
    cells: int = 20000                  # паллетомест на складе
    counts_per_year: float = 12.0       # полных пересчётов в год (цель клиента)
    manual_sec_per_cell: float = 20.0   # ручной пересчёт: ричтрак + оператор, с на ячейку
    crew_rate_rub_h: float = 900.0      # ставка оператора(ов) с налогами, ₽/ч
    truck_rate_rub_h: float = 700.0     # стоимость часа техники (амортизация, энергия, ТО), ₽/ч
    downtime_rub_per_count: float = 0.0  # потери от остановки/помех работе склада за пересчёт, ₽
    drone_cells_per_hour: float = 1000.0  # из планировщика: ячеек в час с учётом замен АКБ
    drone_operator_share: float = 0.25  # доля времени оператора при полётах (замена АКБ, контроль)
    drone_capex_rub: float = 600000.0   # дрон + док + камеры + внедрение (разово)
    drone_opex_rub_year: float = 240000.0  # сервис, АКБ, ремонт, ПО в год
    subscription_rub_month: float = 0.0  # если RaaS — абонентская плата (тогда capex клиента = 0)
    manual_accuracy_errors_per_count: float = 0.0  # стоимость ошибок учёта за пересчёт, ₽ (если известна)


@dataclass
class Result:
    manual_hours_year: float
    manual_cost_year: float
    drone_hours_year: float
    drone_cost_year: float
    savings_year: float
    payback_months: float | None


def evaluate(i: Inputs) -> Result:
    manual_h = i.cells * i.counts_per_year * i.manual_sec_per_cell / 3600
    manual = manual_h * (i.crew_rate_rub_h + i.truck_rate_rub_h) + i.counts_per_year * (
        i.downtime_rub_per_count + i.manual_accuracy_errors_per_count)
    drone_h = i.cells * i.counts_per_year / i.drone_cells_per_hour
    drone = drone_h * i.drone_operator_share * i.crew_rate_rub_h + i.drone_opex_rub_year + 12 * i.subscription_rub_month
    savings = manual - drone
    capex = 0.0 if i.subscription_rub_month > 0 else i.drone_capex_rub
    payback = (capex / (savings / 12)) if savings > 0 and capex > 0 else (0.0 if savings > 0 else None)
    return Result(manual_h, manual, drone_h, drone, savings, payback)


def _n(x: float) -> str:
    """12345678.9 -> «12 345 679» (разделитель тысяч — пробел)."""
    return f"{x:,.0f}".replace(",", " ")


def report(i: Inputs, r: Result) -> str:
    lines = [
        "ВНИМАНИЕ: значения по умолчанию — иллюстрация; подставьте данные интервью.",
        f"Склад {_n(i.cells)} ячеек, {i.counts_per_year:g} пересчётов в год",
        f"Вручную: {_n(r.manual_hours_year)} ч/год бригады с техникой -> {_n(r.manual_cost_year)} ₽/год",
        f"Дрон:    {_n(r.drone_hours_year)} ч полётов/год ({_n(i.drone_cells_per_hour)} ячеек/ч) -> "
        f"{_n(r.drone_cost_year)} ₽/год",
        f"Экономия: {_n(r.savings_year)} ₽/год",
        ("Окупаемость: не окупается при этих параметрах" if r.payback_months is None
         else f"Окупаемость вложений {_n(i.drone_capex_rub)} ₽: {r.payback_months:.1f} мес"),
    ]
    return "\n".join(lines)


def breakeven_counts(i: Inputs) -> float:
    """При скольких пересчётах в год экономия становится положительной (при прочих равных)."""
    per_count_manual = i.cells * i.manual_sec_per_cell / 3600 * (i.crew_rate_rub_h + i.truck_rate_rub_h) \
        + i.downtime_rub_per_count + i.manual_accuracy_errors_per_count
    per_count_drone = i.cells / i.drone_cells_per_hour * i.drone_operator_share * i.crew_rate_rub_h
    fixed = i.drone_opex_rub_year + 12 * i.subscription_rub_month
    margin = per_count_manual - per_count_drone
    return fixed / margin if margin > 0 else float("inf")


def as_dict(i: Inputs) -> dict:
    return asdict(i)
