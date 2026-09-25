"""SIM-04: метрики чтения этикеток по истине симуляции.

    python sim/eval_read_rate.py sim/worlds/demo/truth.csv ~/invent_runs/<вылет>.csv [--targets addr.csv]

read rate   — доля занятых ячеек (из тех, что были в плане вылета), где прочитан верный код;
attribution — доля прочтений, привязанных к верному адресу;
misattributed — код верный, но приписан чужой ячейке (ошибка привязки PER-06);
garbage     — код, которого нет на складе (ошибка декодирования).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from invent_core.reconcile import Observation, load_expected_csv, load_observations_csv  # noqa: E402


def evaluate(truth: dict[str, str | None], observations: list[Observation],
             targeted: set[str] | None = None) -> dict:
    where = {code: addr for addr, code in truth.items() if code}
    scope = set(truth) if targeted is None else targeted & set(truth)
    occupied = {a for a in scope if truth[a]}
    reads = [o for o in observations if o.code]
    correct = [o for o in reads if truth.get(o.address) == o.code]
    misattributed = [o for o in reads if truth.get(o.address) != o.code and o.code in where]
    garbage = [o for o in reads if o.code not in where]
    read_ok = {o.address for o in correct} & occupied
    return {
        "occupied_in_scope": len(occupied),
        "read_rate": len(read_ok) / len(occupied) if occupied else 0.0,
        "attribution": len(correct) / len(reads) if reads else 0.0,
        "reads": len(reads),
        "misattributed": len(misattributed),
        "garbage": len(garbage),
        "missed": sorted(occupied - read_ok)[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("truth")
    ap.add_argument("observations")
    ap.add_argument("--targets", help="адреса в плане вылета (по одному в строке или CSV с address)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    targeted = None
    if a.targets:
        lines = [ln.strip().split(";")[0] for ln in open(a.targets, encoding="utf-8-sig") if ln.strip()]
        targeted = {ln for ln in lines if ln.lower() not in ("address", "адрес")}
    obs = load_observations_csv(a.observations)
    if targeted is None:  # по умолчанию — ячейки, которые рекордер видел в плане
        targeted = {o.address for o in obs}
    m = evaluate(load_expected_csv(a.truth), obs, targeted)
    print(json.dumps(m, ensure_ascii=False, indent=1))
    print(f"read rate {m['read_rate']:.1%} (цель Ф1 >= 95%), атрибуция {m['attribution']:.1%} (цель >= 99%)")


if __name__ == "__main__":
    main()
