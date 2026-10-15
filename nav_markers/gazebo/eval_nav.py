"""Оценка прогона навигации по меткам: точность EKF2 и позы по меткам относительно истины Gazebo, захват ячеек.

    python3 nav_markers/gazebo/eval_nav.py nav_markers/runs/<время>      # -> summary.txt, summary.json, track.png

Метрики:
* ошибка EKF2 по горизонтали (то, по чему летит PX4): RMSE, 95-й перцентиль, максимум — в полёте;
  отдельно — «свежая метка» (поза по меткам моложе 0.5 с) и «между метками» (дрейф потока);
* ошибка позы по меткам (выход marker_localizer) и согласованность её ковариации;
* захват ячейки: расстояние от истинного положения до точки захвата (KPI проекта — 10 см);
* покрытие метками: доля времени полёта со свежей позой, самый длинный пропуск;
* события: поиск метки, посадка по потере, неудачный захват.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


# Истина Gazebo (PosePublisher) — начало модели x500_nav, а EKF2 и уставки — base_link. x500_base включён
# с merge='true' и своей позой <pose>0 0 .24 0 0 0</pose> — она переносится на звенья, и base_link на 0.24 м
# выше начала модели (видно и в прогоне: на доке поза по меткам z ≈ 0.22 при истине −0.02). По горизонтали совпадают.
BASE_LINK_Z = 0.24


def read(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def stats(xs):
    xs = sorted(x for x in xs if math.isfinite(x))
    if not xs:
        return {"n": 0}
    rmse = math.sqrt(sum(x * x for x in xs) / len(xs))
    return {"n": len(xs), "rmse_cm": round(rmse * 100, 1), "p95_cm": round(xs[int(0.95 * (len(xs) - 1))] * 100, 1),
            "max_cm": round(xs[-1] * 100, 1)}


def evaluate(run: Path) -> dict:
    track = read(run / "track.csv")
    fixes = read(run / "fixes.csv") if (run / "fixes.csv").exists() else []
    events = read(run / "events.csv") if (run / "events.csv").exists() else []
    f = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else math.nan  # noqa: E731
    flying = [r for r in track if r["state"] in ("FOLLOW", "SEARCH", "CAPTURE")]
    err = lambda r: math.hypot(f(r, "ex") - f(r, "tx"), f(r, "ey") - f(r, "ty"))  # noqa: E731
    fresh = [err(r) for r in flying if f(r, "fix_age") < 0.5]
    stale = [err(r) for r in flying if not f(r, "fix_age") < 0.5]
    gaps, cur, prev_t = [], 0.0, None
    for r in flying:
        t = f(r, "t")
        if prev_t is not None and not f(r, "fix_age") < 0.5:
            cur += t - prev_t
        elif cur:
            gaps.append(cur)
            cur = 0.0
        prev_t = t
    gaps.append(cur)
    fix_err = [math.hypot(f(r, "fx") - f(r, "tx"), f(r, "fy") - f(r, "ty")) for r in fixes]
    fix_nees = [(e / max(f(r, "fsigma_xy"), 1e-4)) ** 2 for e, r in zip(fix_err, fixes) if math.isfinite(e)]
    caps = []
    counts: dict[str, int] = {}
    for r in events:
        ev = json.loads(r["event"])
        counts[ev["event"]] = counts.get(ev["event"], 0) + 1
        if ev["event"] == "capture":
            tgt = ev["target"]
            truth = (float(r["tx"]), float(r["ty"]), float(r["tz"]) + BASE_LINK_Z)
            caps.append({"cell": ev["label"], "err_cm": round(math.dist(tgt, truth) * 100, 1),
                         "err_xy_cm": round(math.hypot(tgt[0] - truth[0], tgt[1] - truth[1]) * 100, 1),
                         "fix_sigma_cm": round(ev.get("fix_sigma", math.nan) * 100, 1), "waited_s": ev.get("waited_s")})
    duration = (f(flying[-1], "t") - f(flying[0], "t")) if flying else 0.0
    return {
        "run": run.name, "flight_s": round(duration, 1), "events": counts,
        "ekf2_xy_all": stats(fresh + stale), "ekf2_xy_fresh_marker": stats(fresh), "ekf2_xy_between_markers": stats(stale),
        "marker_fix_xy": stats(fix_err), "marker_fix_nees_xy": round(sum(fix_nees) / len(fix_nees), 2) if fix_nees else None,
        "marker_coverage": round(len(fresh) / max(1, len(flying)), 3), "longest_gap_s": round(max(gaps), 1),
        "captures": caps,
    }


def plot(run: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    track = read(run / "track.csv")
    fl = [r for r in track if r["state"] in ("FOLLOW", "SEARCH", "CAPTURE")]
    if not fl:
        return
    g = lambda k: [float(r[k]) for r in fl]  # noqa: E731
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8))
    a1.plot(g("tx"), g("ty"), "k-", lw=1, label="истина")
    a1.plot(g("ex"), g("ey"), "b--", lw=1, label="EKF2")
    if (run / "marker_map.json").exists():
        mm = json.loads((run / "marker_map.json").read_text(encoding="utf-8"))
        a1.scatter([m["x"] for m in mm["markers"]], [m["y"] for m in mm["markers"]], s=8, c="r", label="метки")
    a1.set_aspect("equal")
    a1.set_xlabel("x, м")
    a1.set_ylabel("y, м")
    a1.legend()
    t = g("t")
    e = [math.hypot(r1 - r2, s1 - s2) * 100 for r1, r2, s1, s2 in zip(g("ex"), g("tx"), g("ey"), g("ty"))]
    a2.plot([x - t[0] for x in t], e, "b-", lw=1)
    a2.set_xlabel("время полёта, с")
    a2.set_ylabel("ошибка EKF2 по горизонтали, см")
    a2.grid(True)
    fig.tight_layout()
    fig.savefig(run / "track.png", dpi=120)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run = Path(sys.argv[1])
    s = evaluate(run)
    (run / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"Прогон {s['run']}: полёт {s['flight_s']} с, события {s['events']}",
             f"EKF2 xy (вся траектория):      {s['ekf2_xy_all']}",
             f"EKF2 xy (метка свежая <0.5 с):  {s['ekf2_xy_fresh_marker']}",
             f"EKF2 xy (между метками):        {s['ekf2_xy_between_markers']}",
             f"Поза по меткам xy:             {s['marker_fix_xy']}, NEES_xy {s['marker_fix_nees_xy']} (норма ≈ 2)",
             f"Покрытие метками: {s['marker_coverage'] * 100:.0f}% времени полёта, самый длинный пропуск {s['longest_gap_s']} с"]
    for c in s["captures"]:
        ok = "OK" if c["err_cm"] <= 10 else "ХУЖЕ KPI"
        lines.append(f"Захват {c['cell']}: ошибка {c['err_cm']} см (xy {c['err_xy_cm']}), σ по меткам {c['fix_sigma_cm']} см — {ok}")
    text = "\n".join(lines)
    print(text)
    with open(run / "summary.txt", "a", encoding="utf-8") as fh:
        fh.write(text + "\n")
    plot(run)


if __name__ == "__main__":
    main()
