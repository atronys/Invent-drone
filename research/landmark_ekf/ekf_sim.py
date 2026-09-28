"""Прототип LOC-08: EKF-локализация дрона по складским этикеткам с известными координатами.

Основа — Gao & Zhang, «Introduction to Visual SLAM»:
* EKF, гл. 8.1.3: прогноз ф. (8.32), усиление ф. (8.34), апостериорная оценка ф. (8.35).
  Обозначения книги: R — шум движения, Q — шум наблюдения (в других книгах наоборот).
* Гл. 8.1.4, п. 4: «у EKF нет механизма отсева выбросов» -> добавлен χ²-гейтинг по расстоянию
  Махаланобиса. На складе выбросы неизбежны: ошибка привязки этикетки к ячейке, переставленная паллета.
* Метрика — ATE_trans, гл. 3.4.2, ф. (3.45): СКО ошибки положения по всей траектории.
* Ориентиры известны заранее (карта = план склада), поэтому это EKF-локализация, а не EKF-SLAM:
  ориентиры не входят в вектор состояния, и проблема квадратичного роста ковариации (8.1.4, п. 3) не возникает.

Сценарий: дрон летит по плану invent_core.coverage (демо-склад). Состояние 2D: (x, y, ψ);
высота — по дальномеру, вне этой модели. Одометрия VIO — с ошибкой масштаба, дрейфом курса
и шумом. Сравниваются:
  odom          — только VIO;
  tags          — EKF + AprilTag на торцах рядов (решение MVP, ADR-4);
  labels        — EKF + адресные этикетки ячеек (есть у каждой ячейки), 2% ошибок привязки, гейтинг;
  labels_nogate — то же без гейтинга (иллюстрация п. 4 из 8.1.4);
  pallets       — EKF + паллетные этикетки, 3% паллет стоят не там, где думает WMS, гейтинг.

    python research/landmark_ekf/ekf_sim.py --seeds 10 --plot
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
from invent_core.coverage import ScanParams, plan, segment_time  # noqa: E402
from invent_core.layout import Warehouse, make_grid_warehouse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window_smoother import WindowSmoother, compose  # noqa: E402

CHI2_2DOF_99 = 9.21  # χ²(2 степени свободы, 99%)


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# --- истинная траектория по плану ---------------------------------------------------

def true_trajectory(wps, p: ScanParams, dt: float) -> np.ndarray:
    """Позы (x, y, ψ) с шагом dt: движение по отрезкам плана с длительностью segment_time."""
    out = [(wps[0].x, wps[0].y, wps[0].yaw)]
    for a, b in zip(wps, wps[1:]):
        n = max(1, round(segment_time(a, b, p) / dt))
        dyaw = wrap(b.yaw - a.yaw)
        for i in range(1, n + 1):
            k = i / n
            out.append((a.x + (b.x - a.x) * k, a.y + (b.y - a.y) * k, wrap(a.yaw + dyaw * k)))
    return np.array(out)


# --- ориентиры ---------------------------------------------------------------------

@dataclass
class Landmark:
    x: float          # истинное положение этикетки
    y: float
    face_normal: int  # куда смотрит фасад (+1/-1 по y)
    map_x: float = 0.0  # где этикетка по карте (у переставленной паллеты — не там)
    map_y: float = 0.0


# Точность «карты» ориентиров: где этикетка реально относительно плана склада (σ, м)
MAP_SIGMA = {"tags": 0.01, "labels": 0.02, "pallets": 0.08}  # паллета стоит в ячейке неточно


def make_landmarks(wh: Warehouse, kind: str, rng: random.Random, misplaced: float = 0.03,
                   occupancy: float = 0.8) -> list[Landmark]:
    """tags — AprilTag на торцах; labels — адресные этикетки на балке (у каждой ячейки);
    pallets — этикетки паллет (только занятые ячейки, часть паллет стоит не там, где в WMS)."""
    lms = []
    sig = MAP_SIGMA[kind]
    for row in wh.rows:
        if kind == "tags":
            xs = [row.origin_x - 0.15, row.end_x + 0.15]
        else:
            xs = [row.origin_x + b * row.bay_width + (s + 0.5) * row.slot_width
                  for b in range(row.bays) for s in range(row.slots_per_bay)]
        for x in xs:
            if kind == "pallets" and rng.random() > occupancy:
                continue
            true_x = x + rng.gauss(0, sig)             # реальное положение отличается от плана
            lm = Landmark(true_x, row.face_y, row.normal_y, x, row.face_y)
            if kind == "pallets" and rng.random() < misplaced:
                lm.map_x = x + rng.choice((-1, 1)) * row.bay_width  # WMS думает, что паллета в соседней секции
            lms.append(lm)
    return lms


# --- модель датчиков -------------------------------------------------------------------

@dataclass
class SensorModel:
    scale_err: float = 0.01          # СКО ошибки масштаба VIO (1%)
    yaw_bias_deg_min: float = 0.3    # СКО дрейфа курса, град/мин
    trans_noise: float = 0.02        # шум приращения, доля шага
    trans_noise_min: float = 1e-4    # минимальный шум приращения, м/шаг
    yaw_noise_deg: float = 0.01      # шум приращения курса, град/шаг (10 Гц)
    obs_rate_hz: float = 5.0         # частота декодирования кадров
    p_detect: float = 0.8            # вероятность прочитать этикетку в кадре
    hfov_deg: float = 50.0           # боковые камеры
    max_range: float = 3.0
    obs_sigma0: float = 0.01         # шум PnP: σ = σ0 + k·дальность
    obs_sigma_k: float = 0.02
    p_misassoc: float = 0.02         # ошибка привязки к соседней ячейке
    slot_width: float = 0.9


def visible(pose, lm: Landmark, sm: SensorModel) -> bool:
    """Этикетка в кадре одной из боковых камер: фасад смотрит на дрон и попадает в угол обзора."""
    x, y, psi = pose
    dx, dy = lm.x - x, lm.y - y
    if (y - lm.y) * lm.face_normal <= 0 or math.hypot(dx, dy) > sm.max_range:
        return False
    bearing = math.atan2(dy, dx)
    look = psi + (math.pi / 2 if wrap(bearing - psi) > 0 else -math.pi / 2)  # левая или правая камера
    ang = wrap(bearing - look)
    return abs(ang) <= math.radians(sm.hfov_deg) / 2


def body_frame(pose, px: float, py: float) -> np.ndarray:
    x, y, psi = pose
    c, s = math.cos(psi), math.sin(psi)
    dx, dy = px - x, py - y
    return np.array([c * dx + s * dy, -s * dx + c * dy])


# --- EKF (гл. 8.1.3) ----------------------------------------------------------------

@dataclass
class EKF:
    mu: np.ndarray
    P: np.ndarray
    gate: bool = True
    rejected: int = 0
    accepted: int = 0
    # LOC-12: после inflate_after отказов подряд фильтр признаёт, что «потерялся», и раздувает P,
    # чтобы снова принимать верные наблюдения (ловушка редких ориентиров при плохом VIO)
    inflate_after: int | None = None
    inflate_factor: float = 4.0
    streak: int = 0
    inflations: int = 0

    def predict(self, u: np.ndarray, R_u: np.ndarray) -> None:
        """Ф. (8.32): μ = f(μ, u), P = F P Fᵀ + R. u = (dx, dy, dψ) в кадре тела."""
        x, y, psi = self.mu
        c, s = math.cos(psi), math.sin(psi)
        dx, dy, dpsi = u
        self.mu = np.array([x + c * dx - s * dy, y + s * dx + c * dy, wrap(psi + dpsi)])
        F = np.array([[1, 0, -s * dx - c * dy], [0, 1, c * dx - s * dy], [0, 0, 1]])
        G = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])   # ∂f/∂u: шум одометрии -> шум состояния
        self.P = F @ self.P @ F.T + G @ R_u @ G.T

    def update(self, z: np.ndarray, lx: float, ly: float, Q: np.ndarray) -> bool:
        """Ф. (8.34)–(8.35) с χ²-гейтингом. z — положение ориентира в кадре тела."""
        x, y, psi = self.mu
        c, s = math.cos(psi), math.sin(psi)
        dx, dy = lx - x, ly - y
        h = np.array([c * dx + s * dy, -s * dx + c * dy])
        H = np.array([[-c, -s, -s * dx + c * dy], [s, -c, -c * dx - s * dy]])
        S = H @ self.P @ H.T + Q
        nu = z - h
        if self.gate and float(nu @ np.linalg.solve(S, nu)) > CHI2_2DOF_99:
            self.rejected += 1
            self.streak += 1
            if self.inflate_after and self.streak >= self.inflate_after:
                self.P = self.P * self.inflate_factor
                self.streak = 0
                self.inflations += 1
            return False
        self.streak = 0
        K = self.P @ H.T @ np.linalg.inv(S)                 # (8.34)
        self.mu = self.mu + K @ nu                           # (8.35)
        self.mu[2] = wrap(self.mu[2])
        self.P = (np.eye(3) - K @ H) @ self.P
        self.accepted += 1
        return True


# --- прогон -------------------------------------------------------------------------

@dataclass
class RunResult:
    ate: float           # ATE_trans, м (ф. 3.45)
    path: float          # длина пути, м
    max_err: float
    final_err: float
    accepted: int = 0
    rejected: int = 0
    errors: list = field(default_factory=list)
    cpu_s: float = 0.0   # время оценщика (без моделирования)


def run(truth: np.ndarray, dt: float, scenario: str, lms: list[Landmark], sm: SensorModel, seed: int) -> RunResult:
    # раздельные потоки случайных чисел: у всех сценариев ОДИНАКОВАЯ одометрия,
    # различаются только коррекции — иначе сравнение нечестное
    rng = np.random.default_rng(seed)
    obs_rng = np.random.default_rng(seed + 1_000_000)
    scale = 1 + rng.normal(0, sm.scale_err)
    yaw_bias = math.radians(rng.normal(0, sm.yaw_bias_deg_min)) / 60 * dt
    path = 0.0
    est = truth[0].copy()
    P0 = np.diag([0.02, 0.02, math.radians(1)]) ** 2
    win = "_win" in scenario                      # оптимизация в окне (гл. 9) вместо EKF (гл. 8)
    base = scenario.split("_")[0]                 # tags / labels / pallets / odom
    ekf = EKF(truth[0].copy(), P0, gate=scenario != "labels_nogate",
              inflate_after=3 if scenario.endswith("_adapt") else None)
    # _win — ядро Хьюбера + порог χ² (как у EKF); _winh — только ядро Хьюбера (как в книге, гл. 8.2.6)
    sw = WindowSmoother(truth[0].copy(), P0, gate=None if scenario.endswith("_winh") else CHI2_2DOF_99) if win else None
    rel, rel_var = np.zeros(3), np.zeros(3)       # одометрия с последнего ключевого кадра
    obs_every = max(1, round(1 / (sm.obs_rate_hz * dt)))
    errors = []
    cpu = 0.0
    for k in range(1, len(truth)):
        a, b = truth[k - 1], truth[k]
        c, s = math.cos(a[2]), math.sin(a[2])
        dxw, dyw = b[0] - a[0], b[1] - a[1]
        d_body = np.array([c * dxw + s * dyw, -s * dxw + c * dyw])
        step = float(np.hypot(*d_body))
        path += step
        d_meas = scale * d_body + rng.normal(0, sm.trans_noise * step + sm.trans_noise_min, 2)
        dpsi = wrap(b[2] - a[2]) + yaw_bias + rng.normal(0, math.radians(sm.yaw_noise_deg))
        u = np.array([d_meas[0], d_meas[1], dpsi])
        sig_t = 0.03 * step + 2e-3            # шум процесса с запасом: фильтр не знает ни масштаб, ни дрейф
        R_u = np.diag([sig_t, sig_t, math.radians(0.03)]) ** 2
        if scenario == "odom":
            ce, se = math.cos(est[2]), math.sin(est[2])
            est = np.array([est[0] + ce * u[0] - se * u[1], est[1] + se * u[0] + ce * u[1], wrap(est[2] + u[2])])
        else:
            t0 = time.perf_counter()
            if win:
                rel = compose(rel, u)
                rel_var += np.diag(R_u)
                if k % obs_every == 0:                  # ключевой кадр на каждом кадре камеры
                    sw.add_keyframe(rel, np.diag(rel_var))
                    rel, rel_var = np.zeros(3), np.zeros(3)
            else:
                ekf.predict(u, R_u)
            cpu += time.perf_counter() - t0
            if k % obs_every == 0:
                for lm in lms:
                    if not visible(b, lm, sm) or obs_rng.random() > sm.p_detect:
                        continue
                    rng_m = math.hypot(lm.x - b[0], lm.y - b[1])
                    sigma = sm.obs_sigma0 + sm.obs_sigma_k * rng_m
                    z = body_frame(b, lm.x, lm.y) + obs_rng.normal(0, sigma, 2)
                    mx, my = lm.map_x, lm.map_y
                    if base != "tags" and obs_rng.random() < sm.p_misassoc:
                        mx += obs_rng.choice((-1, 1)) * sm.slot_width  # прочитали, но привязали к соседней ячейке
                    sig_map = MAP_SIGMA["tags" if base == "tags" else "pallets" if base == "pallets" else "labels"]
                    q = sigma ** 2 + sig_map ** 2                 # ошибка карты — тоже шум наблюдения
                    t0 = time.perf_counter()
                    if win:
                        sw.add_obs(z, (mx, my), np.diag([q, q]))
                    else:
                        ekf.update(z, mx, my, np.diag([q, q]))
                    cpu += time.perf_counter() - t0
            t0 = time.perf_counter()
            if win:
                if k % obs_every == 0:
                    sw.optimize()
                est = compose(sw.latest, rel)
            else:
                est = ekf.mu
            cpu += time.perf_counter() - t0
        errors.append(math.hypot(est[0] - b[0], est[1] - b[1]))
    e = np.array(errors)
    acc, rej = (sw.total_obs - sw.downweighted, sw.downweighted) if win else (ekf.accepted, ekf.rejected)
    return RunResult(float(np.sqrt(np.mean(e ** 2))), path, float(e.max()), float(e[-1]), acc, rej, errors, cpu)


SCENARIOS = {"odom": "labels", "tags": "tags", "labels": "labels", "labels_nogate": "labels", "pallets": "pallets"}
# оптимизация в скользящем окне (гл. 9) на тех же данных: --window
WINDOW_SCENARIOS = {"tags_win": "tags", "labels_win": "labels", "pallets_win": "pallets", "pallets_winh": "pallets"}
# LOC-12: инфляция ковариации после серии отказов: --adapt
ADAPT_SCENARIOS = {"tags_adapt": "tags", "labels_adapt": "labels", "pallets_adapt": "pallets"}


def experiment(wh: Warehouse, seeds: int, dt: float = 0.1, sm: SensorModel | None = None, window: bool = False,
               adapt: bool = False):
    sm = sm or SensorModel()
    p = ScanParams()
    pl = plan(wh, p)
    trajs = [true_trajectory(s.waypoints, p, dt) for s in pl.sorties]
    scenarios = {**SCENARIOS, **(WINDOW_SCENARIOS if window else {}), **(ADAPT_SCENARIOS if adapt else {})}
    results = {name: [] for name in scenarios}
    for seed in range(seeds):
        for name, kind in scenarios.items():
            lms = make_landmarks(wh, kind, random.Random(seed))
            for i, tr in enumerate(trajs):
                results[name].append(run(tr, dt, name, lms, sm, seed * 100 + i))
    return results, trajs


def summarize(results) -> str:
    lines = ["| Сценарий | ATE_trans, м (ф. 3.45) | Макс. ошибка, м | Ошибка в конце, м | В конце, % пути "
             "| Отброшено гейтингом / ослаблено ядром | CPU оценщика, с/вылет |", "|---|---|---|---|---|---|---|"]
    for name, rs in results.items():
        ate = np.array([r.ate for r in rs])
        mx = np.array([r.max_err for r in rs])
        fin = np.array([r.final_err for r in rs])
        rej = sum(r.rejected for r in rs) / max(1, sum(r.rejected + r.accepted for r in rs))
        pct = np.mean([r.final_err / r.path for r in rs])
        cpu = np.mean([r.cpu_s for r in rs])
        lines.append(f"| {name} | {ate.mean():.3f} ± {ate.std():.3f} | {mx.mean():.3f} | {fin.mean():.3f} | "
                     f"{pct:.2%} | {rej:.1%} | {cpu:.2f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", help="JSON раскладки (по умолчанию — демо-склад 2 прохода × 10 секций)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--plot", action="store_true", help="сохранить error_vs_time.png")
    ap.add_argument("--scale-err", type=float, default=SensorModel.scale_err, help="СКО ошибки масштаба VIO")
    ap.add_argument("--yaw-bias", type=float, default=SensorModel.yaw_bias_deg_min, help="СКО дрейфа курса, град/мин")
    ap.add_argument("--window", action="store_true", help="добавить оптимизацию в скользящем окне (гл. 9)")
    ap.add_argument("--adapt", action="store_true", help="добавить EKF с инфляцией ковариации (LOC-12)")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    wh = Warehouse.load(a.layout) if a.layout else make_grid_warehouse()
    sm = SensorModel(scale_err=a.scale_err, yaw_bias_deg_min=a.yaw_bias)
    results, trajs = experiment(wh, a.seeds, sm=sm, window=a.window, adapt=a.adapt)
    table = summarize(results)
    print(table)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        t = np.arange(len(results["odom"][0].errors)) * 0.1
        for name in results:
            ax.plot(t, results[name][0].errors, label=name, lw=1)
        ax.set_xlabel("время, с (вылет 1, seed 0)")
        ax.set_ylabel("ошибка положения, м")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        out = Path(__file__).with_name("error_vs_time.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"график: {out}")


if __name__ == "__main__":
    main()
