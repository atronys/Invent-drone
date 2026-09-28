"""Оптимизация в скользящем окне (Gao & Zhang, гл. 9.1–9.2) для локализации по ориентирам с известными координатами.

Переменные — позы ключевых кадров x_i = (x, y, ψ) в окне из N последних кадров. Факторы:
* одометрия между соседними кадрами: e = [R_iᵀ(p_j − p_i), ψ_j − ψ_i] − Δ_изм (аналог ф. 9.4 на плоскости);
* наблюдение ориентира L в кадре j: e = R_jᵀ(L − p_j) − z (как в EKF, гл. 8.1.3);
* априорное условие на самый старый кадр — результат маргинализации (гл. 9.1.2, дополнение Шура).

Отличия от EKF, о которых говорит гл. 8.1.4: (1) каждая итерация Гаусса–Ньютона заново линеаризует
все факторы окна, а EKF линеаризует один раз; (2) выбросы гасятся робастным ядром Хьюбера (гл. 8.2.6,
ф. 8.44) через веса IRLS, а не отбрасываются жёстким порогом.

Ориентиры — не переменные (карта известна), поэтому при маргинализации старого кадра «заполнения»
матрицы между ориентирами (рис. 9.2) не возникает: остаётся только априор 3×3 на следующий кадр.
"""
from __future__ import annotations

import math

import numpy as np


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def compose(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """x ⊕ u: приращение u = (dx, dy, dψ) в кадре позы x."""
    c, s = math.cos(x[2]), math.sin(x[2])
    return np.array([x[0] + c * u[0] - s * u[1], x[1] + s * u[0] + c * u[1], wrap(x[2] + u[2])])


def _rt(psi: float) -> tuple[np.ndarray, np.ndarray]:
    """Rᵀ(ψ) и d(Rᵀ)/dψ для плоского поворота."""
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, s], [-s, c]]), np.array([[-s, c], [-c, -s]])


class WindowSmoother:
    def __init__(self, x0, P0, size: int = 25, huber: float | None = 2.45, iters: int = 3,
                 gate: float | None = None):
        self.X = [np.asarray(x0, dtype=float).copy()]
        self.prior_mu, self.prior_info = self.X[0].copy(), np.linalg.inv(P0)
        self.odo: list[tuple[np.ndarray, np.ndarray]] = []     # (Δ_изм, информация) между X[i] и X[i+1]
        self.obs: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[]]
        self.size, self.huber, self.iters, self.gate = size, huber, iters, gate
        self.downweighted = self.total_obs = 0

    # --- добавление данных -------------------------------------------------------------
    def add_keyframe(self, rel: np.ndarray, cov: np.ndarray) -> None:
        self.X.append(compose(self.X[-1], rel))
        self.odo.append((np.asarray(rel, dtype=float), np.linalg.inv(cov)))
        self.obs.append([])

    def add_obs(self, z, landmark, cov) -> None:
        self.obs[-1].append((np.asarray(z, dtype=float), np.asarray(landmark, dtype=float), np.linalg.inv(cov)))

    # --- факторы: (индексы, e, J-блоки, информация) ------------------------------------------
    def _odo_factor(self, i: int):
        xi, xj = self.X[i], self.X[i + 1]
        rt, drt = _rt(xi[2])
        d = xj[:2] - xi[:2]
        meas, info = self.odo[i]
        e = np.concatenate([rt @ d, [xj[2] - xi[2]]]) - meas
        e[2] = wrap(e[2])
        ji = np.zeros((3, 3))
        ji[:2, :2], ji[:2, 2], ji[2, 2] = -rt, drt @ d, -1.0
        jj = np.zeros((3, 3))
        jj[:2, :2], jj[2, 2] = rt, 1.0
        return e, ji, jj, info

    def _obs_factor(self, j: int, z, landmark):
        xj = self.X[j]
        rt, drt = _rt(xj[2])
        d = landmark - xj[:2]
        e = rt @ d - z
        jac = np.hstack([-rt, (drt @ d)[:, None]])
        return e, jac

    def _weight(self, e, info) -> float:
        r2 = max(float(e @ info @ e), 0.0)
        if self.gate is not None and r2 > self.gate:
            return 0.0                                             # жёсткий порог χ², как у EKF
        if self.huber is None:
            return 1.0
        r = math.sqrt(r2)
        return 1.0 if r <= self.huber else self.huber / r        # IRLS-вес ядра Хьюбера (ф. 8.44)

    # --- Гаусс–Ньютон по окну -------------------------------------------------------------
    def _system(self, count: bool = False):
        n = len(self.X)
        H, b = np.zeros((3 * n, 3 * n)), np.zeros(3 * n)
        e = self.X[0] - self.prior_mu
        e[2] = wrap(e[2])
        H[:3, :3] += self.prior_info
        b[:3] += self.prior_info @ e
        for i in range(n - 1):
            e, ji, jj, info = self._odo_factor(i)
            sl_i, sl_j = slice(3 * i, 3 * i + 3), slice(3 * i + 3, 3 * i + 6)
            H[sl_i, sl_i] += ji.T @ info @ ji
            H[sl_i, sl_j] += ji.T @ info @ jj
            H[sl_j, sl_i] += jj.T @ info @ ji
            H[sl_j, sl_j] += jj.T @ info @ jj
            b[sl_i] += ji.T @ info @ e
            b[sl_j] += jj.T @ info @ e
        for j in range(n):
            sl = slice(3 * j, 3 * j + 3)
            for z, lm, info in self.obs[j]:
                e, jac = self._obs_factor(j, z, lm)
                w = self._weight(e, info)
                if count:
                    self.total_obs += 1
                    self.downweighted += w < 1.0
                H[sl, sl] += w * jac.T @ info @ jac
                b[sl] += w * jac.T @ info @ e
        return H, b

    def optimize(self) -> None:
        for it in range(self.iters):
            H, b = self._system(count=(it == self.iters - 1))
            dx = np.linalg.solve(H, -b)
            for k in range(len(self.X)):
                self.X[k] = self.X[k] + dx[3 * k:3 * k + 3]
                self.X[k][2] = wrap(self.X[k][2])
        if len(self.X) > self.size:
            self._marginalize_oldest()

    def _marginalize_oldest(self) -> None:
        """Дополнение Шура по X0 (гл. 9.1.2): факторы X0 -> априор на X1; X0 выбрасывается."""
        e0 = self.X[0] - self.prior_mu
        e0[2] = wrap(e0[2])
        H = np.zeros((6, 6))
        b = np.zeros(6)
        H[:3, :3] += self.prior_info
        b[:3] += self.prior_info @ e0
        e, ji, jj, info = self._odo_factor(0)
        J = np.hstack([ji, jj])
        H += J.T @ info @ J
        b += J.T @ info @ e
        for z, lm, info_o in self.obs[0]:
            e, jac = self._obs_factor(0, z, lm)
            w = self._weight(e, info_o)
            H[:3, :3] += w * jac.T @ info_o @ jac
            b[:3] += w * jac.T @ info_o @ e
        A, B, C = H[:3, :3], H[:3, 3:], H[3:, 3:]
        A_inv = np.linalg.inv(A)
        H1 = C - B.T @ A_inv @ B
        b1 = b[3:] - B.T @ A_inv @ b[:3]
        self.prior_info = H1
        self.prior_mu = self.X[1] - np.linalg.solve(H1, b1)   # минимум ½(x−μ)ᵀH1(x−μ) при градиенте b1 в X1
        self.prior_mu[2] = wrap(self.prior_mu[2])
        self.X.pop(0)
        self.odo.pop(0)
        self.obs.pop(0)

    @property
    def latest(self) -> np.ndarray:
        return self.X[-1]
