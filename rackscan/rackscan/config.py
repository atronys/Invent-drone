"""Настройки стенда: один JSON-файл, всё остальное — значения по умолчанию.

Пароли в файл не пишутся: для 1С указываются имена переменных окружения.
"""
from __future__ import annotations

import copy
import json
from dataclasses import fields
from pathlib import Path

from .guidance import Tolerances
from .occupancy import OccupancyConfig
from .scanner import ScanConfig

DEFAULTS: dict = {
    "layout": None,                      # путь к JSON модели стенда; None — учебный стенд
    "camera": {
        "source": "synthetic",           # synthetic | picamera | opencv | files
        "preset": "imx708",              # см. optics.CAMERA_PRESETS
        "preview_size": [1536, 864],
        "still_size": None,              # None — полный кадр сенсора
        "calibration": None,             # путь к JSON калибровки (fx, fy, cx, cy, dist, width, height)
        "device": 0,                     # номер USB-камеры для source=opencv
        "path": None,                    # папка с кадрами для source=files
        "fps": 15,
        "lens_position": None,           # диоптрии (1/м) для ручного фокуса; None — автофокус
        "short_exposure": True,          # режим автоэкспозиции с короткими выдержками
    },
    "label": {"code_mm": 40.0, "module_mm": 1.905, "sticker_mm": 62.0, "format": "QRCode"},
    "scan": {},                          # любые поля ScanConfig, включая coarse/fine/occ словарями
    "sensors": {"enabled": False, "port": "/dev/ttyUSB0", "baud": 115200},
    "wms": {
        "backend": "emulator",           # emulator | onec_http
        "db": "data/wms.sqlite",
        "demo": True,                    # наполнить эмулятор учебным складом, если он пуст
        "url": "",                       # для onec_http: http://сервер/база/hs/rackscan/v1
        "user_env": "RACKSCAN_1C_USER",
        "password_env": "RACKSCAN_1C_PASSWORD",
    },
    "server": {"host": "0.0.0.0", "port": 8080, "jpeg_quality": 80, "stream_width": 1280},
    "evidence_dir": "data/evidence",
    "references_dir": "data/references",
    "seed": 7,
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        cfg = _merge(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def scan_config(cfg: dict) -> ScanConfig:
    """Собрать ScanConfig из разделов label и scan."""
    sc = ScanConfig()
    lab = cfg.get("label", {})
    sc.label_code_mm = float(lab.get("code_mm", sc.label_code_mm))
    sc.label_module_mm = float(lab.get("module_mm", sc.label_module_mm))
    sc.label_sticker_mm = float(lab.get("sticker_mm", sc.label_sticker_mm))
    names = {f.name for f in fields(ScanConfig)}
    for k, v in (cfg.get("scan") or {}).items():
        if k not in names:
            raise KeyError(f"в разделе scan нет параметра {k!r}")
        if k in ("coarse", "fine"):
            v = Tolerances(**v)
        elif k == "occ":
            v = OccupancyConfig(**{kk: (tuple(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()})
        setattr(sc, k, v)
    return sc
