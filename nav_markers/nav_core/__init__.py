"""Навигация БПЛА на складе по статическим реперным маркерам (ArUco).

Ядро без ROS: карта меток, поза по метке (PnP), выбор метки при повторяющихся ID, маршрут по графу проходов.
Новая тема ВКР — см. документ «Изменение вектора развития» (разд. 4–8). Модели склада и камеры
переиспользуются из invent_core (только импорт, файлы INVENT_DRONE не меняются).
"""
import sys
from pathlib import Path

_CORE = Path(__file__).resolve().parents[2] / "core"
if _CORE.is_dir() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
