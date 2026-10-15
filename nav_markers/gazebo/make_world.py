"""Мир Gazebo для навигации по ArUco-меткам = склад из sim/generate_world.py + метки + пол с текстурой.

    python3 nav_markers/gazebo/make_world.py core/examples/warehouse_demo.json -o nav_markers/runs/world \
        --heights 0.5 1.5 2.3 --step-bays 2

Результат в каталоге -o:
  warehouse_nav.sdf   мир (стеллажи, паллеты и этикетки — как в INVENT_DRONE; метки; пол; плагин потока PX4)
  marker_map.json     карта меток «ID -> поза» (nav_core.markers) — то, что на реальном складе меряют рулеткой
  markers/*.png       текстуры меток, floor.png — текстура пола

Отличия от sim/generate_world.py (сам файл не меняется — он импортируется):
* метки — плоскости со стороной size·(1 + 2·border) (белое поле вокруг чёрного квадрата), кадр как у OpenCV;
* пол — плитки с «бетонной» текстурой: однотонный пол не даёт оптическому потоку ни одной точки
  (тот же довод, что у преподавателя про гладкий бетон; --floor-texture-mm 0 — проверить это);
* плагин custom::OpticalFlowSystem PX4: в server.config он есть, но мир со своим списком плагинов
  его отключает (Gazebo берёт server.config, только если в мире плагинов нет).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nav_markers"))
import nav_core  # noqa: E402,F401
from invent_core.layout import Warehouse  # noqa: E402
from nav_core.markers import MarkerMap, PlacementParams, marker_rotation, min_repeat_distance, place_markers, render_marker  # noqa: E402
from sim.generate_world import Options, build  # noqa: E402

WORLD = "warehouse_nav"
BORDER = 0.25         # белое поле вокруг метки, доля стороны (как на распечатке A4 с полями)
FLOW_PLUGIN = '    <plugin filename="libOpticalFlowSystem.so" name="custom::OpticalFlowSystem"/>\n'


def rpy_from_matrix(r) -> tuple[float, float, float]:
    """R = Rz(yaw)·Ry(pitch)·Rx(roll) -> (roll, pitch, yaw) для <pose> SDF."""
    pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
    roll = math.atan2(r[2][1], r[2][2])
    yaw = math.atan2(r[1][0], r[0][0])
    return roll, pitch, yaw


def marker_link_pose(m) -> str:
    """Поза звена метки: оси звена = оси метки (x вправо, y вверх, z — нормаль). Плоскость SDF с нормалью +z
    кладёт текстуру так, что верх картинки — по +y звена (проверено рендером в Gazebo Harmonic,
    nav_markers/gazebo/check_marker_texture.py)."""
    roll, pitch, yaw = rpy_from_matrix(marker_rotation(m.normal))
    return f"{m.x:.4f} {m.y:.4f} {m.z:.4f} {roll:.6f} {pitch:.6f} {yaw:.6f}"


def marker_models(mm: MarkerMap, tex_dir: Path) -> str:
    side = mm.size * (1 + 2 * BORDER)
    out = []
    for i, m in enumerate(mm.markers):
        tex = (tex_dir / f"aruco_{m.id}.png").resolve().as_posix()
        out.append(f"""    <model name="marker_{i}_{m.kind}_{m.id}"><static>true</static>
      <link name="link"><pose>{marker_link_pose(m)}</pose>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>{side:.4f} {side:.4f}</size></plane></geometry>
          <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>
            <pbr><metal><albedo_map>{tex}</albedo_map><roughness>1.0</roughness><metalness>0.0</metalness></metal></pbr>
          </material></visual>
      </link></model>
""")
    return "".join(out)


def floor_model(wh: Warehouse, tex: Path | None, tile: float = 4.0, margin: float = 4.0) -> str:
    """Пол: большая плоскость столкновений + плитки tile×tile с текстурой над зоной склада."""
    xs = [r.origin_x for r in wh.rows] + [r.end_x for r in wh.rows] + [wh.cross_aisle_x, wh.dock[0]]
    ys = [r.face_y for r in wh.rows] + [a.y for a in wh.aisles] + [wh.dock[1]]
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    visuals = []
    if tex is not None:
        t = tex.resolve().as_posix()
        nx, ny = math.ceil((x1 - x0) / tile), math.ceil((y1 - y0) / tile)
        for i in range(nx):
            for j in range(ny):
                cx, cy = x0 + (i + 0.5) * tile, y0 + (j + 0.5) * tile
                visuals.append(f'<visual name="t{i}_{j}"><pose>{cx:.3f} {cy:.3f} 0.001 0 0 0</pose>'
                               f'<geometry><plane><normal>0 0 1</normal><size>{tile} {tile}</size></plane></geometry>'
                               f'<material><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>'
                               f'<pbr><metal><albedo_map>{t}</albedo_map><roughness>1.0</roughness></metal></pbr>'
                               f'</material></visual>')
    return ('    <model name="floor"><static>true</static><link name="link">\n'
            '      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry></collision>\n'
            '      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>\n'
            '        <material><diffuse>0.55 0.55 0.5 1</diffuse></material></visual>\n      '
            + "\n      ".join(visuals) + "\n    </link></model>\n")


def floor_texture(path: Path, tile: float, mm_per_px: float, seed: int = 0) -> None:
    """«Бетон»: шум нескольких масштабов + редкие тёмные пятна. Размер пятен ~1–10 см — на высоте 1.5 м
    камера потока (HFOV 42°, PX4 x500_flow) видит их десятками."""
    import cv2
    import numpy as np
    n = int(round(tile * 1000 / mm_per_px))
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n), np.float32)
    for scale, amp in [(1, 0.25), (4, 0.35), (16, 0.3), (64, 0.25)]:
        k = max(2, n // scale // 8)
        small = rng.random((k, k)).astype(np.float32)
        img += amp * cv2.resize(small, (n, n), interpolation=cv2.INTER_CUBIC)
    img += 0.15 * rng.random((n, n)).astype(np.float32)
    for _ in range(int(tile * tile * 15)):
        x, y = rng.integers(0, n, 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, max(3, n // 80))), float(rng.uniform(-0.6, -0.2)), -1)
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    cv2.imwrite(str(path), (60 + 150 * img).astype(np.uint8))


def build_world(wh: Warehouse, pp: PlacementParams, out: Path, labels: bool, floor_mm: float, ambient: float,
                seed: int):
    mm = place_markers(wh, pp)
    tex_dir = out / "markers"
    tex_dir.mkdir(parents=True, exist_ok=True)
    import cv2
    for mid in sorted({m.id for m in mm.markers}):
        cv2.imwrite(str(tex_dir / f"aruco_{mid}.png"), render_marker(mid, 600, mm.dictionary, BORDER))
    tex = None
    if floor_mm > 0:
        tex = out / "floor.png"
        floor_texture(tex, 4.0, floor_mm, seed)
    (out / "labels").mkdir(exist_ok=True)
    sdf, *_ = build(wh, Options(ambient=ambient, seed=seed), out if labels else None)
    head, sep, rest = sdf.partition('    <model name="floor">')
    _, _, rest = rest.partition("</link></model>\n")       # старый однотонный пол
    sdf = head + floor_model(wh, tex) + rest
    anchor = "</plugin>\n    <spherical_coordinates>"
    if anchor not in sdf:
        raise RuntimeError("sim/generate_world.py изменился: не найдено место для плагина потока")
    sdf = sdf.replace(anchor, "</plugin>\n" + FLOW_PLUGIN + "    <spherical_coordinates>", 1)
    sdf = sdf.replace("  </world>", marker_models(mm, tex_dir) + "  </world>", 1)
    # PX4 (gz_bridge) ищет мир по имени из PX4_GZ_WORLD = имя файла без .sdf
    sdf = sdf.replace('<world name="warehouse">', f'<world name="{WORLD}">', 1)
    # Без теней: карта теней строится на каждый кадр каждой камеры, а камер здесь четыре (3 метки + поток 50 Гц).
    # С тенями и программным рендером WSL симуляция шла в 0.02 реального времени (прогон 12.10.2026).
    sdf = sdf.replace("<shadows>true</shadows>", "<shadows>false</shadows>", 1)
    sdf = sdf.replace("<cast_shadows>true</cast_shadows>", "<cast_shadows>false</cast_shadows>", 1)
    return sdf, mm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout")
    ap.add_argument("-o", "--out", default="nav_markers/runs/world")
    ap.add_argument("--size", type=float, default=PlacementParams.size, help="сторона метки, м")
    ap.add_argument("--dictionary", default=PlacementParams.dictionary)
    ap.add_argument("--step-bays", type=int, default=PlacementParams.step_bays, help="шаг путевых меток, секций")
    ap.add_argument("--heights", type=float, nargs="+", default=[1.5], help="высоты рядов путевых меток, м")
    ap.add_argument("--anchor-z", type=float, default=PlacementParams.anchor_z)
    ap.add_argument("--no-anchors", action="store_true")
    ap.add_argument("--labels", action="store_true", help="текстуры этикеток паллет (нужен python-barcode)")
    ap.add_argument("--floor-texture-mm", type=float, default=4.0, help="мм на пиксель текстуры пола; 0 — однотонный")
    ap.add_argument("--ambient", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wh = Warehouse.load(a.layout)
    pp = PlacementParams(size=a.size, dictionary=a.dictionary, step_bays=a.step_bays, heights=a.heights,
                         anchor_z=a.anchor_z, anchors=not a.no_anchors)
    sdf, mm = build_world(wh, pp, out, a.labels, a.floor_texture_mm, a.ambient, a.seed)
    (out / f"{WORLD}.sdf").write_text(sdf, encoding="utf-8")
    mm.save(out / "marker_map.json")
    n_anchor = len(mm.anchors())
    print(f"{out / (WORLD + '.sdf')}: меток {len(mm.markers)} (якорей {n_anchor}, путевых {len(mm.markers) - n_anchor}), "
          f"словарь {mm.dictionary}, сторона {mm.size * 100:.0f} см, повтор ID не ближе {min_repeat_distance(mm):.1f} м")


if __name__ == "__main__":
    main()
