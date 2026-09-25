"""Генератор мира Gazebo (Harmonic, SDF 1.9) из раскладки склада + эталонные данные.

    python sim/generate_world.py core/examples/warehouse_demo.json -o sim/worlds/demo

Результат в каталоге -o:
  warehouse.sdf        мир для PX4 SITL (стеллажи, паллеты, этикетки паллет и адресов)
  truth.csv            что реально стоит в ячейках (истина для метрик)
  wms_expected.csv     что «думает» WMS — истина + внесённые ошибки (для теста сверки)
  address_labels.csv   где висят адресные этикетки (карта ориентиров для LOC-08)
  labels/*.png         текстуры (если установлены python-barcode и Pillow)

Этикетка паллеты — A6 (148x105 мм), Code128 "00"+SSCC; адресная — 100x50 мм на балке, Code128 адреса.
Ширина модуля --module-mm (реальные SSCC: ~0.5–0.8 мм) — симуляция честно проверяет оптику.
Сценарии (SIM-06): --rotate-max (наклон этикеток), --damaged (доля повреждённых), --obstacle
(препятствие в первом проходе, для SAFE-01), --ambient (освещённость сцены).
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from invent_core.barcode import gs1_check_digit  # noqa: E402
from invent_core.layout import Warehouse  # noqa: E402

PALLET_LABEL = (0.148, 0.105)
ADDRESS_LABEL = (0.100, 0.050)
PALLET_W, PALLET_D = 0.8, 1.0   # вдоль ряда, в глубину

HEADER = """<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{name}">
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene><grid>false</grid><ambient>{amb} {amb} {amb} 1</ambient><background>0.7 0.7 0.7 1</background><shadows>true</shadows></scene>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>
    <plugin filename="gz-sim-apply-link-wrench-system" name="gz::sim::systems::ApplyLinkWrench"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model><world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>55.7558</latitude_deg><longitude_deg>37.6173</longitude_deg><elevation>150</elevation>
    </spherical_coordinates>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows><pose>0 0 20 0 0 0</pose>
      <diffuse>{sun} {sun} {sun} 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.3 0.2 -0.9</direction>
    </light>
    <model name="floor"><static>true</static><link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
        <material><diffuse>0.55 0.55 0.5 1</diffuse></material></visual>
    </link></model>
"""


@dataclass
class Options:
    occupancy: float = 0.8
    wms_errors: float = 0.03
    module_mm: float = 0.5
    rack_depth: float = 1.1
    rotate_max: float = 0.0      # град: наклон этикетки в её плоскости (как наклеили)
    damaged: float = 0.0         # доля повреждённых паллетных этикеток
    obstacle: bool = False
    ambient: float = 0.5
    seed: int = 42


def box(name: str, pose, size, rgba="0.2 0.3 0.7 1", collide=True, texture: Path | None = None, rpy=(0, 0, 0)) -> str:
    p = " ".join(f"{v:.4f}" for v in (*pose, *rpy))
    s = " ".join(f"{v:.4f}" for v in size)
    mat = f"<diffuse>{rgba}</diffuse>"
    if texture:
        mat = f"<diffuse>1 1 1 1</diffuse><pbr><metal><albedo_map>{escape(texture.as_posix())}</albedo_map></metal></pbr>"
    col = f'<collision name="{name}_c"><pose>{p}</pose><geometry><box><size>{s}</size></box></geometry></collision>' if collide else ""
    return (f'{col}<visual name="{name}"><pose>{p}</pose><geometry><box><size>{s}</size></box></geometry>'
            f"<material>{mat}</material></visual>\n")


def make_sscc(rng: random.Random, prefix: str = "4600000") -> str:
    """SSCC: цифра расширения + префикс компании GS1 (460–469 — Россия) + серия + контрольная."""
    body = "0" + prefix + "".join(rng.choice("0123456789") for _ in range(16 - len(prefix)))
    return body + str(gs1_check_digit(body))


def render_label(data: str, path: Path, module_mm: float, size_m: tuple[float, float],
                 damage: random.Random | None = None) -> bool:
    """Текстура этикетки: Code128 с данными data на белом поле size_m; False — нет библиотек."""
    try:
        import barcode
        from barcode.writer import ImageWriter
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    px_mm = 10  # 254 dpi
    small = size_m[1] < 0.08
    img = barcode.get("code128", data, writer=ImageWriter()).render(
        {"module_width": module_mm, "module_height": 18 if small else 40, "quiet_zone": 4 if small else 6,
         "dpi": 254, "font_size": 8 if small else 12, "text_distance": 3 if small else 6, "write_text": True})
    canvas = Image.new("RGB", (int(size_m[0] * 1000 * px_mm), int(size_m[1] * 1000 * px_mm)), "white")
    img.thumbnail(canvas.size)  # не увеличивает; уменьшит, только если код не влез
    canvas.paste(img, ((canvas.width - img.width) // 2, (canvas.height - img.height) // 2))
    if damage is not None:  # SIM-06: закрыть часть кода (надрыв, скотч, грязь)
        w, h = canvas.size
        x0 = damage.randint(w // 5, w // 2)
        ImageDraw.Draw(canvas).rectangle([x0, h // 4, x0 + w // 4, 3 * h // 4], fill=(170, 170, 170))
    canvas.save(path)
    return True


def build(wh: Warehouse, o: Options, out: Path | None = None):
    """-> (sdf, truth, wms, address_labels). out=None — без текстур (для тестов)."""
    rng = random.Random(o.seed)
    truth = {loc.address: (make_sscc(rng) if rng.random() < o.occupancy else None) for loc in wh.locations()}
    textures = out is not None
    addr_labels = []
    parts = [HEADER.format(name="warehouse", amb=o.ambient, sun=min(1.0, o.ambient + 0.3))]
    for row in wh.rows:
        n, d, top = row.normal_y, o.rack_depth, row.level_z[-1] + 1.5
        back_y = row.face_y - n * d
        v = []
        for k in range(row.bays + 1):  # стойки
            x = row.origin_x + k * row.bay_width
            for y in (row.face_y - n * 0.04, back_y + n * 0.04):
                v.append(box(f"up_{k}_{y:.2f}", (x, y, top / 2), (0.08, 0.08, top)))
        for i, lz in enumerate(row.level_z[1:], start=2):  # балки (ярус 1 — пол)
            for y in (row.face_y - n * 0.03, back_y + n * 0.03):
                v.append(box(f"beam_{i}_{y:.2f}", ((row.origin_x + row.end_x) / 2, y, lz - 0.06),
                             (row.end_x - row.origin_x, 0.05, 0.12), rgba="0.9 0.45 0.1 1"))
        for loc in row.locations():
            lz = row.level_z[loc.level - 1]
            # адресная этикетка (SIM-07): на лицевой балке яруса, у пола — на отбойнике
            az = lz - 0.06 if loc.level > 1 else 0.10
            ay = row.face_y + n * 0.004
            addr_labels.append((loc.address, loc.x, row.face_y, az))
            tex = out / "labels" / f"addr_{loc.address}.png" if textures else None
            if tex is not None and not render_label(loc.address, tex, o.module_mm, ADDRESS_LABEL):
                textures, tex = False, None
            v.append(box(f"addr_{loc.address}", (loc.x, ay, az), (ADDRESS_LABEL[0], 0.002, ADDRESS_LABEL[1]),
                         rgba="1 1 0.8 1", collide=False, texture=tex))
            code = truth[loc.address]
            if not code:
                continue
            h = min(1.2, (row.level_z[loc.level] - lz - 0.25) if loc.level < len(row.level_z) else 1.2)
            y_front = row.face_y - n * 0.05
            v.append(box(f"pal_{loc.address}", (loc.x, y_front - n * PALLET_D / 2, lz + h / 2),
                         (PALLET_W, PALLET_D, h), rgba="0.6 0.45 0.3 1"))
            tex = out / "labels" / f"{loc.address}.png" if textures else None
            damaged = rng.random() < o.damaged
            if tex is not None and not render_label("00" + code, tex, o.module_mm, PALLET_LABEL,
                                                    random.Random(rng.random()) if damaged else None):
                textures, tex = False, None
            tilt = rng.uniform(-1, 1) * o.rotate_max * 3.14159265 / 180
            v.append(box(f"lbl_{loc.address}", (loc.x, y_front + n * 0.002, loc.z), (PALLET_LABEL[0], 0.002, PALLET_LABEL[1]),
                         rgba="1 1 1 1", collide=False, texture=tex, rpy=(0.0, tilt, 0.0)))
        parts.append(f'    <model name="row_{row.id}"><static>true</static><link name="link">\n'
                     + "".join(v) + "    </link></model>\n")
    if o.obstacle:  # препятствие в первом проходе (SAFE-01): «погрузчик» 1x1x1.5 м
        a = wh.aisles[0]
        x_mid = (wh.row(a.rows[0]).origin_x + wh.row(a.rows[0]).end_x) / 2
        parts.append(f'    <model name="obstacle"><static>true</static><link name="link">\n'
                     + box("forklift", (x_mid, a.y, 0.75), (1.0, 1.0, 1.5), rgba="0.9 0.8 0.1 1")
                     + "    </link></model>\n")
    parts.append("  </world>\n</sdf>\n")

    # WMS = истина + ошибки учёта: пересорт, «потерянная» паллета, фантом, неучтённая паллета
    wms = dict(truth)
    addrs = sorted(wms)
    for _ in range(int(len(addrs) * o.wms_errors)):
        kind = rng.choice(["swap", "moved", "phantom", "unrecorded"])
        x, y = rng.sample(addrs, 2)
        if kind == "swap":
            wms[x], wms[y] = wms[y], wms[x]
        elif kind == "moved" and wms[x] and not wms[y]:
            wms[x], wms[y] = None, wms[x]
        elif kind == "phantom" and not wms[x]:
            wms[x] = make_sscc(rng)
        elif kind == "unrecorded" and wms[x]:
            wms[x] = None
    return "".join(parts), truth, wms, addr_labels, textures


def write_csv(path: Path, header, rows) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout")
    ap.add_argument("-o", "--out", default="sim/worlds/demo")
    ap.add_argument("--occupancy", type=float, default=Options.occupancy, help="доля занятых ячеек")
    ap.add_argument("--wms-errors", type=float, default=Options.wms_errors, help="доля ошибок учёта в WMS")
    ap.add_argument("--module-mm", type=float, default=Options.module_mm)
    ap.add_argument("--rack-depth", type=float, default=Options.rack_depth)
    ap.add_argument("--rotate-max", type=float, default=0.0, help="макс. наклон этикеток, град")
    ap.add_argument("--damaged", type=float, default=0.0, help="доля повреждённых этикеток")
    ap.add_argument("--obstacle", action="store_true", help="препятствие в первом проходе")
    ap.add_argument("--ambient", type=float, default=0.5, help="освещённость сцены 0..1")
    ap.add_argument("--no-textures", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    o = Options(a.occupancy, a.wms_errors, a.module_mm, a.rack_depth, a.rotate_max, a.damaged, a.obstacle,
                a.ambient, a.seed)
    out = Path(a.out).resolve()
    (out / "labels").mkdir(parents=True, exist_ok=True)
    sdf, truth, wms, addr_labels, textures = build(Warehouse.load(a.layout), o, None if a.no_textures else out)
    (out / "warehouse.sdf").write_text(sdf, encoding="utf-8")
    write_csv(out / "truth.csv", ["address", "code"], ((k, v or "") for k, v in sorted(truth.items())))
    write_csv(out / "wms_expected.csv", ["address", "code"], ((k, v or "") for k, v in sorted(wms.items())))
    write_csv(out / "address_labels.csv", ["address", "x", "y", "z"],
              ((ad, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}") for ad, x, y, z in addr_labels))
    diff = sum(truth[k] != wms[k] for k in truth)
    print(f"{out / 'warehouse.sdf'}: ячеек {len(truth)}, занято {sum(map(bool, truth.values()))}, "
          f"расхождений WMS/истина {diff}, адресных этикеток {len(addr_labels)}")
    if not textures and not a.no_textures:
        print("Текстуры этикеток не созданы: pip install python-barcode pillow")


if __name__ == "__main__":
    main()
