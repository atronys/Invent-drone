"""Генератор мира Gazebo (Harmonic, SDF 1.9) из раскладки склада + эталонные данные.

    python sim/generate_world.py core/examples/warehouse_demo.json -o sim/worlds/demo

Результат в каталоге -o:
  warehouse.sdf       мир для PX4 SITL (стеллажи, паллеты, этикетки)
  truth.csv           что реально стоит в ячейках (истина для метрик)
  wms_expected.csv    что «думает» WMS — истина + внесённые ошибки (для теста сверки)
  labels/*.png        текстуры этикеток SSCC (если установлены python-barcode и Pillow)

Этикетка — A6 (148x105 мм), Code128 с данными "00"+SSCC; ширина модуля --module-mm
(реальные SSCC-этикетки: ~0.5–0.8 мм). Так симуляция честно проверяет оптику.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from invent_core.barcode import gs1_check_digit  # noqa: E402
from invent_core.layout import Warehouse  # noqa: E402

LABEL_W, LABEL_H = 0.148, 0.105
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
    <scene><grid>false</grid><ambient>0.5 0.5 0.5 1</ambient><background>0.7 0.7 0.7 1</background><shadows>true</shadows></scene>
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
      <diffuse>0.8 0.8 0.8 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.3 0.2 -0.9</direction>
    </light>
    <model name="floor"><static>true</static><link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
        <material><diffuse>0.55 0.55 0.5 1</diffuse></material></visual>
    </link></model>
"""


def box(name: str, pose, size, rgba="0.2 0.3 0.7 1", collide=True, texture: Path | None = None) -> str:
    p = " ".join(f"{v:.4f}" for v in pose) + " 0 0 0"
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


def render_label(sscc: str, path: Path, module_mm: float) -> bool:
    try:
        import barcode
        from barcode.writer import ImageWriter
        from PIL import Image
    except ImportError:
        return False
    px_mm = 10  # 254 dpi
    img = barcode.get("code128", "00" + sscc, writer=ImageWriter()).render(
        {"module_width": module_mm, "module_height": 40, "quiet_zone": 6, "dpi": 254,
         "font_size": 12, "text_distance": 6, "write_text": True})
    canvas = Image.new("RGB", (int(LABEL_W * 1000 * px_mm), int(LABEL_H * 1000 * px_mm)), "white")
    img.thumbnail(canvas.size)  # не увеличивает; уменьшит, только если код не влез в A6
    canvas.paste(img, ((canvas.width - img.width) // 2, (canvas.height - img.height) // 2))
    canvas.save(path)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout")
    ap.add_argument("-o", "--out", default="sim/worlds/demo")
    ap.add_argument("--occupancy", type=float, default=0.8, help="доля занятых ячеек")
    ap.add_argument("--wms-errors", type=float, default=0.03, help="доля ошибок учёта в WMS")
    ap.add_argument("--module-mm", type=float, default=0.5)
    ap.add_argument("--rack-depth", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rng = random.Random(a.seed)
    wh = Warehouse.load(a.layout)
    out = Path(a.out).resolve()
    (out / "labels").mkdir(parents=True, exist_ok=True)

    truth = {loc.address: (make_sscc(rng) if rng.random() < a.occupancy else None) for loc in wh.locations()}
    textures = True
    parts = [HEADER.format(name="warehouse")]
    for row in wh.rows:
        n, d, top = row.normal_y, a.rack_depth, row.level_z[-1] + 1.5
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
            code = truth[loc.address]
            if not code:
                continue
            lz = row.level_z[loc.level - 1]
            h = min(1.2, (row.level_z[loc.level] - lz - 0.25) if loc.level < len(row.level_z) else 1.2)
            y_front = row.face_y - n * 0.05
            v.append(box(f"pal_{loc.address}", (loc.x, y_front - n * PALLET_D / 2, lz + h / 2),
                         (PALLET_W, PALLET_D, h), rgba="0.6 0.45 0.3 1"))
            tex = out / "labels" / f"{loc.address}.png"
            textures = textures and render_label(code, tex, a.module_mm)
            v.append(box(f"lbl_{loc.address}", (loc.x, y_front + n * 0.002, loc.z), (LABEL_W, 0.002, LABEL_H),
                         rgba="1 1 1 1", collide=False, texture=tex if textures else None))
        parts.append(f'    <model name="row_{row.id}"><static>true</static><link name="link">\n'
                     + "".join(v) + "    </link></model>\n")
    parts.append("  </world>\n</sdf>\n")
    (out / "warehouse.sdf").write_text("".join(parts), encoding="utf-8")

    # WMS = истина + ошибки учёта: пересорт, «потерянная» паллета, фантом, неучтённая паллета
    wms = dict(truth)
    addrs = sorted(wms)
    for _ in range(int(len(addrs) * a.wms_errors)):
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
    for name, data in (("truth.csv", truth), ("wms_expected.csv", wms)):
        with open(out / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["address", "code"])
            w.writerows((k, v or "") for k, v in sorted(data.items()))

    diff = sum(truth[k] != wms[k] for k in truth)
    print(f"{out / 'warehouse.sdf'}: ячеек {len(truth)}, занято {sum(map(bool, truth.values()))}, "
          f"расхождений WMS/истина {diff}")
    if not textures:
        print("Текстуры этикеток не созданы: pip install python-barcode pillow")


if __name__ == "__main__":
    main()
