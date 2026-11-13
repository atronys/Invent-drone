"""Чертежи и схемы сборки в SVG:  python docs/make_drawings.py

Размеры плат взяты из официальных чертежей производителя:
  Raspberry Pi 5 — RP-008347-DS-1 (плата 85 × 56, отверстия Ø2,7 по сетке 58 × 49, 3,5 от краёв;
                   оси разъёмов от левого края: USB-C 11,2; micro-HDMI 25,8 и 39,2);
  Camera Module 3 — RP-008153-DS-1 (плата 25 × 23,862; отверстия Ø2,2 по сетке 21 × 12,5,
                   2,0 от краёв; ось объектива 12,5 от бокового края и 14,4 от нижнего).
Размеры ESP32, GY-521, VL53L1X и линейки WS2812B у разных изготовителей различаются —
их отверстия размечаются по месту и на чертеже показаны условно.
"""
from __future__ import annotations

from pathlib import Path

FIG = Path(__file__).parent / "figures"
S = 4.0          # пикселей на миллиметр
FONT = 'font-family="Arial, sans-serif"'


class Svg:
    def __init__(self, w_mm: float, h_mm: float, pad: float = 22.0):
        self.w, self.h, self.pad = w_mm, h_mm, pad
        self.items: list[str] = []

    # координаты чертежа: начало — левый нижний угол детали, ось Y вверх
    def X(self, x): return (self.pad + x) * S
    def Y(self, y): return (self.pad + self.h - y) * S

    def rect(self, x, y, w, h, dash=False, fill="none", sw=1.4, rx=0):
        d = ' stroke-dasharray="6 4"' if dash else ""
        self.items.append(f'<rect x="{self.X(x):.1f}" y="{self.Y(y + h):.1f}" width="{w * S:.1f}" height="{h * S:.1f}" '
                          f'rx="{rx * S:.1f}" fill="{fill}" stroke="#000" stroke-width="{sw}"{d}/>')

    def circle(self, x, y, d, cross=True, fill="none"):
        self.items.append(f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{d / 2 * S:.1f}" fill="{fill}" '
                          f'stroke="#000" stroke-width="1.4"/>')
        if cross:
            k = max(d, 3) * 0.9
            self.line(x - k, y, x + k, y, dash="8 3 2 3", sw=0.7)
            self.line(x, y - k, x, y + k, dash="8 3 2 3", sw=0.7)

    def line(self, x0, y0, x1, y1, dash=None, sw=1.0, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        a = ' marker-end="url(#arr)" marker-start="url(#arr0)"' if arrow else ""
        self.items.append(f'<line x1="{self.X(x0):.1f}" y1="{self.Y(y0):.1f}" x2="{self.X(x1):.1f}" '
                          f'y2="{self.Y(y1):.1f}" stroke="#000" stroke-width="{sw}"{d}{a}/>')

    def text(self, x, y, s, size=12, anchor="middle", bold=False, rot=0):
        w = ' font-weight="bold"' if bold else ""
        r = f' transform="rotate({rot} {self.X(x):.1f} {self.Y(y):.1f})"' if rot else ""
        self.items.append(f'<text x="{self.X(x):.1f}" y="{self.Y(y):.1f}" font-size="{size}" text-anchor="{anchor}"{w}{r}>{s}</text>')

    def dim_h(self, x0, x1, y, label=None, off_text=1.2):
        """Горизонтальный размер на высоте y (мм)."""
        self.line(x0, y, x1, y, sw=0.8, arrow=True)
        self.text((x0 + x1) / 2, y + off_text, label or f"{abs(x1 - x0):g}".replace(".", ","), 11)

    def dim_v(self, y0, y1, x, label=None):
        self.line(x, y0, x, y1, sw=0.8, arrow=True)
        self.text(x - 1.2, (y0 + y1) / 2, label or f"{abs(y1 - y0):g}".replace(".", ","), 11, rot=-90)

    def ext(self, x0, y0, x1, y1):
        self.line(x0, y0, x1, y1, sw=0.5)

    def save(self, name: str, extra_h: float = 0.0) -> None:
        W, H = (self.w + 2 * self.pad) * S, (self.h + 2 * self.pad + extra_h) * S
        defs = ('<defs><marker id="arr" markerWidth="9" markerHeight="7" refX="8.5" refY="3.5" orient="auto">'
                '<path d="M0,0.8 L9,3.5 L0,6.2 z"/></marker><marker id="arr0" markerWidth="9" markerHeight="7" '
                'refX="0.5" refY="3.5" orient="auto"><path d="M9,0.8 L0,3.5 L9,6.2 z"/></marker></defs>')
        (FIG / name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
                                f'width="{W:.0f}" height="{H:.0f}" {FONT}>{defs}{"".join(self.items)}</svg>\n',
                                encoding="utf-8", newline="\n")


# --- размеры ----------------------------------------------------------------------
PLATE_W, PLATE_H = 160.0, 110.0
LENS = (80.0, 78.0)                                        # ось объектива на пластине
CAM0 = (LENS[0] - 12.5, LENS[1] - 14.4)                    # левый нижний угол платы камеры
CAM_HOLES = [(CAM0[0] + 2.0 + dx, CAM0[1] + 2.0 + dy) for dy in (0, 12.5) for dx in (0, 21.0)]
PI_HOLES = [(8.0 + dx, 6.0 + dy) for dy in (0, 49.0) for dx in (0, 58.0)]
PI0 = (PI_HOLES[0][0] - 3.5, PI_HOLES[0][1] - 3.5)         # левый нижний угол платы Pi (вид со стороны камеры)
BRACKET = [(100.0, 6.0), (130.0, 6.0)]
TOF = (108.0, 78.0)
SLOT = (71.0, 57.5, 18.0, 3.0)


def mount_plate() -> None:
    d = Svg(PLATE_W, PLATE_H, pad=26)
    d.rect(0, 0, PLATE_W, PLATE_H, sw=2, rx=3)
    # плата Pi 5 с обратной стороны — штриховой контур
    d.rect(PI0[0], PI0[1], 85, 56, dash=True, sw=0.9)
    d.text(PI0[0] + 30, PI0[1] + 30, "A1 — Raspberry Pi 5 (с обратной стороны,", 10.5)
    d.text(PI0[0] + 30, PI0[1] + 26, "на стойках 10 мм)", 10.5)
    for x, y in PI_HOLES:
        d.circle(x, y, 2.7)
    # камера
    d.rect(CAM0[0], CAM0[1], 25, 23.862, sw=0.9)
    for x, y in CAM_HOLES:
        d.circle(x, y, 2.2)
    d.circle(LENS[0], LENS[1], 7.5)
    d.text(LENS[0], CAM0[1] + 34.5, "A5 — камера", 10.5)
    d.rect(*SLOT, sw=1.4)
    d.text(SLOT[0] + 9, SLOT[1] - 3.6, "паз под шлейф 18 × 3", 10)
    # дальномер, индикатор, подсветка, кнопка, ESP32, MPU
    d.rect(TOF[0] - 7, TOF[1] - 6.5, 14, 13, sw=0.9)
    d.circle(TOF[0], TOF[1], 4, cross=True)
    d.text(TOF[0], TOF[1] - 10, "A4", 10.5)
    d.rect(54, 96, 52, 10, sw=0.9)
    d.text(80, 99.7, "HL1 — индикатор", 10.5)
    d.rect(100, 14, 52, 28, sw=0.9)
    d.text(126, 29.5, "EL1 — подсветка", 10.5)
    d.text(126, 25, "(ниже оси объектива)", 9.5)
    d.circle(20, 84, 7, cross=True)
    d.text(20, 91.5, "SB1", 10.5)
    d.rect(122, 50, 30, 52, dash=True, sw=0.9)
    d.text(137, 72, "A2 — ESP32", 10.5, rot=-90)
    d.rect(30, 66, 22, 17, dash=True, sw=0.9)
    d.text(41, 73, "A3", 10.5)
    for x, y in BRACKET:
        d.circle(x, y, 4.3)
    d.text(115, 9.2, "под уголок штатива", 9.5)
    # размеры
    d.ext(0, 0, 0, -12); d.ext(PLATE_W, 0, PLATE_W, -12); d.dim_h(0, PLATE_W, -10)
    d.ext(0, 0, -14, 0); d.ext(0, PLATE_H, -14, PLATE_H); d.dim_v(0, PLATE_H, -12)
    d.ext(LENS[0], LENS[1], LENS[0], PLATE_H + 12); d.ext(0, PLATE_H, 0, PLATE_H + 12)
    d.dim_h(0, LENS[0], PLATE_H + 10)
    d.ext(LENS[0], LENS[1], PLATE_W + 14, LENS[1]); d.ext(PLATE_W, 0, PLATE_W + 14, 0)
    d.dim_v(0, LENS[1], PLATE_W + 12)
    d.ext(TOF[0], TOF[1], TOF[0], PLATE_H + 6); d.dim_h(LENS[0], TOF[0], PLATE_H + 4)
    d.ext(PI_HOLES[0][0], PI_HOLES[0][1], PI_HOLES[0][0], -6); d.ext(PI_HOLES[1][0], PI_HOLES[1][1], PI_HOLES[1][0], -6)
    d.dim_h(PI_HOLES[0][0], PI_HOLES[1][0], -4)
    d.ext(PI_HOLES[0][0], PI_HOLES[0][1], -8, PI_HOLES[0][1]); d.ext(PI_HOLES[2][0], PI_HOLES[2][1], -8, PI_HOLES[2][1])
    d.dim_v(PI_HOLES[0][1], PI_HOLES[2][1], -6)
    d.dim_h(CAM_HOLES[0][0], CAM_HOLES[1][0], CAM_HOLES[2][1] + 7.5, "21")
    d.ext(CAM_HOLES[1][0], CAM_HOLES[1][1], CAM_HOLES[1][0] + 5, CAM_HOLES[1][1]); d.ext(CAM_HOLES[3][0], CAM_HOLES[3][1], CAM_HOLES[3][0] + 5, CAM_HOLES[3][1])
    d.dim_v(CAM_HOLES[1][1], CAM_HOLES[3][1], CAM_HOLES[1][0] + 4.5, "12,5")
    d.text(PLATE_W / 2, -19, "Вид со стороны стеллажа. Материал: фанера, акрил или печатный пластик 3 мм. "
           "Размеры для справок, мм.", 11)
    d.save("mount_plate.svg")


def pi5_connections() -> None:
    d = Svg(85, 56, pad=48)
    d.rect(0, 0, 85, 56, sw=2, rx=3)
    for x, y in ((3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)):
        d.circle(x, y, 2.7, cross=False)
    d.rect(7, 50, 51, 5, sw=1)                                  # GPIO
    d.text(32.5, 51.3, "GPIO, 40 выводов (не используется)", 9)
    d.rect(11.2 - 4.5, -1.5, 9, 7.5, fill="#ddd"); d.rect(25.8 - 3.5, -1.5, 7, 7, sw=1); d.rect(39.2 - 3.5, -1.5, 7, 7, sw=1)
    d.rect(46.2, 3, 2.6, 12, fill="#ddd"); d.rect(52.2, 3, 2.6, 12, sw=1)
    d.rect(66.5, 2, 21.5, 16.5, sw=1)                           # Ethernet
    d.rect(70, 22.5, 18, 13.5, fill="#ddd"); d.rect(70, 40.3, 18, 13.5, sw=1)
    d.rect(60.5, 45.5, 5.5, 4, fill="#ddd")                     # разъём вентилятора
    d.rect(-1.5, 24, 3.5, 13, sw=1)                             # PCIe
    d.rect(22, 20, 22, 22, sw=0.8)
    d.text(33, 30, "процессор", 9); d.text(33, 26.5, "(под охладителем)", 8.5)

    def call(x, y, tx, ty, s, anchor="middle"):
        """Выноска: линия от точки платы к подписи; строки подписи идут вниз от (tx, ty)."""
        d.line(x, y, tx, ty + (1.0 if ty < y else -1.0), sw=0.7)
        parts = s.split("|")
        top = ty - 3.0 if ty < y else ty + 1.5 + 4.0 * (len(parts) - 1)
        for i, part in enumerate(parts):
            d.text(tx, top - i * 4.0, part, 10.5, anchor)

    call(11.2, -1, 2, -12, "1. Питание: блок G1,|USB-C 5,1 В / 5 А")
    call(47.5, 3, 52, -12, "2. Камера A5: шлейф,|CAM/DISP (любой из двух)")
    call(88, 29, 92, 31, "3. Сенсорная головка A2:|кабель USB (любой порт)", "start")
    call(88, 10, 92, 12, "4. Сеть: Ethernet|или Wi-Fi", "start")
    call(63, 49.5, 70, 62, "5. Вентилятор|охладителя")
    call(25.8, -1.5, 24, -27, "micro-HDMI — только для первичной|настройки с монитором")
    call(-1.5, 30, -22, 40, "PCIe —|не используется")
    d.text(20, 72, "Карта microSD — снизу платы,", 10.5)
    d.text(20, 68, "с края, противоположного USB", 10.5)
    d.text(42.5, -40, "Серым выделены разъёмы, занятые в стенде. Положение разъёмов CAM/DISP,", 10)
    d.text(42.5, -44, "Ethernet и USB — по чертежу платы, приблизительно.", 10)
    d.save("pi5_connections.svg", extra_h=4)


def assembly_steps() -> None:
    """Шесть шагов сборки одной полосой пиктограмм."""
    W, H, n = 760, 470, 6
    cw, ch = W / 3, H / 2
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" {FONT} font-size="11">']
    steps = [
        ("1. Камера на пластину", "4 винта М2 × 8 с гайками;|шлейф продеть в паз", "cam"),
        ("2. Дальномер и индикатор", "A4 — вплотную к объективу, окном|к стеллажу; HL1 — сверху", "tof"),
        ("3. Raspberry Pi сзади", "4 стойки М2,5 × 10; охладитель|установить до крепления", "pi"),
        ("4. ESP32 и MPU-6050 сзади", "A3 — жёстко и параллельно|плоскости камеры", "esp"),
        ("5. Провода по схеме Э4", "сначала I2C, затем кнопка,|индикатор, ключ подсветки", "wire"),
        ("6. Штатив и кабели", "уголок 1/4\"; шлейф — в Pi, USB —|в Pi, питание — последним", "tripod"),
    ]
    for i, (title, note, kind) in enumerate(steps):
        x0, y0 = (i % 3) * cw, (i // 3) * ch
        out.append(f'<rect x="{x0 + 6}" y="{y0 + 6}" width="{cw - 12}" height="{ch - 12}" fill="none" stroke="#000" stroke-width="1"/>')
        out.append(f'<text x="{x0 + 16}" y="{y0 + 26}" font-size="13" font-weight="bold">{title}</text>')
        for j, part in enumerate(note.split("|")):
            out.append(f'<text x="{x0 + 16}" y="{y0 + ch - 34 + j * 14}">{part}</text>')
        px, py, pw, ph = x0 + 52, y0 + 44, 150, 103         # пластина 160 × 110 в масштабе
        k = pw / 160
        g = [f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="3" fill="#f4f4f4" stroke="#000" stroke-width="1.5"/>']

        def r(x, y, w, h, fill="#fff", dash=False):
            d = ' stroke-dasharray="4 3"' if dash else ""
            return (f'<rect x="{px + x * k:.1f}" y="{py + ph - (y + h) * k:.1f}" width="{w * k:.1f}" height="{h * k:.1f}" '
                    f'fill="{fill}" stroke="#000" stroke-width="1"{d}/>')

        def c(x, y, dia, fill="#000"):
            return f'<circle cx="{px + x * k:.1f}" cy="{py + ph - y * k:.1f}" r="{dia / 2 * k:.1f}" fill="{fill}" stroke="#000"/>'

        cam = [r(CAM0[0], CAM0[1], 25, 23.9, "#bbb"), c(LENS[0], LENS[1], 8, "#333"), r(*SLOT, fill="#000")]
        front_extra = [r(TOF[0] - 7, TOF[1] - 6.5, 14, 13, "#bbb"), r(54, 96, 52, 10, "#bbb")]
        back = [r(160 - PI0[0] - 85, PI0[1], 85, 56, "#bbb"), r(160 - 152, 50, 30, 52, "#ddd", True), r(160 - 52, 66, 22, 17, "#ddd", True)]
        if kind == "cam":
            g += cam
        elif kind == "tof":
            g += cam + front_extra
        elif kind == "pi":
            g += [back[0]] + [c(160 - x, y, 4, "#fff") for x, y in PI_HOLES]
            g.append(f'<text x="{px + pw / 2}" y="{py - 4}" text-anchor="middle" font-size="10">вид сзади</text>')
        elif kind == "esp":
            g += [back[0], r(160 - 152, 50, 30, 52, "#bbb"), r(160 - 52, 66, 22, 17, "#bbb")]
            g.append(f'<text x="{px + pw / 2}" y="{py - 4}" text-anchor="middle" font-size="10">вид сзади</text>')
        elif kind == "wire":
            g += [back[0], r(160 - 152, 50, 30, 52, "#bbb"), r(160 - 52, 66, 22, 17, "#bbb")]
            for (xa, ya, xb, yb) in ((23, 76, 108, 75), (23, 70, 108, 71), (23, 62, 60, 96), (23, 56, 40, 20)):
                g.append(f'<path d="M{px + xa * k:.1f},{py + ph - ya * k:.1f} Q{px + (xa + xb) / 2 * k:.1f},'
                         f'{py + ph - (ya + yb) / 2 * k - 14:.1f} {px + xb * k:.1f},{py + ph - yb * k:.1f}" '
                         f'fill="none" stroke="#000" stroke-width="1.6"/>')
            g.append(f'<text x="{px + pw / 2}" y="{py - 4}" text-anchor="middle" font-size="10">вид сзади</text>')
        else:
            g += cam + front_extra + [r(100, 14, 52, 28, "#bbb")]
            bx = px + 115 * k
            g.append(f'<path d="M{bx - 14},{py + ph} L{bx + 14},{py + ph} L{bx + 14},{py + ph + 8} L{bx - 14},{py + ph + 8} z" fill="#888" stroke="#000"/>')
            g.append(f'<path d="M{bx},{py + ph + 8} L{bx - 22},{py + ph + 34} M{bx},{py + ph + 8} L{bx + 22},{py + ph + 34} '
                     f'M{bx},{py + ph + 8} L{bx},{py + ph + 34}" stroke="#000" stroke-width="2.5" fill="none"/>')
        out += g
    out.append("</svg>\n")
    (FIG / "assembly_steps.svg").write_text("".join(out), encoding="utf-8", newline="\n")


def hole_table() -> str:
    """Таблица координат отверстий (Markdown) — вставляется в руководство."""
    rows = ["| Отверстия | Назначение | Диаметр, мм | Координаты центров X; Y от левого нижнего угла, мм |", "|---|---|---|---|"]
    fmt = lambda pts: " · ".join(f"{x:g}; {y:g}".replace(".", ",") for x, y in pts)
    rows.append(f"| 1–4 | Плата камеры A5 (винты М2) | 2,2 | {fmt(CAM_HOLES)} |")
    rows.append(f"| 5–8 | Плата Raspberry Pi 5 (стойки М2,5) | 2,7 | {fmt(PI_HOLES)} |")
    rows.append(f"| 9, 10 | Уголок под штатив (винты М4) | 4,3 | {fmt(BRACKET)} |")
    rows.append(f"| — | Паз под шлейф камеры 18 × 3 | — | левый нижний угол {SLOT[0]:g}; {SLOT[1]:g}".replace(".", ",") + " |")
    rows.append(f"| — | Ось объектива (для справки) | — | {LENS[0]:g}; {LENS[1]:g} |")
    rows.append(f"| — | Ось дальномера A4 (отверстия — по модулю) | — | {TOF[0]:g}; {TOF[1]:g} |")
    return "\n".join(rows)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    FIG.mkdir(exist_ok=True)
    mount_plate()
    pi5_connections()
    assembly_steps()
    print(hole_table())
