# ruff: noqa: E501  (long SVG markup strings)
"""Generate src/factors.svg (then run ../build.py for the light and dark files).

python3 docs/assets/readme/src/factors_gen.py
"""

import math
import random
from pathlib import Path

OUT = Path(__file__).with_name("factors.svg")
H = 400
TOP, BOT = 10, 342
rng = random.Random(3)


def bumps(t: float, times: list[float], tau: float = 0.05) -> float:
    return sum(math.exp(-(t - t0) / tau) for t0 in times if t >= t0)


def trace(x0: float, x1: float, y0: float, amp: float, f, n: int = 64, noise: float = 0.0) -> str:
    pts = []
    for i in range(n):
        t = i / (n - 1)
        y = y0 - amp * f(t) + (rng.gauss(0, noise) if noise else 0.0)
        pts.append(f"{x0 + (x1 - x0) * t:.1f},{y:.1f}")
    return " ".join(pts)


CELL = [0.15, 0.47, 0.78]
CELL2 = [0.3, 0.64]


def cell(t: float) -> float:
    return min(bumps(t, CELL), 1.3)


def cell2(t: float) -> float:
    return min(bumps(t, CELL2), 1.3)


def drift(t: float) -> float:
    return 0.5 + 0.5 * math.sin(2 * math.pi * (t * 0.8 - 0.1))


def sub(name: str, k: int) -> str:
    return f'{name}<tspan font-size="15" dy="5">{k}</tspan>'


def rich(segs: list[tuple[str, bool]]) -> str:
    """Text with subscripts: (text, is_sub) pieces, baseline shift tracked."""
    out, low = [], False
    for txt, is_sub in segs:
        if is_sub and not low:
            out.append(f'<tspan font-size="15" dy="5">{txt}</tspan>')
        elif not is_sub and low:
            out.append(f'<tspan dy="-5">{txt}</tspan>')
        elif is_sub:
            out.append(f'<tspan font-size="15">{txt}</tspan>')
        else:
            out.append(txt)
        low = is_sub
    return "".join(out)


s: list[str] = []
add = s.append
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {H}" role="img">')
add("<title>What is a factor: a few shared time courses plus per-pixel weights</title>")
add(
    '<defs><marker id="arr" markerWidth="12" markerHeight="12" refX="11" refY="6" '
    'orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L12,6 L0,12 Z" fill="var(--muted)"/></marker></defs>'
)

# ---- panels + arrows
PANELS = [
    (10, 360, "A", "Every pixel, every frame"),
    (410, 360, "B", "Pixels move together"),
    (810, 380, "C", "A few time courses + weights"),
]
for x, w, letter, title in PANELS:
    add(
        f'<rect x="{x}" y="{TOP}" width="{w}" height="{BOT - TOP}" rx="10" fill="var(--panel)" stroke="var(--line)" stroke-width="2"/>'
    )
    add(
        f'<text x="{x + 18}" y="{TOP + 34}" font-size="20" font-weight="600" fill="var(--ink)">'
        f'<tspan fill="var(--acc)">{letter}</tspan>  {title}</text>'
    )
for xa in (372, 772):
    add(
        f'<line x1="{xa}" y1="176" x2="{xa + 34}" y2="176" stroke="var(--muted)" stroke-width="2.5" marker-end="url(#arr)"/>'
    )

# ---- A: the table
add('<text x="108" y="80" font-size="15" fill="var(--muted)">frame 1, 2, 3, … 36 000 →</text>')
rows = [("pixel 1", 1.0), ("pixel 2", 0.7), ("pixel 3", 0.45), (None, 0), ("pixel 4096", -1)]
cols = 12
y = 92
for label, amp in rows:
    if label is None:
        add(
            f'<text x="115" y="{y + 10}" font-size="16" fill="var(--muted)" text-anchor="middle">⋮</text>'
        )
        y += 18
        continue
    add(
        f'<text x="100" y="{y + 12}" font-size="15" fill="var(--ink)" text-anchor="end">{label}</text>'
    )
    for c in range(cols + 1):
        cx = 108 + c * 16 if c < cols else 326
        t = c / cols
        if amp > 0:
            v = 0.12 + 0.8 * amp * min(bumps(t, [0.15, 0.47, 0.78], 0.08), 1.0)
        else:
            v = 0.12 + 0.18 * rng.random()
        add(
            f'<rect x="{cx}" y="{y}" width="15" height="15" fill="var(--acc)" fill-opacity="{v:.2f}" stroke="var(--line)" stroke-width="1"/>'
        )
    add(
        f'<text x="314" y="{y + 12}" font-size="15" fill="var(--muted)" text-anchor="middle">…</text>'
    )
    y += 20
add(
    '<text x="30" y="240" font-size="16" fill="var(--ink)">one 64 × 64 tile, 30 min at 20 Hz:</text>'
)
add('<text x="30" y="266" font-size="16" fill="var(--ink)">4096 pixels × 36 000 frames</text>')
add(
    '<text x="30" y="300" font-size="22" font-weight="600" fill="var(--data)">= 147 million numbers</text>'
)

# ---- B: one cell + traces
# cell icon: a 4x4 patch of the pixel grid with one cell on it; its four pixels are numbered
GX, GY, PX = 426, 80, 18
for r in range(4):
    for c in range(4):
        inside = r in (1, 2) and c in (1, 2)
        fill, stroke = (
            ("var(--acc-soft)", "var(--acc)") if inside else ("var(--panel)", "var(--line)")
        )
        add(
            f'<rect x="{GX + c * PX}" y="{GY + r * PX}" width="{PX}" height="{PX}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
        )
add(
    f'<circle cx="{GX + 2 * PX}" cy="{GY + 2 * PX}" r="25" fill="none" stroke="var(--acc)" stroke-width="2.5"/>'
)
for k, (r, c) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2)], start=1):
    add(
        f'<text x="{GX + c * PX + PX / 2}" y="{GY + r * PX + 14}" font-size="15" font-weight="600" fill="var(--ink)" text-anchor="middle">{k}</text>'
    )
add(
    f'<text x="{GX + 2 * PX}" y="{GY + 4 * PX + 20}" font-size="15" fill="var(--muted)" text-anchor="middle">one cell,</text>'
)
add(
    f'<text x="{GX + 2 * PX}" y="{GY + 4 * PX + 38}" font-size="15" fill="var(--muted)" text-anchor="middle">its 4 pixels</text>'
)

TX0, TX1 = 536, 752
for t0 in CELL:
    xx = TX0 + (TX1 - TX0) * t0
    add(
        f'<line x1="{xx:.1f}" y1="70" x2="{xx:.1f}" y2="232" stroke="var(--line)" stroke-width="1.5" stroke-dasharray="4 4"/>'
    )
for k, (yb, amp) in enumerate([(106, 26), (146, 23), (186, 20), (226, 17)], start=1):
    add(
        f'<circle cx="{518}" cy="{yb - 6}" r="10" fill="var(--panel)" stroke="var(--ink)" stroke-width="1.5"/>'
    )
    add(
        f'<text x="518" y="{yb - 1}" font-size="15" fill="var(--ink)" text-anchor="middle">{k}</text>'
    )
    add(
        f'<polyline points="{trace(TX0, TX1, yb, amp, cell, noise=0.8)}" fill="none" stroke="var(--acc)" stroke-width="2"/>'
    )
add(
    '<text x="430" y="258" font-size="15" fill="var(--muted)">background: all pixels, slowly</text>'
)
add(
    f'<polyline points="{trace(TX0, TX1, 290, 18, drift)}" fill="none" stroke="var(--old)" stroke-width="2.5" stroke-dasharray="7 5"/>'
)
add(
    '<text x="430" y="316" font-size="16" fill="var(--ink)">the table repeats a few time courses,</text>'
)
add('<text x="430" y="336" font-size="16" fill="var(--ink)">plus noise</text>')

# ---- C: weights + time courses
add('<text x="830" y="76" font-size="15" fill="var(--muted)">weights</text>')
add('<text x="830" y="94" font-size="15" fill="var(--muted)">(per pixel)</text>')
add('<text x="950" y="76" font-size="15" fill="var(--muted)">time courses (per frame)</text>')


def wmap(kind: int, r: int, c: int) -> float:
    if kind == 1:
        return math.exp(-((r - 1.6) ** 2 + (c - 1.8) ** 2) / 1.6)
    if kind == 2:
        return 0.35 + 0.08 * (r + c)
    return math.exp(-((r - 4.0) ** 2 + (c - 3.6) ** 2) / 1.4)


courses = [(1, cell, "var(--acc)"), (2, drift, "var(--old)"), (3, cell2, "var(--acc)")]
for i, (k, f, col) in enumerate(courses):
    yc = 132 + i * 62  # row centre
    gx, gy = 840, yc - 24
    add(
        f'<rect x="{gx - 2}" y="{gy - 2}" width="52" height="52" rx="3" fill="var(--panel)" stroke="var(--line)" stroke-width="1"/>'
    )
    for r in range(6):
        for c in range(6):
            v = wmap(k, r, c)
            if v > 0.04:
                add(
                    f'<rect x="{gx + c * 8}" y="{gy + r * 8}" width="8" height="8" fill="var(--data)" fill-opacity="{min(v, 1):.2f}"/>'
                )
    add(f'<text x="904" y="{yc + 6}" font-size="20" fill="var(--ink)">{sub("w", k)}</text>')
    add(f'<text x="932" y="{yc + 6}" font-size="20" fill="var(--muted)">·</text>')
    add(f'<text x="947" y="{yc + 6}" font-size="20" fill="var(--ink)">{sub("v", k)}</text>')
    dash = ' stroke-dasharray="7 5"' if k == 2 else ""
    add(
        f'<polyline points="{trace(975, 1170, yc + 18, 34 if k != 2 else 30, f)}" fill="none" stroke="{col}" stroke-width="2.5"{dash}/>'
    )

segs = [("pixel ≈ mean", False)]
for k in (1, 2, 3):
    segs += [(" + w", False), (str(k), True), ("·v", False), (str(k), True)]
formula = rich(segs)
add(
    f'<text x="1000" y="318" font-size="19" fill="var(--ink)" text-anchor="middle">{formula}</text>'
)

# ---- bottom line
add(
    f'<text x="600" y="{H - 20}" font-size="18" fill="var(--ink)" text-anchor="middle">'
    '~50 time courses: about <tspan font-weight="600" fill="var(--acc)">2 million</tspan> numbers instead of '
    '<tspan font-weight="600" fill="var(--data)">147 million</tspan>. '
    '<tspan fill="var(--muted)">What is left out is noise.</tspan></text>'
)
add("</svg>")
OUT.write_text("\n".join(s) + "\n")
print("wrote", OUT)
