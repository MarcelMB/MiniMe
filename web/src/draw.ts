import type { NDArray } from "./api";
import type { Unit } from "./types";

export type Img = { data: ArrayLike<number>; W: number; H: number; lo: number; hi: number; cmap?: "gray" | "heat" };

export function percentile(a: ArrayLike<number>, q: number): number {
  const s = Float64Array.from(a as ArrayLike<number>).sort();
  if (s.length === 0) return 0;
  const i = Math.min(s.length - 1, Math.max(0, Math.round((q / 100) * (s.length - 1))));
  return s[i];
}

export function imgFrom(arr: NDArray, cmap: "gray" | "heat" = "gray", qLo = 0.5, qHi = 99.8): Img {
  const [H, W] = arr.shape;
  const lo = percentile(arr.data, qLo);
  const hi = percentile(arr.data, qHi);
  return { data: arr.data, W, H, lo, hi: hi > lo ? hi : lo + 1, cmap };
}

const HEAT: [number, number, number][] = [
  [13, 8, 135],
  [84, 2, 163],
  [139, 10, 165],
  [185, 50, 137],
  [219, 92, 104],
  [244, 136, 73],
  [254, 188, 43],
  [240, 249, 33],
];

/** Plasma-like ramp for maps (rank, noise, background). */
export function heat(v: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, v)) * (HEAT.length - 1);
  const i = Math.min(HEAT.length - 2, Math.floor(x));
  const f = x - i;
  const a = HEAT[i];
  const b = HEAT[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export function fillImageData(img: Img, out: ImageData): ImageData {
  const k = 1 / (img.hi - img.lo);
  const d = out.data;
  for (let i = 0; i < img.W * img.H; i++) {
    const v = Math.min(1, Math.max(0, (img.data[i] - img.lo) * k));
    if (img.cmap === "heat") {
      const [r, g, b] = heat(v);
      d[4 * i] = r;
      d[4 * i + 1] = g;
      d[4 * i + 2] = b;
    } else {
      const g = Math.round(v * 255);
      d[4 * i] = g;
      d[4 * i + 1] = g;
      d[4 * i + 2] = g;
    }
    d[4 * i + 3] = 255;
  }
  return out;
}

export function pointInPolygon(x: number, y: number, poly: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Unit whose outline contains (x, y) in pixel-centre coordinates, else the nearest centroid. */
export function pickUnit(units: Unit[], x: number, y: number, maxDist = 8): number | null {
  for (const u of units) {
    if (u.polygons.some((p) => p.length >= 3 && pointInPolygon(x, y, p))) return u.id;
  }
  let best: number | null = null;
  let bestD = maxDist;
  for (const u of units) {
    const d = Math.hypot(u.centroid[1] - x, u.centroid[0] - y);
    if (d <= bestD) {
      best = u.id;
      bestD = d;
    }
  }
  return best;
}

export function drawPolygons(ctx: CanvasRenderingContext2D, units: Unit[], scale: number, sel: number | null) {
  const path = (u: Unit) => {
    ctx.beginPath();
    for (const poly of u.polygons) {
      poly.forEach(([x, y], i) => {
        const px = (x + 0.5) * scale;
        const py = (y + 0.5) * scale;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
    }
  };
  ctx.lineWidth = Math.max(1, scale / 2);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
  for (const u of units) {
    if (u.id === sel) continue;
    path(u);
    ctx.stroke();
  }
  const s = units.find((u) => u.id === sel);
  if (s) {
    ctx.lineWidth = Math.max(2, scale);
    ctx.strokeStyle = "#fb923c";
    path(s);
    ctx.stroke();
  }
}

export function drawPoints(ctx: CanvasRenderingContext2D, pts: number[][], scale: number, color: string) {
  ctx.fillStyle = color;
  const r = Math.max(1.5, scale * 0.9);
  for (const [h, w] of pts) {
    ctx.beginPath();
    ctx.arc((w + 0.5) * scale, (h + 0.5) * scale, r, 0, 2 * Math.PI);
    ctx.fill();
  }
}

type Box = { left: number; top: number; width: number; height: number };

/** Client point -> pixel-centre coords of a W x H image shown with object-fit: contain in ``rect``; null outside the image. */
export function clientToPixel(
  rect: Box,
  W: number,
  H: number,
  clientX: number,
  clientY: number,
): [number, number] | null {
  const k = Math.min(rect.width / W, rect.height / H);
  const cw = W * k;
  const ch = H * k;
  const dx = clientX - (rect.left + (rect.width - cw) / 2);
  const dy = clientY - (rect.top + (rect.height - ch) / 2);
  if (dx < 0 || dy < 0 || dx > cw || dy > ch) return null;
  return [(dx / cw) * W - 0.5, (dy / ch) * H - 0.5];
}
