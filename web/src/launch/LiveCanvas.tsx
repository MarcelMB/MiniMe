import { useEffect, useRef } from "react";
import type { StitchWindow } from "./liveState";

export type Layer =
  | { kind: "outlines"; polygons: number[][][][]; color: string | ((unit: number) => string) }
  | { kind: "points"; points: Float32Array; color: string; r: number } // (row, col) pairs
  | { kind: "links"; segments: number[][]; color: string }; // [x0, y0, x1, y1] in full-res px

const PALETTE = ["#4f9cf9", "#f59e0b", "#10b981", "#ef4444", "#a855f7", "#14b8a6", "#ec4899", "#84cc16"];
export const windowColor = (i: number): string => PALETTE[i % PALETTE.length];

/** Links between unit centroids. `edges` numbers units over all windows' units (window `w`'s unit `u`
 * is `w.first + u`); links to a unit whose window the page does not have are left out. */
export function linkSegments(windows: Map<number, StitchWindow>, edges: Float64Array): number[][] {
  const cents = new Map<number, [number, number]>();
  for (const w of windows.values())
    for (let u = 0; u < w.centroids.length / 2; u++)
      cents.set(w.first + u, [w.centroids[2 * u], w.centroids[2 * u + 1]]);
  const out: number[][] = [];
  for (let e = 0; e + 1 < edges.length; e += 2) {
    const [a, b] = [cents.get(edges[e]), cents.get(edges[e + 1])];
    if (a && b) out.push([a[1], a[0], b[1], b[0]]);
  }
  return out;
}

const SCALE = 2; // canvas px per preview px

function drawImage(ctx: CanvasRenderingContext2D, image: Uint8Array, w: number, h: number, to: HTMLCanvasElement) {
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < image.length; i++) {
    const v = image[i];
    img.data.set([v, v, v, 255], 4 * i);
  }
  const b = document.createElement("canvas");
  [b.width, b.height] = [w, h];
  b.getContext("2d")?.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(b, 0, 0, to.width, to.height);
}

function drawLayer(ctx: CanvasRenderingContext2D, l: Layer, s: number) {
  if (l.kind === "outlines")
    l.polygons.forEach((polys, u) => {
      ctx.strokeStyle = typeof l.color === "string" ? l.color : l.color(u);
      ctx.lineWidth = 1;
      for (const poly of polys) {
        ctx.beginPath();
        poly.forEach(([x, y], i) => {
          if (i) ctx.lineTo((x + 0.5) * s, (y + 0.5) * s);
          else ctx.moveTo((x + 0.5) * s, (y + 0.5) * s);
        });
        ctx.closePath();
        ctx.stroke();
      }
    });
  else if (l.kind === "points") {
    ctx.fillStyle = l.color;
    for (let i = 0; i + 1 < l.points.length; i += 2) {
      ctx.beginPath();
      ctx.arc((l.points[i + 1] + 0.5) * s, (l.points[i] + 0.5) * s, l.r, 0, 2 * Math.PI);
      ctx.fill();
    }
  } else {
    ctx.strokeStyle = l.color;
    ctx.lineWidth = 1.5;
    for (const [x0, y0, x1, y1] of l.segments) {
      ctx.beginPath();
      ctx.moveTo((x0 + 0.5) * s, (y0 + 0.5) * s);
      ctx.lineTo((x1 + 0.5) * s, (y1 + 0.5) * s);
      ctx.stroke();
    }
  }
}

export function LiveCanvas(p: {
  image: Uint8Array | null;
  h: number;
  w: number;
  f: number;
  layers: Layer[];
  label: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    c.width = p.w * SCALE;
    c.height = p.h * SCALE;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, c.width, c.height);
    if (p.image) drawImage(ctx, p.image, p.w, p.h, c);
    const s = SCALE / p.f; // full-res px → canvas px
    for (const l of p.layers) drawLayer(ctx, l, s);
  }, [p.image, p.h, p.w, p.f, p.layers]);
  return (
    <canvas
      ref={ref}
      className="livecanvas"
      role="img"
      aria-label={p.label}
      style={{ aspectRatio: `${p.w} / ${p.h}` }}
    />
  );
}
