import { useEffect, useRef } from "react";
import { clientToPixel, drawPoints, drawPolygons, fillImageData, pickUnit, type Img } from "./draw";
import type { Unit } from "./types";

const NO_UNITS: Unit[] = [];

export function ImageView(p: {
  image: Img | null;
  units?: Unit[];
  contours?: boolean;
  sel?: number | null;
  onPick?: (id: number | null) => void;
  points?: number[][];
  pointColor?: string;
  label?: string;
  scale?: number;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const buf = useRef<HTMLCanvasElement | null>(null);
  const scale = p.scale ?? 2;
  const units = p.units ?? NO_UNITS;

  useEffect(() => {
    const c = ref.current;
    const img = p.image;
    if (!c || !img) return;
    c.width = img.W * scale;
    c.height = img.H * scale;
    const b = buf.current ?? document.createElement("canvas");
    buf.current = b;
    b.width = img.W;
    b.height = img.H;
    const bctx = b.getContext("2d")!;
    bctx.putImageData(fillImageData(img, bctx.createImageData(img.W, img.H)), 0, 0);
    const ctx = c.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(b, 0, 0, img.W * scale, img.H * scale);
    if (p.contours) drawPolygons(ctx, units, scale, p.sel ?? null);
    if (p.points) drawPoints(ctx, p.points, scale, p.pointColor ?? "#fb923c");
  }, [p.image, units, p.contours, p.sel, p.points, p.pointColor, scale]);

  if (!p.image) return <div className="imageview" aria-busy="true" style={{ aspectRatio: "1 / 1" }} />;
  const img = p.image;
  return (
    <canvas
      ref={ref}
      className={p.className ?? "imageview"}
      style={{ aspectRatio: `${img.W} / ${img.H}` }}
      role="img"
      aria-label={p.label ?? "image"}
      onClick={(e) => {
        if (!p.onPick) return;
        const r = e.currentTarget.getBoundingClientRect();
        const at = clientToPixel(r, img.W, img.H, e.clientX, e.clientY);
        if (at) p.onPick(pickUnit(units, at[0], at[1]));
      }}
    />
  );
}
