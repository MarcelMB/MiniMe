/** The cell-diameter circle on the preview, in image pixels (pixel-centre coordinates). */
export type Circle = { cx: number; cy: number; r: number };

export type Box = { left: number; top: number; width: number; height: number };

/** Client coordinates to image pixels for an image of W x H letter-boxed inside ``rect``;
 * unlike ``clientToPixel`` a point outside the image is not null but clamped. */
export function toImagePixel(rect: Box, W: number, H: number, clientX: number, clientY: number): [number, number] {
  const k = Math.min(rect.width / W, rect.height / H);
  const x = (clientX - rect.left - (rect.width - W * k) / 2) / k - 0.5;
  const y = (clientY - rect.top - (rect.height - H * k) / 2) / k - 0.5;
  return [Math.min(W - 0.5, Math.max(-0.5, x)), Math.min(H - 0.5, Math.max(-0.5, y))];
}

/** ``c`` after a drag to image point (x, y): the whole circle keeps the grab offset,
 * the edge handle sets the radius (at least half a pixel). */
export function dragCircle(
  c: Circle,
  mode: "move" | "resize",
  x: number,
  y: number,
  grab: [number, number],
): Circle {
  if (mode === "move") return { ...c, cx: x - grab[0], cy: y - grab[1] };
  return { ...c, r: Math.max(0.5, Math.hypot(x - c.cx, y - c.cy)) };
}

/** Diameter in px, to half a pixel. */
export function diameterOf(c: Circle): number {
  return Math.round(4 * c.r) / 2;
}
