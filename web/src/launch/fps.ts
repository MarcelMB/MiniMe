/** The frame rate the launcher sends: ``null`` keeps the one read from the files. */
export function fpsOverride(
  text: string,
  detected: number | null,
): { fps: number | null; error: string } {
  const t = text.trim();
  if (t === "") return { fps: null, error: "" };
  const v = Number(t);
  if (!Number.isFinite(v) || v <= 0 || v > 1000)
    return { fps: null, error: "frame rate must be in (0, 1000] Hz" };
  return { fps: v === detected ? null : v, error: "" };
}
