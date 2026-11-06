/** The README's figures, in `public/readme/` as `<name>-light.svg` and `<name>-dark.svg`
 * (copied from `docs/assets/readme/`, where `build.py` makes them from `src/*.svg`). */
export const FIGURES = {
  factors:
    "Every pixel, every frame; pixels of a cell move together; so a few time courses plus weights describe the movie",
  algebra: "Y ≈ mean · 1ᵀ + W · Vᵀ: a tile movie becomes a mean, weights and time courses",
  "size-area":
    "Drawn to scale: the 78 GB movie is a large square, the 2 GB tab limit a small square in its corner, MiniMe's 70 MB a dot inside that",
  budget:
    "A 2-hour 600 × 600 px session at 30 fps is 78 GB as raw video; the browser tab has about 2 GB; MiniMe keeps about 70 MB",
  pipeline:
    "Motion template, windows, stitch, streaming pass and footprints: one window or the cell list is in memory at a time, never the whole session",
} as const;

export type FigureName = keyof typeof FIGURES;

/** The figure's two files, relative to the page (the build's `base` is `./`). */
export function figureSrc(name: FigureName): { light: string; dark: string } {
  return { light: `./readme/${name}-light.svg`, dark: `./readme/${name}-dark.svg` };
}
