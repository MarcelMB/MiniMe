"""Build the README figures: src/NAME.svg (CSS-variable colours) -> NAME-light.svg, NAME-dark.svg.

GitHub shows README images through <img>, which cannot see a page's CSS variables, so each
figure is written twice with the colours filled in; the README picks one with <picture>.

    python3 docs/assets/readme/build.py
"""

import re
from pathlib import Path

HERE = Path(__file__).parent
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "ink": "#1f2328",
        "muted": "#5c6670",
        "line": "#c9ccd1",
        "panel": "#ffffff",
        "acc": "#1f7a8c",
        "acc-soft": "#dceff2",
        "old": "#8a8f98",
        "old-soft": "#ececef",
        "data": "#b7791f",
        "data-soft": "#f8ecd3",
        "warn": "#c2410c",
    },
    "dark": {
        "ink": "#e6e8eb",
        "muted": "#9aa3ad",
        "line": "#3a414a",
        "panel": "#161b22",
        "acc": "#5cc1d4",
        "acc-soft": "#163a42",
        "old": "#7c838c",
        "old-soft": "#262a30",
        "data": "#e2b35a",
        "data-soft": "#3a2f16",
        "warn": "#f0894f",
    },
}

VAR = re.compile(r"var\(--([a-z-]+)\)")


def fill_colours(svg: str, palette: dict[str, str]) -> str:
    """Replace every var(--name) with the palette's hex colour."""

    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in palette:
            raise KeyError(f"unknown colour variable --{name}")
        return palette[name]

    return VAR.sub(sub, svg)


def set_font(svg: str) -> str:
    """Give the root <svg> a system font stack unless it already names one."""
    if "font-family" in svg.split(">", 1)[0]:
        return svg
    return svg.replace("<svg ", f'<svg font-family="{FONT}" ', 1)


def build(src: Path) -> None:
    svg = set_font(src.read_text())
    for theme, palette in PALETTES.items():
        out = HERE / f"{src.stem}-{theme}.svg"
        out.write_text(fill_colours(svg, palette))
        print(f"{src.name} -> {out.name}")


if __name__ == "__main__":
    for path in sorted((HERE / "src").glob("*.svg")):
        build(path)
