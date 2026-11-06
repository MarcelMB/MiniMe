<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/MarcelMB/MiniMe/main/docs/assets/minime-logo-dark-1200.png">
    <img alt="MiniMe" src="https://raw.githubusercontent.com/MarcelMB/MiniMe/main/docs/assets/minime-logo-1200.png" width="420">
  </picture>
</p>

<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/MarcelMB/MiniMe/main/docs/assets/minime-heading-dark-1200.png">
    <img alt="MiniMe (Miniscope Matrix-factorization Engine)" src="https://raw.githubusercontent.com/MarcelMB/MiniMe/main/docs/assets/minime-heading-1200.png" height="32">
  </picture>
</h1>

Finds the cells and their calcium traces in one-photon miniscope recordings, inside your
browser. Nothing to install, nothing to upload.

<p align="center"><b><a href="https://marcelbrosch.com/MiniMe/">Open MiniMe →</a></b></p>

Works in a recent Chrome, Edge, Firefox or Safari. Your videos are read from your own disk
and every step runs on your computer; the page never sends your videos or results anywhere.

## Why it fits in a browser tab

A miniscope movie stores every pixel in every frame, but most of that is repetition: the
pixels of one cell brighten and fade together, and the background drifts across all of them
at once. So a stretch of movie is well described by a few shared time courses plus, for
every pixel, a weight saying how much of each it shows.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/factors-dark.svg">
    <img alt="Every pixel, every frame; pixels of a cell move together; so a few time courses plus weights describe the movie" src="docs/assets/readme/factors-light.svg" width="880">
  </picture>
</p>

MiniMe cuts the movie into 64 × 64 px tiles and compresses each tile into its mean, the
weights `W` and the time courses `V` (a truncated SVD that keeps only the components that
carry signal). Finding cells, their footprints and their traces (CNMF) then works on `W`
and `V` and never reads the movie again.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/algebra-dark.svg">
    <img alt="Y ≈ mean · 1ᵀ + W · Vᵀ: a tile movie becomes a mean, weights and time courses" src="docs/assets/readme/algebra-light.svg" width="880">
  </picture>
</p>

That is what makes long recordings fit. A browser tab gets about 2 GB to work with; a
2-hour session is far bigger than that. Drawn to scale, with area standing for size:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/size-area-dark.svg">
    <img alt="Drawn to scale: the 78 GB movie is a large square, the 2 GB tab limit a small square in its corner, MiniMe's 70 MB a dot inside that" src="docs/assets/readme/size-area-light.svg" width="880">
  </picture>
</p>

The same numbers on a log scale, where each step is 10×:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/budget-dark.svg">
    <img alt="A 2-hour 600 × 600 px session at 30 fps is 78 GB as raw video; the browser tab has about 2 GB; MiniMe keeps about 70 MB" src="docs/assets/readme/budget-light.svg" width="880">
  </picture>
</p>

So MiniMe never holds the whole movie. The videos stay on disk and are read in pieces: it
finds the cells window by window (2.5 min each), merges the windows' cell lists into one, and
then reads every frame once more for each cell's full trace. At any moment only one window,
or the cell list and some running totals, is in memory: about 70 MB, however long the
recording is.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/pipeline-dark.svg">
    <img alt="Motion template, windows, stitch, streaming pass and footprints: one window or the cell list is in memory at a time, never the whole session" src="docs/assets/readme/pipeline-light.svg" width="880">
  </picture>
</p>

## How to use it

You set up the run; the rest is automatic.

| Step | | What happens |
| --- | --- | --- |
| 1. Choose the folder with the raw videos | you | Grayscale `.avi` files whose names start with a number (`0.avi`, `1.avi`, … as the Miniscope DAQ writes them), read as one recording in that order. Set the frame rate if there is no `meta.json` / `metaData.json`. |
| 2. Look at the frames and the preprocessing | you | Drag the circle onto a cell: its size sets the cell diameter and the parameters that follow from it. Check the median filter and background removal (tophat) on a few frames. |
| 3. Name the run and check the parameters | you | Start from the defaults, or from the demo settings for the demo session. |
| 4. Run | automatic | Motion correction, compression into `W` and `V`, finding cells window by window, joining the windows, and the full traces. Each stage shows its progress. |
| 5. Look at the results | you | Overview, motion, compression, seeds, units (footprints) and traces, next to the raw and preprocessed frames. Results are saved to a folder you pick or to the browser's own storage. |

MiniMe gives footprints and fluorescence traces; it does not infer spikes.

## Try the demo

MiniMe's start page has a button that downloads a demo session: a synthetic one-photon
recording through a GRIN lens (600 × 600 px, 20 fps) with known ground truth, as a 5-minute
or a 30-minute session. It goes straight into a folder you pick or into the browser's own
storage, and New run opens on it with the demo settings.

## Python package

MiniMe started as a Python package (`pip install minime-cnmf`), which stays the reference
the browser app is checked against. Its install, command line, parameters and differences
from Minian are in [docs/python.md](docs/python.md).

## License

GPL-3.0-or-later. Parts of the algorithms are ported from
[Minian](https://github.com/denisecailab/minian) (GPL-3.0).
