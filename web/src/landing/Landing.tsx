import type { ReactNode } from "react";
import { href } from "../router";
import { useScheme } from "../scheme";
import { FIGURES, type FigureName, figureSrc } from "./figures";

const REPO = "https://github.com/MarcelMB/MiniMe";

/** A README figure in the colour scheme the page is shown in. */
function Figure({ name }: { name: FigureName }) {
  const { light, dark } = figureSrc(name);
  const scheme = useScheme();
  return (
    <figure className="landing-fig">
      <img src={scheme === "dark" ? dark : light} alt={FIGURES[name]} loading="lazy" />
    </figure>
  );
}

const C = ({ children }: { children: ReactNode }) => <code className="mono">{children}</code>;

const STEPS: { title: string; who: "you" | "automatic"; what: ReactNode }[] = [
  {
    title: "Choose the folder with the raw videos",
    who: "you",
    what: (
      <>
        Grayscale <C>.avi</C> files whose names start with a number (<C>0.avi</C>, <C>1.avi</C>, … as the Miniscope DAQ
        writes them), read as one recording in that order. Set the frame rate if there is no <C>meta.json</C> /{" "}
        <C>metaData.json</C>.
      </>
    ),
  },
  {
    title: "Look at the frames and the preprocessing",
    who: "you",
    what: (
      <>
        Drag the circle onto a cell: its size sets the cell diameter and the parameters that follow from it. Check the
        median filter and background removal (tophat) on a few frames.
      </>
    ),
  },
  {
    title: "Name the run and check the parameters",
    who: "you",
    what: <>Start from the defaults, or from the demo settings for the demo session.</>,
  },
  {
    title: "Run",
    who: "automatic",
    what: (
      <>
        Motion correction, compression into <C>W</C> and <C>V</C>, finding cells window by window, joining the windows,
        and the full traces. Each stage shows its progress.
      </>
    ),
  },
  {
    title: "Look at the results",
    who: "you",
    what: (
      <>
        Overview, motion, compression, seeds, units (footprints) and traces, next to the raw and preprocessed frames.
        Results are saved to a folder you pick or to the browser's own storage.
      </>
    ),
  },
];

/** The page everyone lands on (the browser build): what MiniMe does, why it fits in a browser tab
 * and how to use it, after the README on GitHub. */
export function Landing() {
  return (
    <div className="page landing">
      <section className="landing-hero">
        <span className="label">Miniscope Matrix-factorization Engine</span>
        <h1>Finds the cells and their calcium traces in one-photon miniscope recordings, inside your browser.</h1>
        <p className="lead">Nothing to install, nothing to upload.</p>
        <div className="row">
          <a className="btn primary" href={href({ name: "new" })}>
            Start a new run
          </a>
          <a className="btn" href={href({ name: "home" })}>
            Your runs and the demo
          </a>
        </div>
        <p className="muted">
          Works in a recent Chrome, Edge, Firefox or Safari. Your videos are read from your own disk and every step runs
          on your computer; the page never sends your videos or results anywhere.
        </p>
      </section>

      <section className="card landing-section" aria-labelledby="why">
        <h2 id="why">Why it fits in a browser tab</h2>
        <p>
          A miniscope movie stores every pixel in every frame, but most of that is repetition: the pixels of one cell
          brighten and fade together, and the background drifts across all of them at once. So a stretch of movie is
          well described by a few shared time courses plus, for every pixel, a weight saying how much of each it shows.
        </p>
        <Figure name="factors" />
        <p>
          MiniMe cuts the movie into 64 × 64 px tiles and compresses each tile into its mean, the weights <C>W</C> and
          the time courses <C>V</C> (a truncated SVD that keeps only the components that carry signal). Finding cells,
          their footprints and their traces (CNMF) then works on <C>W</C> and <C>V</C> and never reads the movie again.
        </p>
        <Figure name="algebra" />
        <p>
          That is what makes long recordings fit. A browser tab gets about 2 GB to work with; a 2-hour session is far
          bigger than that. Drawn to scale, with area standing for size:
        </p>
        <Figure name="size-area" />
        <p>The same numbers on a log scale, where each step is 10×:</p>
        <Figure name="budget" />
        <p>
          So MiniMe never holds the whole movie. The videos stay on disk and are read in pieces: it finds the cells
          window by window (2.5 min each), merges the windows' cell lists into one, and then reads every frame once more
          for each cell's full trace. At any moment only one window, or the cell list and some running totals, is in
          memory: about 70 MB, however long the recording is.
        </p>
        <Figure name="pipeline" />
      </section>

      <section className="card landing-section" aria-labelledby="how">
        <h2 id="how">How to use it</h2>
        <p>You set up the run; the rest is automatic.</p>
        <ol className="landing-steps">
          {STEPS.map((s, i) => (
            <li key={s.title}>
              <span className="landing-step-n">{i + 1}</span>
              <div>
                <h3>
                  {s.title} <span className={`landing-who ${s.who}`}>{s.who}</span>
                </h3>
                <p>{s.what}</p>
              </div>
            </li>
          ))}
        </ol>
        <p className="muted">MiniMe gives footprints and fluorescence traces; it does not infer spikes.</p>
        <div className="row">
          <a className="btn primary" href={href({ name: "new" })}>
            Start a new run
          </a>
          <a className="btn" href={href({ name: "home" })}>
            Your runs and the demo
          </a>
        </div>
      </section>

      <div className="landing-pair">
        <section className="card landing-section" aria-labelledby="demo">
          <h2 id="demo">Try the demo</h2>
          <p>
            The <a href={href({ name: "home" })}>Runs</a> page has a button that downloads a demo session: a synthetic
            one-photon recording through a GRIN lens (600 × 600 px, 20 fps) with known ground truth, as a 5-minute or a
            30-minute session. It goes straight into a folder you pick or into the browser's own storage, and New run
            opens on it with the demo settings.
          </p>
        </section>
        <section className="card landing-section" aria-labelledby="more">
          <h2 id="more">Python package and license</h2>
          <p>
            MiniMe started as a Python package (<C>pip install minime-cnmf</C>), which stays the reference the browser
            app is checked against; see <a href={`${REPO}/blob/main/docs/python.md`}>docs/python.md</a>.
          </p>
          <p>
            GPL-3.0-or-later. Parts of the algorithms are ported from{" "}
            <a href="https://github.com/denisecailab/minian">Minian</a> (GPL-3.0). Source on <a href={REPO}>GitHub</a>.
          </p>
        </section>
      </div>
    </div>
  );
}
