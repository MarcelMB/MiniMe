/** Home's banner in the browser build: the analysis runs in the tab, so the data stays on this computer.
 * Below it, its own tile: any laptop can run it (CPU cores set the speed; `device: auto` picks the faster of
 * CPU and GPU from a speed test on the recording, `stream/device.ts`). */
export function PrivacyNote() {
  return (
    <>
      <section className="card privacy" aria-label="Your data stays here">
        <svg className="privacy-icon" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 2 4 5v6c0 5 3.4 9.4 8 11 4.6-1.6 8-6 8-11V5l-8-3Z" />
          <path d="m8.5 12 2.5 2.5 4.5-5" />
        </svg>
        <div className="privacy-text">
          <h2>Your data never leaves this computer.</h2>
          <p>
            MiniMe runs entirely inside this browser tab. Your videos are read from your own disk, every step of the
            analysis runs on this machine, and the results are saved to a folder you pick or to this browser's own
            storage. Nothing is uploaded: the page only loads its own code (and the demo recording, if you download it)
            and never sends your videos or results anywhere.
          </p>
        </div>
      </section>
      <MachineNote />
    </>
  );
}

/** Any laptop runs MiniMe; only the time differs. */
function MachineNote() {
  return (
    <section className="card privacy machine" aria-label="Runs on any laptop">
      <svg className="privacy-icon" viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="5" width="16" height="11" rx="1.5" />
        <path d="M2 19h20" />
      </svg>
      <div className="privacy-text">
        <h2>Runs on any laptop.</h2>
        <p>
          No install and no special hardware. MiniMe spreads the work over this computer's CPU cores, so more cores
          finish sooner. If the browser offers a GPU, MiniMe times the CPU and the GPU on your recording and
          automatically uses whichever is faster.
        </p>
      </div>
    </section>
  );
}
