import { useEffect, useState } from "react";
import { Segmented } from "../controls";
import type { SpeedPanelProps } from "../features";
import { fmtDuration } from "../format";
import { DEVICES, type Device, decide, estimateSeconds, type SpeedTest } from "../gpu/plan";
import { workerCount } from "../stream/pool";
import { listSegments } from "./avi";
import { cachedTest, pageGpu, runGoing, runSpeedTest } from "./speed";
import { state } from "./state";

const LABELS: Record<Device, string> = { auto: "Auto", cpu: "CPU", gpu: "GPU", both: "Both" };
const ROWS: { mode: "cpu" | "gpu" | "both"; what: string }[] = [
  { mode: "cpu", what: "CPU: every frame worker runs the kernels" },
  { mode: "gpu", what: "GPU: the kernels on the GPU, the shift estimate and the rest on the frame workers" },
  { mode: "both", what: "Both: some frame workers on the GPU, the others on the CPU" },
];

/** New run's Speed card (browser build): this machine's GPU and CPU, the measured pass-1 rate and the
 * estimated run time of the chosen videos per mode, and where the run's per-frame stages go. */
export function SpeedPanel({ raw, frames, width, height, params, device, onDevice }: SpeedPanelProps) {
  const workers = workerCount(params);
  const cores = navigator.hardwareConcurrency || 0;
  const [gpu, setGpu] = useState<{ name: string; reason: string } | null>(null);
  const [test, setTest] = useState<SpeedTest | null>(null);
  const [selfTest, setSelfTest] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const [asked, setAsked] = useState(0); // a Re-test click

  useEffect(() => {
    let live = true;
    void pageGpu().then((g) => live && setGpu(g));
    return () => {
      live = false;
    };
  }, []);

  // the cached test of this machine, else a new one (a few seconds); Re-test measures again
  // (`params` only matter through `workers`)
  useEffect(() => {
    if (!gpu || !raw) return;
    let live = true;
    const go = async () => {
      setNote("");
      if (gpu.reason) {
        setTest(null);
        setSelfTest("");
        return;
      }
      const cached = asked ? null : await cachedTest(gpu.name, height, width, workers);
      if (cached) {
        if (live) {
          setTest(cached);
          setSelfTest("pass");
        }
        return;
      }
      if (await runGoing()) {
        if (live) setNote("A run is going: the speed test waits for it (Re-test once it is done).");
        return;
      }
      setBusy(true);
      try {
        const avis = (await listSegments(await state.folders.resolve(raw))).files;
        const a = await runSpeedTest(avis, params, asked > 0);
        if (!live) return;
        setTest(a.test);
        setSelfTest(a.selfTest);
        if (a.reason) setGpu({ name: a.gpu, reason: a.reason });
      } catch (e) {
        if (live) setNote(`The speed test failed: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        if (live) setBusy(false);
      }
    };
    void go();
    return () => {
      live = false;
    };
  }, [gpu?.name, gpu?.reason, raw, width, height, workers, asked]);

  const reason = gpu?.reason ?? "";
  const pick = decide("auto", workers, reason, test).use;
  const eta = test ? estimateSeconds(frames, test.fps) : null;
  const options = DEVICES.map((d) => ({
    value: d,
    label: d === "auto" && test ? `Auto (${LABELS[pick]})` : LABELS[d],
    disabled: (d === "gpu" || d === "both") && Boolean(reason),
    title: d === "both" && workers < 2 ? "one frame worker: Both runs as GPU" : undefined,
  }));
  return (
    <section className="card span4" aria-label="Speed">
      <span className="label">Speed · where the per-frame stages run</span>
      <span className="muted">
        {gpu === null
          ? "Looking for a GPU…"
          : reason
            ? `No usable GPU: ${reason}.`
            : `GPU: ${gpu.name}${selfTest && selfTest !== "pass" ? ` (self-test failed: ${selfTest})` : ""}`}{" "}
        · {cores} CPU cores, {workers} frame {workers === 1 ? "worker" : "workers"}
      </span>
      {test && (
        <table className="data" aria-label="Measured speed">
          <thead>
            <tr>
              <th>Mode</th>
              <th>Pass 1 (measured)</th>
              <th>Whole run (about)</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.filter((r) => r.mode !== "both" || workers > 1).map((r) => (
              <tr key={r.mode} aria-current={(device === "auto" ? pick : device) === r.mode}>
                <td title={r.what}>{LABELS[r.mode]}</td>
                <td>{test.fps[r.mode] > 0 ? `${test.fps[r.mode].toFixed(0)} fps` : "—"}</td>
                <td>{eta && eta[r.mode] > 0 ? fmtDuration(eta[r.mode]) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {busy && <span className="muted">Measuring the GPU and the CPU on these videos (a few seconds)…</span>}
      {note && <span className="muted">{note}</span>}
      <div className="row">
        <Segmented label="Device" options={options} value={device} onChange={onDevice} />
        <span className="spacer" />
        <button
          className="btn"
          type="button"
          disabled={busy || !raw || gpu === null || Boolean(reason)}
          onClick={() => setAsked((n) => n + 1)}
        >
          Re-test
        </button>
      </div>
      <span className="muted">
        Every mode gives the same results bit for bit (the GPU's kernels pass a self-test on these videos first, and a
        GPU that fails during the run hands its frames to the CPU). The estimate scales a CPU run by the measured pass-1
        rates.
      </span>
    </section>
  );
}
