import { useEffect, useMemo, useRef, useState } from "react";
import { getJSON, postArray, postJSON, type NDArray } from "../api";
import { Slider } from "../controls";
import { useDebounced } from "../debounce";
import { fillImageData, imgFrom, type Img } from "../draw";
import { fmtDuration } from "../format";
import { STALE, latest } from "../frames";
import { ImageView } from "../ImageView";
import type { RawInfo } from "../types";
import { diameterOf, dragCircle, toImagePixel, type Circle } from "./circle";

/** Fractions of the session shown as thumbnails: one row, the start included so the
 * bleaching transient is visible. */
export const POSITIONS = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1];
/** The parameters the preview depends on (the pass-1 ones). */
const PASS1 = [
  "median_ksize",
  "tophat_wnd",
  "motion_bandpass",
  "motion_before_tophat",
  "motion_max_shift",
  "motion_down",
  "template_frames",
];

type Motion = { t0: number; frames: number; max_shift_px: number; quality_median: number };

/** The planes of a (n, H, W) array as images with their own contrast. */
function planes(a: NDArray): Img[] {
  const [n, H, W] = a.shape;
  return Array.from({ length: n }, (_, i) =>
    imgFrom({ shape: [H, W], dtype: a.dtype, data: a.data.subarray(i * H * W, (i + 1) * H * W) }, "gray", 1, 99.7),
  );
}

/** A magnified crop around the circle, so a cell a few px wide can be measured. */
function Inset({ img, c }: { img: Img; c: Circle }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const b = document.createElement("canvas");
    b.width = img.W;
    b.height = img.H;
    const bctx = b.getContext("2d")!;
    bctx.putImageData(fillImageData(img, bctx.createImageData(img.W, img.H)), 0, 0);
    const s = Math.max(24, Math.ceil(5 * c.r));
    const x0 = Math.round(c.cx + 0.5 - s / 2);
    const y0 = Math.round(c.cy + 0.5 - s / 2);
    const ctx = cv.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(b, x0, y0, s, s, 0, 0, cv.width, cv.height);
    const k = cv.width / s;
    ctx.strokeStyle = "#fb923c";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc((c.cx + 0.5 - x0) * k, (c.cy + 0.5 - y0) * k, c.r * k, 0, 2 * Math.PI);
    ctx.stroke();
  }, [img, c]);
  return (
    <canvas ref={ref} width={200} height={200} className="imageview" style={{ width: 200 }} role="img" aria-label="zoom on the circle" />
  );
}

/** The pass-1 steps in the order the run applies them: ``shown`` ones this preview computes on
 * the frame, the others are only decided here and run on every frame of the session. */
export function pipelineSteps(v: Record<string, unknown>): { label: string; shown: boolean }[] {
  const bp = v.motion_bandpass as [number, number] | undefined;
  const motionFirst = Boolean(v.motion_before_tophat);
  const median = { label: `median ${v.median_ksize} px`, shown: true };
  const bandpass = { label: `band-pass σ ${bp?.[0] ?? "?"}–${bp?.[1] ?? "?"} px`, shown: true };
  const motion = { label: `motion correction ≤ ${v.motion_max_shift} px`, shown: false };
  const tophat = { label: `tophat ${v.tophat_wnd} px`, shown: true };
  const rest = [
    { label: `${v.tile} px tiles · compression`, shown: false },
    { label: "seeds · init · CNMF", shown: false },
  ];
  return motionFirst ? [median, bandpass, motion, tophat, ...rest] : [median, tophat, motion, ...rest];
}

function Steps({ values }: { values: Record<string, unknown> }) {
  const steps = pipelineSteps(values);
  return (
    <div className="row" style={{ gap: 6, fontSize: 12 }} aria-label="Preprocessing order">
      <span className="chip" aria-pressed="true" style={{ cursor: "default" }}>
        raw frame
      </span>
      {steps.map((s) => (
        <span key={s.label} className="row" style={{ gap: 6 }}>
          <span className="muted">›</span>
          <span className="chip" aria-pressed={s.shown} style={{ cursor: "default" }} title={s.shown ? "computed on this frame" : "decided here, done in the run"}>
            {s.label}
          </span>
        </span>
      ))}
      <span className="muted" style={{ flexBasis: "100%" }}>
        Dark chips are computed on the frame below. The others are set here but run on the whole session: motion correction
        {values.motion_before_tophat ? " is estimated on the band-passed frame (vessels stay as landmarks), then the tophat is applied to the registered frames" : " is estimated on the tophat frame"}
        . "Check motion here" runs it on 150 frames around the chosen one.
      </span>
    </div>
  );
}

export function Preview(p: {
  raw: string;
  info: RawInfo;
  values: Record<string, unknown>;
  onChange: (name: string, v: unknown) => void;
  /** The cell diameter the user applied (the auto-tune after the run starts from it). */
  onDiameter?: (px: number) => void;
}) {
  const { frames, fps, width: W, height: H } = p.info;
  const ts = useMemo(() => POSITIONS.map((f) => Math.round(f * (frames - 1))), [frames]);
  const [t, setT] = useState(ts[3]);
  const params = useMemo(() => {
    const out: Record<string, unknown> = {};
    for (const k of PASS1) if (p.values[k] !== undefined) out[k] = p.values[k];
    return out;
  }, [p.values]);
  const request = useDebounced(JSON.stringify({ raw: p.raw, t, params }), 250);

  const [thumbs, setThumbs] = useState<(Img | null)[]>(() => POSITIONS.map(() => null));
  useEffect(() => {
    let cancelled = false;
    setThumbs(POSITIONS.map(() => null));
    ts.forEach((ti, i) =>
      postArray("/raw/preview", { raw_dir: p.raw, t: ti, params: {} }).then(
        (a) => !cancelled && setThumbs((x) => x.map((v, j) => (j === i ? planes(a)[0] : v))),
        () => undefined, // the main view reports the error
      ),
    );
    return () => {
      cancelled = true;
    };
  }, [p.raw, ts]);

  const [imgs, setImgs] = useState<Img[] | null>(null);
  const [err, setErr] = useState("");
  const [wrap] = useState(latest);
  useEffect(() => {
    const { raw, t, params } = JSON.parse(request);
    wrap(postArray("/raw/preview", { raw_dir: raw, t, params })).then(
      (a) => {
        if (a === STALE) return;
        setErr("");
        setImgs(planes(a));
      },
      (e: Error) => setErr(e.message),
    );
  }, [request, wrap]);

  const [circle, setCircle] = useState<Circle>({ cx: W / 2, cy: H / 2, r: 5 });
  const drag = useRef<{ mode: "move" | "resize"; grab: [number, number] } | null>(null);
  const startDrag = (e: React.PointerEvent<SVGElement>, mode: "move" | "resize") => {
    const svg = e.currentTarget.ownerSVGElement!;
    svg.setPointerCapture(e.pointerId);
    const [x, y] = toImagePixel(svg.getBoundingClientRect(), W, H, e.clientX, e.clientY);
    drag.current = { mode, grab: [x - circle.cx, y - circle.cy] };
  };
  const diameter = diameterOf(circle);
  const [applied, setApplied] = useState("");
  const applyDiameter = () =>
    getJSON<Record<string, unknown>>(`/params/derived?diameter=${diameter}`).then(
      (d) => {
        for (const [k, v] of Object.entries(d)) p.onChange(k, v);
        p.onDiameter?.(diameter);
        setApplied(`set ${Object.keys(d).join(", ")} for ${diameter} px cells; edit them under Parameters`);
      },
      (e: Error) => setApplied(e.message),
    );

  const [motion, setMotion] = useState<{ busy: boolean; result?: Motion; err?: string }>({ busy: false });
  const checkMotion = () => {
    setMotion({ busy: true });
    postJSON<Motion>("/raw/motion", { raw_dir: p.raw, t, params }).then(
      (result) => setMotion({ busy: false, result }),
      (e: Error) => setMotion({ busy: false, err: e.message }),
    );
  };

  const time = (f: number) => (fps ? fmtDuration(f / fps) : `frame ${f}`);
  const tophat = Number(p.values.tophat_wnd);
  const median = Number(p.values.median_ksize);
  const lw = Math.max(1, W / 300);
  const labels = ["raw", `median ${median}`, p.values.motion_before_tophat ? "band-pass (registration image)" : "registration image"];
  return (
    <section className="card span4" aria-label="2. Preprocessing preview">
      <span className="label">2 · Look at the frames and the preprocessing</span>
      <Steps values={p.values} />
      <div style={{ display: "grid", gridTemplateColumns: `repeat(${POSITIONS.length}, minmax(0, 1fr))`, gap: 8 }}>
        {POSITIONS.map((f, i) => (
          <button key={f} type="button" className="thumb" aria-pressed={t === ts[i]} onClick={() => setT(ts[i])}>
            <ImageView image={thumbs[i]} scale={1} label={`raw frame at ${Math.round(f * 100)} %`} />
            <span className="muted">
              {Math.round(f * 100)} % · {time(ts[i])}
            </span>
          </button>
        ))}
      </div>
      <Slider label="Frame" min={0} max={frames - 1} step={1} value={t} onChange={setT} format={(v) => `${v} · ${time(v)}`} />
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 3fr) minmax(0, 1fr)", gap: 12 }}>
        <div style={{ gridRow: "span 3", display: "flex", flexDirection: "column", gap: 6 }}>
          <span className="label">tophat {tophat} px · drag the circle onto a cell, its edge sets the size</span>
          <div style={{ position: "relative" }}>
            <ImageView image={imgs?.[3] ?? null} label="tophat frame" />
            {imgs && (
              <svg
                viewBox={`0 0 ${W} ${H}`}
                style={{ position: "absolute", inset: 0, width: "100%", height: "100%", touchAction: "none" }}
                aria-label="cell diameter circle"
                onPointerMove={(e) => {
                  if (!drag.current) return;
                  const [x, y] = toImagePixel(e.currentTarget.getBoundingClientRect(), W, H, e.clientX, e.clientY);
                  const { mode, grab } = drag.current;
                  setCircle((c) => dragCircle(c, mode, x, y, grab));
                }}
                onPointerUp={() => (drag.current = null)}
                onPointerCancel={() => (drag.current = null)}
              >
                <circle
                  cx={circle.cx + 0.5}
                  cy={circle.cy + 0.5}
                  r={circle.r}
                  fill="rgba(251, 146, 60, 0.15)"
                  stroke="#fb923c"
                  strokeWidth={lw}
                  style={{ cursor: "move" }}
                  onPointerDown={(e) => startDrag(e, "move")}
                />
                <circle
                  cx={circle.cx + 0.5 + circle.r}
                  cy={circle.cy + 0.5}
                  r={1.5 * lw}
                  fill="#fb923c"
                  style={{ cursor: "ew-resize" }}
                  onPointerDown={(e) => startDrag(e, "resize")}
                />
              </svg>
            )}
          </div>
        </div>
        {labels.map((label, i) => (
          <div key={label} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <span className="label">{label}</span>
            <ImageView image={imgs?.[i] ?? null} label={`${label} frame`} />
          </div>
        ))}
      </div>
      {err && <span className="error">{err}</span>}
      <div className="row">
        <Slider label="median_ksize" min={1} max={21} step={2} value={median} onChange={(v) => p.onChange("median_ksize", v)} />
        <Slider label="tophat_wnd" min={0} max={60} step={1} value={tophat} onChange={(v) => p.onChange("tophat_wnd", v)} />
      </div>
      <div className="row" style={{ alignItems: "flex-start" }}>
        {imgs && <Inset img={imgs[3]} c={circle} />}
        <div className="field" style={{ flex: "1 1 240px" }}>
          <span>
            cell diameter <b className="mono">{diameter} px</b>
          </span>
          <button className="btn" type="button" onClick={applyDiameter}>
            Set the size parameters from it
          </button>
          {applied && <span className="muted">{applied}</span>}
        </div>
        <div className="field" style={{ flex: "1 1 240px" }}>
          <span>motion around this frame</span>
          <button className="btn" type="button" disabled={motion.busy} onClick={checkMotion}>
            {motion.busy ? "Checking…" : "Check motion here"}
          </button>
          {motion.result && (
            <span className="muted">
              max shift {motion.result.max_shift_px.toFixed(1)} px, template match {motion.result.quality_median.toFixed(2)},{" "}
              {motion.result.frames} frames from {time(motion.result.t0)}
            </span>
          )}
          {motion.err && <span className="error">{motion.err}</span>}
        </div>
      </div>
    </section>
  );
}
