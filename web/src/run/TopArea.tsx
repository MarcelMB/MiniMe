import { useEffect, useMemo, useRef, useState } from "react";
import { Computing } from "../api";
import { Segmented, Slider, Toggle } from "../controls";
import type { Img } from "../draw";
import { STALE, latest } from "../frames";
import { fmtClock } from "../format";
import { ImageView } from "../ImageView";
import { playStep } from "../playback";
import { sourceOptions, useRun, type Source } from "./RunContext";

const SPEEDS = ["1", "4", "16"] as const;

export function TopArea() {
  const run = useRun();
  const { summary, t, setT } = run;
  const [source, setSource] = useState<Source>("max");
  const [contours, setContours] = useState(true);
  const [gain, setGain] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>("1");
  const [frame, setFrame] = useState<Img | null>(null);
  const [note, setNote] = useState("");
  const [retry, setRetry] = useState(0);
  const wrap = useRef(latest());
  const tRef = useRef(t);
  tRef.current = t;
  const fps = summary.fps ?? 20;

  // The frame index last drawn; playback only advances once the current frame is on screen.
  const drawn = useRef<number | null>(null);

  useEffect(() => {
    setFrame(null);
    setNote("");
    drawn.current = null;
    if (source === "max") setPlaying(false);
  }, [source]);

  useEffect(() => {
    if (source === "max") return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    wrap.current(run.cache.get(source, t)).then(
      (a) => {
        if (cancelled || a === STALE) return;
        setNote("");
        drawn.current = t;
        setFrame({ data: a.data, W: summary.W, H: summary.H, lo: 0, hi: 255 });
      },
      (e: unknown) => {
        if (cancelled) return;
        if (e instanceof Computing) {
          setNote("Computing the frame contrast…");
          timer = setTimeout(() => {
            if (!cancelled) setRetry((r) => r + 1);
          }, 1500);
        } else {
          setNote(e instanceof Error ? e.message : String(e));
          setPlaying(false);
        }
      },
    );
    if (playing) {
      const step = Math.max(1, Math.round((Number(speed) * fps) / 60));
      for (let k = 1; k <= 8; k++) {
        const u = t + k * step;
        if (u < summary.T && !run.cache.has(source, u)) run.cache.get(source, u).catch(() => {});
      }
    } else run.cache.prefetch(source, t, summary.T, 12, 3);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [source, t, run.cache, summary.W, summary.H, summary.T, playing, speed, fps, retry]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last: number | null = null;
    let carry = 0;
    const tick = (now: number) => {
      if (last === null) last = now;
      const a = playStep(tRef.current, drawn.current, summary.T, Number(speed), fps, now - last, carry);
      if (a) {
        last = now;
        carry = a.carry;
        if (a.t !== tRef.current) setT(a.t);
        if (a.ended) {
          setPlaying(false);
          return;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, fps, summary.T, setT]);

  const base = source === "max" ? run.maxProj : frame;
  const shown = useMemo(() => base && { ...base, hi: base.lo + (base.hi - base.lo) / gain }, [base, gain]);
  return (
    <section className="map" aria-label="Field of view" style={{ aspectRatio: `${summary.W} / ${summary.H}`, maxHeight: "72vh" }}>
      <ImageView
        image={shown}
        units={run.units}
        contours={contours}
        sel={run.sel}
        onPick={run.setSel}
        label={`${source} frame with unit outlines`}
        scale={summary.W > 400 ? 2 : 4}
      />
      <div className="overlay">
        <div className="row">
          <Segmented label="Image" glass options={sourceOptions(summary.available)} value={source} onChange={setSource} />
          <span className="spacer" />
          <div className="glass" style={{ padding: "6px 14px" }}>
            <Toggle label="Contours" checked={contours} onChange={setContours} />
          </div>
        </div>
        <div className="glass player">
          <button
            type="button"
            className="btn"
            aria-label={playing ? "Pause" : "Play"}
            disabled={source === "max" && !playing}
            title={source === "max" ? "choose Raw, Preprocessed or Reconstruction to play" : undefined}
            onClick={() => setPlaying(!playing)}
            style={{ background: "rgba(255,255,255,0.92)", color: "#0b1220", width: 44, padding: 0, justifyContent: "center" }}
          >
            {playing ? "❚❚" : "▶"}
          </button>
          <span className="mono" style={{ minWidth: 80 }}>
            {fmtClock(t, summary.fps)}
          </span>
          <input
            type="range"
            aria-label="Scrub recording"
            min={0}
            max={summary.T - 1}
            value={t}
            onChange={(e) => setT(Number(e.target.value))}
          />
          <Segmented label="Speed" glass options={SPEEDS.map((s) => ({ value: s, label: `${s}×` }))} value={speed} onChange={setSpeed} />
          <div style={{ minWidth: 180 }}>
            <Slider label="Contrast" min={0.5} max={4} step={0.1} value={gain} onChange={setGain} format={(v) => `${v.toFixed(1)}×`} />
          </div>
        </div>
        {note && (
          <span className="glass" style={{ padding: "6px 14px", alignSelf: "flex-start" }}>
            {note}
          </span>
        )}
      </div>
    </section>
  );
}
