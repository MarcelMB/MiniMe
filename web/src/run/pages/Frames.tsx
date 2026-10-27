import { useEffect, useState } from "react";
import { Computing } from "../../api";
import type { Img } from "../../draw";
import { STALE, latest } from "../../frames";
import { ImageView } from "../../ImageView";
import { sourceOptions, useRun } from "../RunContext";

const SOURCES = ["raw", "pre", "recon"] as const;

function FrameTile({ source }: { source: (typeof SOURCES)[number] }) {
  const run = useRun();
  const opt = sourceOptions(run.summary.available).find((o) => o.value === source)!;
  const [img, setImg] = useState<Img | null>(null);
  const [err, setErr] = useState("");
  const [retry, setRetry] = useState(0);
  const [wrap] = useState(latest);
  useEffect(() => {
    if (opt.disabled) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    wrap(run.cache.get(source, run.t)).then(
      (a) => {
        if (cancelled || a === STALE) return;
        setErr("");
        setImg({ data: a.data, W: run.summary.W, H: run.summary.H, lo: 0, hi: 255 });
      },
      (e: unknown) => {
        if (cancelled) return;
        if (e instanceof Computing) {
          setErr("Computing the contrast…");
          timer = setTimeout(() => {
            if (!cancelled) setRetry((r) => r + 1);
          }, 1500);
        } else setErr(e instanceof Error ? e.message : String(e));
      },
    );
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [source, run.t, run.cache, run.summary.W, run.summary.H, opt.disabled, wrap, retry]);
  return (
    <section className={opt.disabled ? "card dim" : "card"} aria-label={opt.label}>
      <span className="label">{opt.label}</span>
      {opt.disabled ? (
        <span className="muted">Not available: {opt.title}.</span>
      ) : (
        <>
          <ImageView image={img} units={run.units} contours={false} label={`${opt.label} frame`} />
          {err && <span className="muted">{err}</span>}
        </>
      )}
    </section>
  );
}

export function Frames() {
  return (
    <div className="bento" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
      {SOURCES.map((s) => (
        <FrameTile key={s} source={s} />
      ))}
    </div>
  );
}
