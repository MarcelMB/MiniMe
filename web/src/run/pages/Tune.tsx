import { useMemo, useState } from "react";
import { ApiError, getJSON, postJSON } from "../../api";
import { ErrorNote, Loading, Segmented } from "../../controls";
import { fmtDuration, fmtInt } from "../../format";
import { ImageView } from "../../ImageView";
import { ParamsForm } from "../../launch/ParamsForm";
import { changedOnly, failingCount } from "../../launch/params";
import { useResource } from "../../poll";
import { go } from "../../router";
import type { Checks, CurvePoint, Flags, Job, Pairs, ParamMeta, Proposal, Rerun, Unit } from "../../types";
import { rerunQuery, useRun } from "../RunContext";

/** The parameters auto-tune searches, in its order (also the common ones of the form). */
export const SEARCHED = ["seeds_diff_thres", "pnr_thres", "merge_corr", "init_merge_corr", "inita_corr", "sparse_snr", "loop_merge_corr"];
export const REASONS = ["low_pnr", "duplicate", "in_pieces", "bad_size"] as const;
type Reason = (typeof REASONS)[number];
const REASON_LABEL: Record<Reason, string> = { low_pnr: "low PNR", duplicate: "duplicate", in_pieces: "in pieces", bad_size: "bad size" };

export function fmtValue(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (Array.isArray(v)) return v.map(fmtValue).join(", ");
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}

/** ``sparse_snr 3.74→0.96 · merge_corr 0.85→0.8`` for a rerun's changed parameters. */
export function changedText(changed: Record<string, [unknown, unknown]> | undefined): string {
  if (!changed) return "";
  return Object.entries(changed)
    .map(([k, [a, b]]) => `${k} ${fmtValue(a)}→${fmtValue(b)}`)
    .join(" · ");
}

/** What a flag reason means, with the thresholds that applied. */
export function criterion(reason: Reason, c: Checks): string {
  const area = Math.PI * (c.diameter / 2) ** 2;
  switch (reason) {
    case "low_pnr":
      return `trace peak-to-noise below ${c.pnr_min} (pure noise scores about 6)`;
    case "duplicate":
      return `a neighbour closer than ${c.diameter} px whose trace correlates above ${c.dup_corr} and has the higher PNR`;
    case "in_pieces":
      return `the largest connected piece holds under ${Math.round(c.piece_frac * 100)} % of the footprint above 10 % of its peak`;
    case "bad_size":
      return `footprint above 10 % of peak outside ${Math.round(c.size_lo * area)}–${Math.round(c.size_hi * area)} px (${c.size_lo}–${c.size_hi} × the cell area)`;
  }
}

// --- small SVG figures -----------------------------------------------------------------

const W = 320;
const H = 72;

function Curve({ points, chosen }: { points: CurvePoint[]; chosen: unknown }) {
  const n = points.length;
  const top = Math.max(1, ...points.map((p) => Math.max(p.good, p.flagged)));
  const x = (i: number) => (n === 1 ? W / 2 : 12 + (i * (W - 24)) / (n - 1));
  const y = (v: number) => 8 + (1 - v / top) * (H - 26);
  const line = (key: "good" | "flagged") => points.map((p, i) => `${x(i)},${y(p[key])}`).join(" ");
  const hit = points.findIndex((p) => fmtValue(p.value) === fmtValue(chosen));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H }} role="img" aria-label="sweep curve">
      {hit >= 0 && <line x1={x(hit)} x2={x(hit)} y1={4} y2={H - 16} style={{ stroke: "var(--muted)" }} strokeDasharray="4 3" />}
      <polyline points={line("good")} fill="none" strokeWidth={2} style={{ stroke: "var(--accent)" }} />
      <polyline points={line("flagged")} fill="none" strokeWidth={2} style={{ stroke: "var(--bad)" }} />
      {points.map((p, i) => (
        <text key={i} x={x(i)} y={H - 3} fontSize={10} textAnchor="middle" style={{ fill: "var(--muted)" }}>
          {fmtValue(p.value)}
        </text>
      ))}
    </svg>
  );
}

function Hist(p: { bins: number[]; series: { counts: number[]; color: string; outline?: boolean }[]; lines: { x: number; color: string }[]; label: string }) {
  const [lo, hi] = [p.bins[0], p.bins[p.bins.length - 1]];
  const px = (v: number) => ((v - lo) / (hi - lo)) * W;
  // each series as a fraction of its own total, so a sparse run and a dense proposal compare
  const norm = (s: number[]) => {
    const total = Math.max(1, s.reduce((a, b) => a + b, 0));
    return s.map((c) => c / total);
  };
  const top = Math.max(1e-9, ...p.series.flatMap((s) => norm(s.counts)));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H }} role="img" aria-label={p.label}>
      {p.series.map((s, k) =>
        norm(s.counts).map((v, i) => (
          <rect
            key={`${k}-${i}`}
            x={px(p.bins[i])}
            width={Math.max(0.5, px(p.bins[i + 1]) - px(p.bins[i]) - 0.5)}
            y={8 + (1 - v / top) * (H - 24)}
            height={(v / top) * (H - 24)}
            style={s.outline ? { fill: "none", stroke: s.color } : { fill: s.color, opacity: 0.55 }}
          />
        )),
      )}
      {p.lines.map((l, i) => (
        <line key={i} x1={px(l.x)} x2={px(l.x)} y1={4} y2={H - 14} style={{ stroke: l.color }} strokeDasharray="4 3" />
      ))}
      <text x={0} y={H - 2} fontSize={10} style={{ fill: "var(--muted)" }}>
        {lo}
      </text>
      <text x={W} y={H - 2} fontSize={10} textAnchor="end" style={{ fill: "var(--muted)" }}>
        {hi}
      </text>
      <text x={W / 2} y={H - 2} fontSize={10} textAnchor="middle" style={{ fill: "var(--muted)" }}>
        {p.label}
      </text>
    </svg>
  );
}

// --- tiles -----------------------------------------------------------------------------

function ScoreCell({ id, rerun, enabled }: { id: string; rerun: string | null; enabled: boolean }) {
  const res = useResource(enabled ? `flags:${id}:${rerun ?? ""}` : null, () =>
    getJSON<Flags>(`/runs/${id}/flags${rerunQuery(rerun)}`),
  );
  if (!enabled) return <>–</>;
  if (res.error) return <span className="error">{res.error}</span>;
  if (!res.data) return <>…</>;
  return (
    <>
      {res.data.good} / <span style={{ color: "var(--bad)" }}>{res.data.flagged}</span>
    </>
  );
}

function RerunsTable({ reruns, hasChecks }: { reruns: Rerun[]; hasChecks: boolean }) {
  const { summary: s, rerun, setRerun } = useRun();
  const shown = (name: string | null) => (
    <button type="button" className="chip" aria-pressed={rerun === name} onClick={() => setRerun(name)}>
      {rerun === name ? "shown" : "show"}
    </button>
  );
  const counts = (c: Record<string, number> | undefined) =>
    c ? `${fmtInt(c.seeds_init ?? 0)} / ${fmtInt(c.seeds_pnr ?? 0)} / ${fmtInt(c.seeds_merged ?? 0)}` : "";
  return (
    <div className="scroll">
      <table className="data">
        <thead>
          <tr>
            <th>name</th>
            <th>changed vs run</th>
            <th>seeds init / PNR / merged</th>
            <th>units</th>
            <th>good / flagged</th>
            <th>wall</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr aria-current={rerun === null ? "true" : undefined}>
            <td>run</td>
            <td className="muted">—</td>
            <td className="mono">{counts(s.counts)}</td>
            <td className="mono">{fmtInt(s.counts.units_final ?? 0)}</td>
            <td className="mono">
              <ScoreCell id={s.id} rerun={null} enabled={hasChecks} />
            </td>
            <td className="mono">{s.wall_total ? fmtDuration(s.wall_total) : ""}</td>
            <td>{shown(null)}</td>
          </tr>
          {reruns.map((r) => (
            <tr key={r.name} aria-current={rerun === r.name ? "true" : undefined}>
              <td>{r.name}</td>
              <td style={{ color: "var(--hot)" }}>
                {r.status === "done" ? changedText(r.changed) || <span className="muted">—</span> : <span className="muted">{r.status}</span>}
                {r.status === "failed" && r.error && <span className="error"> {r.error.join(" ")}</span>}
              </td>
              <td className="mono">{counts(r.counts)}</td>
              <td className="mono">{r.counts ? fmtInt(r.counts.units_final ?? 0) : ""}</td>
              <td className="mono">{r.status === "done" ? <ScoreCell id={s.id} rerun={r.name} enabled={hasChecks} /> : ""}</td>
              <td className="mono">{r.wall_total ? fmtDuration(r.wall_total) : ""}</td>
              <td>{r.status === "done" && shown(r.name)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ProposalTile({ doc }: { doc: Proposal }) {
  const s = doc.score;
  const reasons = REASONS.filter((r) => s[r]).map((r) => `${s[r]} ${REASON_LABEL[r]}`);
  const derived = Object.entries(doc.derived)
    .filter(([k]) => !doc.kept.includes(k))
    .map(([k, v]) => `${k} ${fmtValue(v)}`);
  return (
    <section className="card" aria-label="Proposal">
      <span className="label">Proposal · auto</span>
      <span className="big" style={{ color: "var(--accent)" }}>{fmtInt(s.good)}</span>
      <span className="muted">
        good units · <span style={{ color: "var(--bad)" }}>{s.flagged} flagged</span>
        {reasons.length ? ` (${reasons.join(", ")})` : ""}
        {s.lost ? ` · ${s.lost} of the run's good units lost` : ""}
      </span>
      {doc.run_score && (
        <span className="muted">
          run: {doc.run_score.good} good · {doc.run_score.flagged} flagged
          {!doc.better_than_run && " · nothing better found: the proposal is the run's values"}
        </span>
      )}
      <div className="row">
        <span className="chip" style={{ cursor: "default" }}>
          diameter {doc.diameter} px · {doc.diameter_source === "given" ? "measured" : "from the footprints"}
        </span>
        <span className="chip" style={{ cursor: "default" }}>
          {doc.n_trials} settings · {fmtDuration(doc.wall_s)}
        </span>
      </div>
      <span className="muted" style={{ fontSize: 12 }}>
        {doc.segment
          ? `explored on frames ${fmtInt(doc.segment[0])}–${fmtInt(doc.segment[1])} of ${fmtInt(doc.frames)}, decided on all`
          : `explored on all ${fmtInt(doc.frames)} frames`}
        {derived.length ? ` · from the diameter: ${derived.join(", ")}` : ""}
        {doc.kept.length ? ` · kept from the run: ${doc.kept.join(", ")}` : ""}
      </span>
      {doc.search_error && <ErrorNote message={doc.search_error} />}
    </section>
  );
}

function SweepTile({ doc }: { doc: Proposal }) {
  return (
    <section className="card span4" aria-label="Sweep">
      <span className="label">
        Sweep · best <span style={{ color: "var(--accent)" }}>good</span> and <span style={{ color: "var(--bad)" }}>flagged</span> count at each
        value · dashed = proposed · <span style={{ color: "var(--hot)" }}>at edge</span> = the best value sits at the end of the grid
      </span>
      <div style={{ display: "grid", gridTemplateColumns: "150px 70px 70px 1fr 110px", gap: 12, alignItems: "center", fontSize: 13 }}>
        <span className="label">parameter</span>
        <span className="label">run</span>
        <span className="label">proposed</span>
        <span />
        <span />
        {SEARCHED.map((k) => {
          const a = fmtValue(doc.run_values[k]);
          const b = fmtValue(doc.proposed[k]);
          return [
            <b key={`${k}-n`} className="mono">
              {k}
            </b>,
            <span key={`${k}-a`} className="mono">
              {a}
            </span>,
            <span key={`${k}-b`} className="mono" style={a !== b ? { color: "var(--hot)" } : undefined}>
              {b}
            </span>,
            <div key={`${k}-c`}>{doc.curves[k] ? <Curve points={doc.curves[k]} chosen={doc.proposed[k]} /> : <span className="muted">not searched</span>}</div>,
            <span key={`${k}-e`} className="chip" style={doc.at_edge.includes(k) ? { color: "var(--hot)", cursor: "default" } : { visibility: "hidden" }}>
              at edge
            </span>,
          ];
        })}
      </div>
    </section>
  );
}

function ThresholdsTile() {
  const { summary: s, rerun } = useRun();
  const res = useResource(`pairs:${s.id}:${rerun ?? ""}`, () => getJSON<Pairs>(`/runs/${s.id}/pairs${rerunQuery(rerun)}`));
  return (
    <section className="card span2" aria-label="Thresholds">
      <span className="label">Thresholds against the data · {rerun ?? "run"}</span>
      <Loading res={res} what="the unit pairs">
        {(p) => (
          <>
            <Hist
              bins={p.corr_bins}
              series={[
                { counts: p.far_corr, color: "var(--muted)" },
                { counts: p.near_corr, color: "var(--accent)" },
              ]}
              lines={[{ x: p.dup_corr, color: "var(--hot)" }]}
              label="trace corr of pairs within D (grey: pairs beyond 2 D)"
            />
            <Hist
              bins={p.nn_bins}
              series={[{ counts: p.nn_dist, color: "var(--accent)" }]}
              lines={[
                { x: p.min_dist_frac, color: "var(--hot)" },
                { x: 1, color: "var(--muted)" },
              ]}
              label="nearest neighbour / D"
            />
            <span className="muted" style={{ fontSize: 12 }}>
              {p.n_near} pairs within D, {p.n_near_above} above dup_corr {p.dup_corr}
              {p.far_p99 !== null ? `; far pairs reach ${p.far_p99.toFixed(2)}` : ""}. {p.n_closer} units closer than {p.min_dist_frac} D. A split
              cell shows as a bump of highly correlated near pairs and very short neighbour distances; real neighbours sit near D with low
              correlation.
            </span>
          </>
        )}
      </Loading>
    </section>
  );
}

function FlaggedTile() {
  const { summary: s, rerun, units, maxProj, sel, setSel } = useRun();
  const res = useResource(`flags:${s.id}:${rerun ?? ""}`, () => getJSON<Flags>(`/runs/${s.id}/flags${rerunQuery(rerun)}`));
  const [reason, setReason] = useState<Reason>("low_pnr");
  const flagged = useMemo(() => {
    const ids = new Set(res.data?.reasons[reason] ?? []);
    return units.filter((u: Unit) => ids.has(u.id));
  }, [res.data, reason, units]);
  return (
    <section className="card span2" aria-label="Flagged units">
      <span className="label">Flagged units · {rerun ?? "run"} · click one to select it on every page</span>
      <Loading res={res} what="the unit checks">
        {(f) => (
          <>
            <Segmented
              label="Reason"
              options={REASONS.map((r) => ({ value: r, label: `${REASON_LABEL[r]} ${f.reasons[r]?.length ?? 0}` }))}
              value={reason}
              onChange={setReason}
            />
            <span className="muted" style={{ fontSize: 12 }}>
              {criterion(reason, f.checks)}
            </span>
            <ImageView image={maxProj} units={flagged} contours sel={sel} onPick={setSel} label={`units flagged ${REASON_LABEL[reason]}`} />
          </>
        )}
      </Loading>
    </section>
  );
}

function download(name: string, doc: unknown) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(doc, null, 1)], { type: "application/json" }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

function TuneAgain({ doc }: { doc: Proposal | null }) {
  const { summary: s } = useRun();
  const [open, setOpen] = useState(doc === null);
  const [f, setF] = useState({
    diameter: doc ? String(doc.diameter) : "",
    dup_corr: doc ? String(doc.checks.dup_corr) : "",
    min_dist_frac: doc ? String(doc.min_dist_frac) : "",
    tune_frames: doc ? String(doc.tune_frames) : "",
  });
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const start = () => {
    setBusy(true);
    setErr("");
    postJSON<Job>("/jobs/autotune", { run_dir: s.path, tune_args: f }).then(
      (job) => go({ name: "job", id: job.id }),
      (e: Error) => {
        setErr(e.message);
        setBusy(false);
      },
    );
  };
  const hints: Record<keyof typeof f, string> = {
    diameter: "cell diameter in px; blank = estimated from the run's footprints",
    dup_corr: "a unit with a neighbour within D whose trace correlates above this is a duplicate",
    min_dist_frac: "units closer than this × D always merge; real cells rarely sit closer than 0.4 D",
    tune_frames: `explore on a middle segment of this many frames, decide on all ${fmtInt(s.T)}; 0 = explore on all`,
  };
  return (
    <details open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="label" style={{ cursor: "pointer" }}>
        {doc ? "Auto-tune again" : "Auto-tune"} (diameter, dup_corr, min_dist_frac, tune_frames)
      </summary>
      <div className="bento" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", marginTop: 12 }}>
        {(Object.keys(f) as (keyof typeof f)[]).map((k) => (
          <label key={k} className="field">
            <span className="mono">{k}</span>
            <input type="text" inputMode="decimal" value={f[k]} onChange={(e) => setF({ ...f, [k]: e.target.value })} />
            <span className="muted">{hints[k]}</span>
          </label>
        ))}
      </div>
      {err && <ErrorNote message={err} />}
      <div className="row" style={{ marginTop: 12 }}>
        <span className="spacer" />
        <button className="btn primary" type="button" disabled={busy} onClick={start}>
          {busy ? "Starting…" : "Auto-tune (a few minutes)"}
        </button>
      </div>
    </details>
  );
}

function Window2Form({ metas, groups, doc }: { metas: ParamMeta[]; groups: string[]; doc: Proposal | null }) {
  const { summary: s, rerun, reruns } = useRun();
  const base = useMemo(() => {
    const shown = reruns.find((r) => r.name === rerun);
    return { ...s.params, ...(shown?.params ?? {}) } as Record<string, unknown>;
  }, [s.params, reruns, rerun]);
  const w2 = useMemo(() => metas.filter((m) => m.window2).map((m) => ({ ...m, common: SEARCHED.includes(m.name) })), [metas]);
  const w2groups = groups.filter((g) => w2.some((m) => m.group === g && !m.common));
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [name, setName] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const merged = { ...base, ...values };
  const failing = failingCount(errors);
  const next = `rerun_${reruns.filter((r) => !r.autotune).length + 1}`;
  const start = () => {
    setBusy(true);
    setErr("");
    const params = changedOnly(merged, w2.map((m) => ({ name: m.name, default: s.params[m.name] })));
    postJSON<Job>("/jobs/rerun", { run_dir: s.path, name: name || next, params }).then(
      (job) => go({ name: "job", id: job.id }),
      (e: Error) => {
        setErr(e.message);
        setBusy(false);
      },
    );
  };
  return (
    <section className="card span4" aria-label="Window-2 parameters">
      <span className="label">
        Seed, init and CNMF parameters · prefilled from <b>{rerun ?? "the run"}</b> · a rerun starts from the saved factors, so passes 1 and 2
        are skipped
      </span>
      <ParamsForm
        key={rerun ?? "run"}
        metas={w2}
        groups={w2groups}
        values={merged}
        errors={errors}
        onChange={(n, v, e) => {
          if (v !== undefined) setValues((x) => ({ ...x, [n]: v }));
          setErrors((x) => ({ ...x, [n]: e }));
        }}
      />
      {err && <ErrorNote message={err} />}
      <div className="row">
        <label className="field">
          Rerun name
          <input type="text" value={name} placeholder={next} onChange={(e) => setName(e.target.value)} />
        </label>
        <span className="spacer" />
        {failing > 0 && (
          <span className="error" role="status">
            {failing} {failing === 1 ? "parameter needs" : "parameters need"} fixing
          </span>
        )}
        <button className="btn" type="button" onClick={() => download(`params_${rerun ?? s.name}.json`, merged)}>
          Download params.json
        </button>
        <button className="btn primary" type="button" disabled={busy || failing > 0} onClick={start}>
          {busy ? "Starting…" : "Rerun window 2"}
        </button>
      </div>
      <TuneAgain doc={doc} />
    </section>
  );
}

export function Tune() {
  const { summary: s, reruns, refreshReruns } = useRun();
  const proposal = useResource(`autotune:${s.id}:${reruns.length}`, () =>
    getJSON<Proposal>(`/runs/${s.id}/autotune`).catch((e: unknown) => {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }),
  );
  const defaults = useResource("params", () => getJSON<{ groups: string[]; params: ParamMeta[] }>("/params/defaults"));
  const doc = proposal.data ?? null;
  const running = reruns.some((r) => r.status === "running");
  return (
    <div className="bento">
      {proposal.error && (
        <section className="card span4">
          <ErrorNote message={proposal.error} />
        </section>
      )}
      {doc ? (
        <ProposalTile doc={doc} />
      ) : (
        <section className="card" aria-label="Proposal">
          <span className="label">Proposal</span>
          <span className="big">–</span>
          <span className="muted">no auto-tune of this run yet; start one below, or rerun window 2 with your own parameters</span>
        </section>
      )}
      <section className="card span3" aria-label="Reruns">
        <div className="row">
          <span className="label">Reruns · window 2 on this run's factors</span>
          <span className="spacer" />
          {running && (
            <button className="btn" type="button" onClick={refreshReruns}>
              Refresh
            </button>
          )}
        </div>
        <RerunsTable reruns={reruns} hasChecks={doc !== null} />
      </section>
      {doc && <SweepTile doc={doc} />}
      {doc && (
        <>
          <ThresholdsTile />
          <FlaggedTile />
        </>
      )}
      <Loading res={defaults} what="parameters">
        {(d) => <Window2Form metas={d.params} groups={d.groups} doc={doc} />}
      </Loading>
    </div>
  );
}
