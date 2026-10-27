import type { ReactNode } from "react";
import type { Resource } from "./poll";

export function Toggle(p: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <label className="switch" title={p.title}>
      <input
        type="checkbox"
        role="switch"
        checked={p.checked}
        disabled={p.disabled}
        onChange={(e) => p.onChange(e.target.checked)}
      />
      {p.label}
    </label>
  );
}

export function Slider(p: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  title?: string;
  format?: (v: number) => string;
}) {
  return (
    <label className="slider" title={p.title}>
      <span>{p.label}</span>
      <input
        type="range"
        min={p.min}
        max={p.max}
        step={p.step}
        value={p.value}
        disabled={p.disabled}
        onChange={(e) => p.onChange(Number(e.target.value))}
      />
      <span className="mono">{p.format ? p.format(p.value) : p.value}</span>
    </label>
  );
}

export type SegOption<T extends string> = { value: T; label: string; disabled?: boolean; title?: string };

export function Segmented<T extends string>(p: {
  label: string;
  options: SegOption<T>[];
  value: T;
  onChange: (v: T) => void;
  glass?: boolean;
}) {
  return (
    <div className={p.glass ? "seg glass" : "seg"} role="group" aria-label={p.label}>
      {p.options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={o.value === p.value}
          disabled={o.disabled}
          title={o.title}
          onClick={() => p.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <p role="alert" className="error">
      {message}
    </p>
  );
}

export function ComputingNote({ elapsed, what }: { elapsed: number; what: string }) {
  return (
    <div className="computing" aria-live="polite">
      <span>
        Computing {what}… {elapsed.toFixed(0)} s
      </span>
      <div className="bar">
        <span />
      </div>
    </div>
  );
}

/** Render ``children(data)`` once a resource is loaded; error / computing notes before. */
export function Loading<T>(p: { res: Resource<T>; what: string; children: (data: T) => ReactNode }) {
  if (p.res.error) return <ErrorNote message={p.res.error} />;
  if (p.res.computing) return <ComputingNote elapsed={p.res.elapsed} what={p.what} />;
  if (p.res.data === undefined) return <p className="muted">Loading {p.what}…</p>;
  return <>{p.children(p.res.data)}</>;
}
