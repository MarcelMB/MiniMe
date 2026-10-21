import { useState } from "react";
import type { ParamMeta } from "../types";
import { parseValue } from "./params";

type Raw = string | boolean | [string, string];

function toRaw(m: ParamMeta, v: unknown): Raw {
  if (m.type === "bool") return Boolean(v);
  if (m.type.startsWith("tuple")) {
    const [a, b] = (v as (number | null)[]) ?? [null, null];
    return [a === null ? "" : String(a), b === null ? "" : String(b)];
  }
  return String(v);
}

function Field({
  m,
  value,
  onChange,
}: {
  m: ParamMeta;
  value: unknown;
  onChange: (v: unknown, err: string) => void;
}) {
  const [raw, setRaw] = useState<Raw>(() => toRaw(m, value));
  const [err, setErr] = useState("");
  const set = (r: Raw) => {
    setRaw(r);
    try {
      onChange(parseValue(m.type, r), "");
      setErr("");
    } catch (e) {
      setErr((e as Error).message);
      onChange(value, (e as Error).message);
    }
  };
  const id = `p-${m.name}`;
  return (
    <div className="field">
      <label htmlFor={id} className="mono">
        {m.name}
      </label>
      {m.type === "bool" ? (
        <input
          id={id}
          type="checkbox"
          role="switch"
          checked={raw as boolean}
          onChange={(e) => set(e.target.checked)}
        />
      ) : m.type.startsWith("tuple") ? (
        <div className="row">
          <input
            id={id}
            type="text"
            aria-label={`${m.name} first`}
            value={(raw as [string, string])[0]}
            onChange={(e) =>
              set([e.target.value, (raw as [string, string])[1]])
            }
          />
          <input
            type="text"
            aria-label={`${m.name} second`}
            value={(raw as [string, string])[1]}
            onChange={(e) =>
              set([(raw as [string, string])[0], e.target.value])
            }
          />
        </div>
      ) : (
        <input
          id={id}
          type="text"
          inputMode="decimal"
          value={raw as string}
          onChange={(e) => set(e.target.value)}
        />
      )}
      <span className="muted">{m.help}</span>
      {err && <span className="error">{err}</span>}
    </div>
  );
}

export function ParamsForm(p: {
  metas: ParamMeta[];
  groups: string[];
  values: Record<string, unknown>;
  onChange: (name: string, v: unknown, err: string) => void;
}) {
  const common = p.metas.filter((m) => m.common);
  return (
    <>
      <div
        className="bento"
        style={{ gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}
      >
        {common.map((m) => (
          <Field
            key={m.name}
            m={m}
            value={p.values[m.name]}
            onChange={(v, e) => p.onChange(m.name, v, e)}
          />
        ))}
      </div>
      <details>
        <summary className="label" style={{ cursor: "pointer" }}>
          All parameters
        </summary>
        {p.groups.map((g) => (
          <fieldset key={g} style={{ border: 0, padding: 0, marginTop: 16 }}>
            <legend className="label">{g}</legend>
            <div
              className="bento"
              style={{
                gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              }}
            >
              {p.metas
                .filter((m) => m.group === g && !m.common)
                .map((m) => (
                  <Field
                    key={m.name}
                    m={m}
                    value={p.values[m.name]}
                    onChange={(v, e) => p.onChange(m.name, v, e)}
                  />
                ))}
            </div>
          </fieldset>
        ))}
      </details>
    </>
  );
}
