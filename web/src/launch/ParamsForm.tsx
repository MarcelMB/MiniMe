import { useEffect, useRef, useState } from "react";
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
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  // a field that disappears must not keep blocking Start
  useEffect(() => () => onChangeRef.current(undefined, ""), []);
  const aria = {
    "aria-invalid": !!err,
    "aria-describedby": err ? `${id}-help ${id}-err` : `${id}-help`,
  };
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
          {...aria}
          checked={raw as boolean}
          onChange={(e) => set(e.target.checked)}
        />
      ) : m.type.startsWith("tuple") ? (
        <div className="row">
          <input
            id={id}
            type="text"
            {...aria}
            aria-label={`${m.name} first`}
            value={(raw as [string, string])[0]}
            onChange={(e) =>
              set([e.target.value, (raw as [string, string])[1]])
            }
          />
          <input
            type="text"
            {...aria}
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
          {...aria}
          value={raw as string}
          onChange={(e) => set(e.target.value)}
        />
      )}
      <span className="muted" id={`${id}-help`}>
        {m.help}
      </span>
      {err && (
        <span className="error" id={`${id}-err`} role="alert">
          {err}
        </span>
      )}
    </div>
  );
}

export function ParamsForm(p: {
  metas: ParamMeta[];
  groups: string[];
  values: Record<string, unknown>;
  errors: Record<string, string>;
  onChange: (name: string, v: unknown, err: string) => void;
}) {
  const hiddenInvalid = p.metas.some((m) => !m.common && p.errors[m.name]);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (hiddenInvalid) setOpen(true);
  }, [hiddenInvalid]);
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
      <details open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
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
