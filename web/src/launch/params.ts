function num(s: string, what: string): number {
  const v = Number(s.trim());
  if (s.trim() === "" || !Number.isFinite(v))
    throw new Error(`${what}: enter a number`);
  return v;
}

/** Value for a ``Params`` field from the form; ``type`` is the dataclass annotation. */
export function parseValue(
  type: string,
  raw: string | boolean | [string, string],
): unknown {
  if (type === "bool") return Boolean(raw);
  if (type.startsWith("tuple")) {
    const [a, b] = raw as [string, string];
    const nullable = type.includes("None");
    const one = (s: string) =>
      nullable && s.trim() === "" ? null : num(s, "each value");
    return [one(a), one(b)];
  }
  const s = String(raw);
  if (type === "int") {
    const v = num(s, "value");
    if (!Number.isInteger(v)) throw new Error("value: enter a whole number");
    return v;
  }
  if (type === "float | str")
    return s.trim() === "auto" ? "auto" : num(s, "value ('auto' or a number)");
  return num(s, "value");
}

export function changedOnly(
  values: Record<string, unknown>,
  metas: { name: string; default: unknown }[],
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const m of metas) {
    if (
      m.name in values &&
      JSON.stringify(values[m.name]) !== JSON.stringify(m.default)
    )
      out[m.name] = values[m.name];
  }
  return out;
}

export function defaultRunName(now: Date): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `caanfast_${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())}_${p(now.getHours())}${p(now.getMinutes())}`;
}

export type ProgressEvent = {
  stage: string;
  event: "start" | "end" | "error";
  t: number;
  seconds?: number;
};
export type StageRow = {
  stage: string;
  seconds: number | null;
  running: boolean;
  failed: boolean;
};

export function stageRows(events: ProgressEvent[]): StageRow[] {
  const rows = new Map<string, StageRow>();
  for (const e of events) {
    if (e.event === "start")
      rows.set(e.stage, {
        stage: e.stage,
        seconds: null,
        running: true,
        failed: false,
      });
    else
      rows.set(e.stage, {
        stage: e.stage,
        seconds: e.seconds ?? null,
        running: false,
        failed: e.event === "error",
      });
  }
  return [...rows.values()];
}
