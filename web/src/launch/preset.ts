/** A set of parameters New run starts from instead of the defaults (the browser build's demo
 * settings), handed over from another page for the next New run only. */
export type Preset = {
  /** What New run says is applied ("Demo settings"). */
  name: string;
  /** Where the values come from, shown under the name. */
  note: string;
  /** Only the parameters that differ from the defaults. */
  values: Record<string, unknown>;
};

const KEY = "minime-new-run-preset";

/** Hand `p` to the next New run (this tab only). */
export function offerPreset(p: Preset): void {
  try {
    window.sessionStorage.setItem(KEY, JSON.stringify(p));
  } catch {
    /* storage blocked: New run starts from the defaults */
  }
}

/** The preset handed over, if any (it stays until `clearPreset`). */
export function readPreset(): Preset | null {
  try {
    const s = window.sessionStorage.getItem(KEY);
    return s ? (JSON.parse(s) as Preset) : null;
  } catch {
    return null;
  }
}

export function clearPreset(): void {
  try {
    window.sessionStorage.removeItem(KEY);
  } catch {
    /* nothing to clear */
  }
}
