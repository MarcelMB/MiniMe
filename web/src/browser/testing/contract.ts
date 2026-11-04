import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect } from "vitest";
import { readNpy } from "../npy";

export type ContractEntry = { request: string; body: unknown; status: number; file: string };
export type Contract = {
  entries: ContractEntry[];
  find(request: string): { entry: ContractEntry; json: () => unknown; npy: () => ReturnType<typeof readNpy> };
};

export function loadContract(dir: string): Contract {
  const entries = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8")) as ContractEntry[];
  return {
    entries,
    find(request) {
      const entry = entries.find((e) => e.request === request);
      if (!entry) throw new Error(`no ${request} in ${dir}`);
      return {
        entry,
        json: () => JSON.parse(readFileSync(join(dir, entry.file), "utf8")),
        npy: () => readNpy(new Uint8Array(readFileSync(join(dir, entry.file)))),
      };
    },
  };
}

/** Deep equality with numbers within `tol` (relative to max(1, |b|)); keys in `skip` ignored. */
export function expectClose(a: unknown, b: unknown, tol = 1e-5, skip: string[] = [], path = "$"): void {
  if (typeof b === "number" && typeof a === "number") {
    if (Number.isNaN(b)) expect(Number.isNaN(a), path).toBe(true);
    else expect(Math.abs(a - b), path).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)));
    return;
  }
  if (Array.isArray(b)) {
    expect(Array.isArray(a), path).toBe(true);
    expect((a as unknown[]).length, `${path}.length`).toBe(b.length);
    b.forEach((v, i) => {
      expectClose((a as unknown[])[i], v, tol, skip, `${path}[${i}]`);
    });
    return;
  }
  if (b && typeof b === "object") {
    const keys = Object.keys(b).filter((k) => !skip.includes(k));
    expect(
      Object.keys(a as object)
        .filter((k) => !skip.includes(k))
        .sort(),
      path,
    ).toEqual(keys.sort());
    for (const k of keys)
      expectClose((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], tol, skip, `${path}.${k}`);
    return;
  }
  expect(a, path).toEqual(b);
}
