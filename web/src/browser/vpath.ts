/** Virtual paths: `<root name>/<relative path>`. The first segment names a folder root (see
 * `Folders`); every path the router returns is one, and every path it receives is resolved as one. */
export function splitVpath(v: string): { root: string; rel: string } {
  const i = v.indexOf("/");
  return i < 0 ? { root: v, rel: "" } : { root: v.slice(0, i), rel: v.slice(i + 1) };
}

export const joinVpath = (...parts: string[]): string => parts.filter(Boolean).join("/");

/** The vpath one level up, or null for a root. */
export function parentVpath(v: string): string | null {
  const i = v.lastIndexOf("/");
  return i < 0 ? null : v.slice(0, i);
}
