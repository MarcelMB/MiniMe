import type { NDArray } from "../api";

export class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export type Req = { method: string; path: string; query: URLSearchParams; body: unknown };
/** One answer: JSON, an array, or "still computing" (202 with the seconds so far). */
export type Reply = { status?: number; json?: unknown; array?: NDArray; computing?: number };
type Handler = (req: Req, p: Record<string, string>) => Reply | Promise<Reply>;

/** Main's `/api` routes, matched in the order they were added (first match wins). */
export class Router {
  private routes: { method: string; re: RegExp; keys: string[]; h: Handler }[] = [];

  add(method: string, pattern: string, h: Handler): void {
    const keys: string[] = [];
    const re = new RegExp(
      `^${pattern.replace(/:([a-z]+)/g, (_m, k: string) => {
        keys.push(k);
        return "([^/]+)";
      })}$`,
    );
    this.routes.push({ method, re, keys, h });
  }

  async handle(req: Req): Promise<Reply> {
    for (const r of this.routes) {
      if (r.method !== req.method) continue;
      const m = r.re.exec(req.path);
      if (!m) continue;
      const p: Record<string, string> = {};
      r.keys.forEach((k, i) => {
        p[k] = decodeURIComponent(m[i + 1]);
      });
      return r.h(req, p);
    }
    throw new HttpError(404, "Not Found");
  }
}

export const router = new Router();
