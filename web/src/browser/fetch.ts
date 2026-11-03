import { HttpError, type Router, router } from "./router";
import "./routes";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

/** `fetch` for `/api/...` paths, answered by `r` in the page: main's status codes and bodies. */
export function createFetch(r: Router): (input: string, init?: RequestInit) => Promise<Response> {
  return async (input, init = {}) => {
    const url = new URL(input, "http://local");
    const path = url.pathname.replace(/^\/api/, "");
    const method = (init.method ?? "GET").toUpperCase();
    const body = typeof init.body === "string" && init.body ? JSON.parse(init.body) : null;
    try {
      const rep = await r.handle({ method, path, query: url.searchParams, body });
      if (rep.computing !== undefined) return json({ elapsed_s: rep.computing }, 202);
      if (rep.array) {
        const a = rep.array;
        const bytes = new Uint8Array(a.data.buffer, a.data.byteOffset, a.data.byteLength).slice();
        return new Response(bytes, {
          status: rep.status ?? 200,
          headers: { "X-Shape": a.shape.join(","), "X-Dtype": a.dtype, "Content-Type": "application/octet-stream" },
        });
      }
      return json(rep.json ?? null, rep.status ?? 200);
    } catch (e) {
      if (e instanceof HttpError) return json({ detail: e.message }, e.status);
      return json({ detail: e instanceof Error ? e.message : String(e) }, 500);
    }
  };
}

export const browserFetch = createFetch(router);
