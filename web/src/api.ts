export type DType = "uint8" | "int32" | "float32";
export type NDArray = { shape: number[]; dtype: DType; data: Uint8Array | Int32Array | Float32Array };

/** The back end answered 202: the result is being computed. */
export class Computing extends Error {
  constructor(public elapsed: number) {
    super("computing");
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

const KEY = "minime-token";
let token = "";

export function readToken(search: string, store: Pick<Storage, "getItem" | "setItem">): string {
  const t = new URLSearchParams(search).get("token");
  try {
    if (t) {
      store.setItem(KEY, t);
      return t;
    }
    return store.getItem(KEY) ?? "";
  } catch {
    return t ?? ""; // sessionStorage blocked: the URL token still works for this page load
  }
}

export function initToken(): void {
  try {
    token = readToken(window.location.search, window.sessionStorage);
  } catch {
    token = new URLSearchParams(window.location.search).get("token") ?? "";
  }
}

export function tokenQuery(): string {
  return `token=${encodeURIComponent(token)}`;
}

const CTORS = { uint8: Uint8Array, int32: Int32Array, float32: Float32Array } as const;

export function decodeArray(buf: ArrayBuffer, shape: string, dtype: string): NDArray {
  const ctor = CTORS[dtype as DType];
  if (!ctor) throw new Error(`unknown dtype ${dtype}`);
  const dims = shape ? shape.split(",").map(Number) : [];
  return { shape: dims, dtype: dtype as DType, data: new ctor(buf) };
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = { "X-Token": token, ...(init.headers as Record<string, string> | undefined) };
  const r = await fetch(`/api${path}`, { ...init, headers });
  if (r.status === 202) {
    const b = await r.json();
    throw new Computing(Number(b.elapsed_s ?? 0));
  }
  if (!r.ok) {
    const text = await r.text();
    let msg = text || r.statusText;
    try {
      const j = JSON.parse(text);
      msg = j.detail ?? j.error ?? msg;
    } catch {
      /* plain text body */
    }
    if (r.status === 401) {
      msg = "This page's access key is no longer valid. Open the link that `minime serve` printed again.";
    }
    throw new ApiError(r.status, String(msg));
  }
  return r;
}

export async function getJSON<T>(path: string): Promise<T> {
  return (await request(path)).json() as Promise<T>;
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json() as Promise<T>;
}

export async function getArray(path: string): Promise<NDArray> {
  const r = await request(path);
  return decodeArray(await r.arrayBuffer(), r.headers.get("X-Shape") ?? "", r.headers.get("X-Dtype") ?? "");
}

export async function postArray(path: string, body: unknown): Promise<NDArray> {
  const r = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return decodeArray(await r.arrayBuffer(), r.headers.get("X-Shape") ?? "", r.headers.get("X-Dtype") ?? "");
}
