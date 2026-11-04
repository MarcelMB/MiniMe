import { pyFloat } from "../num";
import { HttpError, router } from "../router";
import { state } from "../state";

router.add("GET", "/folders", async () => {
  const roots = await state.folders.list();
  return {
    json: {
      folders: roots.map((r) => ({ path: r.name, kind: "run", raw: r.raw, time: r.time / 1e3 })),
    },
  };
});

/** The folder's permission asked for again. A browser shows its prompt only for a request that
 * comes from a click; the page's `FolderPicker` does this itself (`Folders.pick`) when a folder
 * that is a root already is picked again. */
router.add("POST", "/folders/reconnect", async (q) => {
  const name = (q.body as { name?: unknown } | null)?.name;
  if (typeof name !== "string") throw new HttpError(422, "name must be text");
  return { json: { ok: await state.folders.reconnect(name) } };
});

router.add("GET", "/fs", () => {
  throw new HttpError(404, "the browser app picks folders with the system dialog");
});

router.add("GET", "/runs", async (q) => {
  const roots = await state.folders.list();
  const root = q.query.get("root") || roots[0]?.name;
  if (!root) return { json: { root: "", runs: [] } };
  const out = [];
  for (const v of await state.registry.find(root)) {
    try {
      out.push(await state.registry.listing(await state.registry.open(v)));
    } catch (e) {
      out.push({ path: v, name: v.split("/").pop(), error: e instanceof Error ? e.message : String(e) });
    }
  }
  return { json: { root, runs: out } };
});

router.add("GET", "/runs/:rid/summary", async (_q, p) => ({
  json: await state.registry.summary(await state.registry.get(p.rid)),
}));

router.add("POST", "/runs/:rid/fps", async (q, p) => {
  const raw = (q.body as { fps?: unknown } | null)?.fps;
  const fps = typeof raw === "string" && raw.trim() !== "" ? Number(raw) : raw; // a numeric string is fine, as in pydantic
  if (typeof fps !== "number" || Number.isNaN(fps)) throw new HttpError(422, "fps must be a number");
  // io.check_fps
  if (!(fps > 0 && fps <= 1000)) throw new HttpError(400, `frame rate must be in (0, 1000] Hz, got ${pyFloat(fps)}`);
  const r = await state.registry.get(p.rid);
  await state.registry.setFps(r, fps);
  return { json: { fps: await state.registry.fps(r) } };
});

// Tune (reruns, auto-tune, unit checks) is not part of the browser app: no reruns, and 404 for the rest.
router.add("GET", "/runs/:rid/reruns", () => ({ json: { reruns: [] } }));
for (const path of ["/runs/:rid/autotune", "/runs/:rid/flags", "/runs/:rid/pairs"])
  router.add("GET", path, () => {
    throw new HttpError(404, "auto-tune is not part of the browser app");
  });
for (const path of ["/jobs/rerun", "/jobs/autotune"])
  router.add("POST", path, () => {
    throw new HttpError(404, "auto-tune and reruns are not part of the browser app");
  });
