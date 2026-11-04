import { RawVideo } from "../avi";
import type { Dir } from "../dir";
import { fpsFromFolder } from "../fps";
import { HttpError, router } from "../router";
import { state } from "../state";
import { parentVpath } from "../vpath";

/** `routes_jobs._info`: the raw folder at `path` and its videos. 404 without videos, 400 for
 * videos that cannot be read; a folder that needs reconnecting stays the 403 `resolve` raised. */
export async function openRaw(path: string): Promise<{ raw: Dir; v: RawVideo }> {
  const raw = await state.folders.resolve(path);
  const v = await RawVideo.open(raw).catch((e) => {
    throw e instanceof HttpError ? e : new HttpError(400, `cannot read the videos in ${path}: ${e}`);
  });
  return { raw, v };
}

/** `routes_jobs.raw_info`: what New run shows about a raw folder. `path` stays the vpath it was
 * asked with. */
router.add("GET", "/raw/info", async (q) => {
  const path = q.query.get("path") ?? "";
  const { raw, v } = await openRaw(path);
  const parent = parentVpath(path);
  return {
    json: {
      path,
      segments: v.names.length,
      frames: v.frames,
      height: v.height,
      width: v.width,
      fps: await fpsFromFolder(raw, parent ? await state.folders.resolve(parent).catch(() => null) : null),
      bytes: v.bytes,
      names: v.names,
      skipped: v.skipped,
      naming_rule: RawVideo.NAMING_RULE,
    },
  };
});
