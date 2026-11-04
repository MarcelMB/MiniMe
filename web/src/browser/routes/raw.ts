import { RawVideo } from "../avi";
import { fpsFromFolder } from "../fps";
import { HttpError, router } from "../router";
import { state } from "../state";
import { parentVpath } from "../vpath";

/** `routes_jobs.raw_info`: what New run shows about a raw folder. `path` stays the vpath it was
 * asked with. */
router.add("GET", "/raw/info", async (q) => {
  const path = q.query.get("path") ?? "";
  const raw = await state.folders.resolve(path);
  const v = await RawVideo.open(raw).catch((e) => {
    throw e instanceof HttpError ? e : new HttpError(400, `cannot read the videos in ${path}: ${e}`);
  });
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
