import { mkdirSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { getJSON, postJSON } from "../api";
import { type Dir, FileListDir, HandleDir } from "./dir";
import { Folders } from "./folders";
import { memoryKV } from "./kv";
import { Registry } from "./runs";
import { configure } from "./state";
import { NodeDir } from "./testing/nodeDir";

const tmp = (name: string) => {
  const p = join(mkdtempSync(join(tmpdir(), "f-")), name);
  mkdirSync(join(p, "raw"), { recursive: true });
  return new NodeDir(p);
};

describe("Folders", () => {
  it("names roots uniquely and resolves vpaths", async () => {
    const f = new Folders(memoryKV(), async () => null);
    expect(await f.add(tmp("synth"), "picked")).toBe("synth");
    expect(await f.add(tmp("synth"), "picked")).toBe("synth (2)");
    expect((await f.resolve("synth (2)/raw")).name).toBe("raw");
    await expect(f.resolve("other/raw")).rejects.toMatchObject({ status: 404 });
    expect((await f.list()).map((r) => r.name)).toEqual(["synth (2)", "synth"]);
  });

  it("answers 403 when permission is gone", async () => {
    const d = tmp("s");
    d.permission = async () => false;
    const f = new Folders(memoryKV(), async () => null);
    await f.add(d, "picked");
    await expect(f.resolve("s/raw")).rejects.toMatchObject({ status: 403 });
    // the UI sees it too, with the message that tells the user what to click
    configure({ folders: f, registry: new Registry(f, memoryKV()) });
    await expect(getJSON("/runs?root=s")).rejects.toMatchObject({
      status: 403,
      message: expect.stringMatching(/^Reconnect the folder/),
    });
  });

  it("lists a root that needs reconnecting, and POST /folders/reconnect asks for the permission again", async () => {
    const d: Dir = tmp("s");
    const asked: [string, boolean][] = [];
    let granted = false;
    d.permission = async (mode, ask) => {
      asked.push([mode, ask]);
      if (ask) granted = true;
      return granted;
    };
    const f = new Folders(memoryKV(), async () => null);
    await f.add(d, "picked");
    configure({ folders: f, registry: new Registry(f, memoryKV()) });
    // Home still lists it, so "Change folder" can pick it again
    const { folders } = await getJSON<{ folders: { path: string; kind: string }[] }>("/folders");
    expect(folders.map((x) => [x.path, x.kind])).toEqual([["s", "run"]]);
    await expect(getJSON("/runs?root=s")).rejects.toMatchObject({
      status: 403,
      message: expect.stringMatching(/^Reconnect the folder s\b/),
    });
    expect(asked.every(([, ask]) => !ask)).toBe(true); // reading never prompts
    expect(await postJSON("/folders/reconnect", { name: "s" })).toEqual({ ok: true });
    expect(asked.at(-1)).toEqual(["readwrite", true]);
    expect(await getJSON("/runs?root=s")).toMatchObject({ root: "s", runs: [] });
    expect(await postJSON("/folders/reconnect", { name: "unknown" })).toEqual({ ok: false });
    await expect(postJSON("/folders/reconnect", {})).rejects.toMatchObject({ status: 422 });
  });

  it("says a file-list root was opened read-only when a run would write into it", async () => {
    const file = new File(["{}"], "params.json");
    Object.defineProperty(file, "webkitRelativePath", { value: "sess/params.json" });
    const f = new Folders(memoryKV(), async () => null);
    const name = await f.add(FileListDir.fromFiles([file]), "files");
    expect(name).toBe("sess");
    expect((await f.resolve(name)).name).toBe("sess");
    await expect(f.resolve(name, "readwrite")).rejects.toMatchObject({
      status: 403,
      message: "sess was opened read-only; runs from it go to Browser storage",
    });
  });

  it("lists OPFS as Browser storage", async () => {
    const f = new Folders(memoryKV(), async () => tmp("runs"));
    expect((await f.list()).map((r) => [r.name, r.kind])).toContainEqual(["Browser storage", "opfs"]);
  });

  it("keeps a picked folder's handle (not the Dir) across reloads, with its last raw folder", async () => {
    const handle = { name: "sess", kind: "directory" } as unknown as FileSystemDirectoryHandle;
    const kv = memoryKV();
    const f = new Folders(kv, async () => null);
    const name = await f.add(new HandleDir(handle), "picked");
    await f.touch(name, "sess/raw");
    for (const k of await kv.keys()) {
      const stored = await kv.get<{ handle: unknown; dir?: unknown }>(k);
      expect(stored?.handle).toBe(handle);
      expect(stored?.dir).toBeUndefined();
    }
    const again = new Folders(kv, async () => null); // a new page load
    const [root] = await again.list();
    expect(root).toMatchObject({ name: "sess", kind: "picked", raw: "sess/raw" });
    expect((root.dir as HandleDir).handle).toBe(handle);
  });

  describe("pick (a folder the system dialog just returned)", () => {
    /** A handle that is the same entry as every handle made with the same `id`; its permission is
     * "prompt" until asked for, and `asked` records every query and request. */
    const handles = () => {
      const asked: [string, boolean][] = [];
      let granted = false;
      const make = (id: string) =>
        ({
          name: id,
          kind: "directory",
          id,
          isSameEntry: async (o: { id?: string }) => o.id === id,
          queryPermission: async ({ mode }: { mode: string }) => {
            asked.push([mode, false]);
            return granted ? "granted" : "prompt";
          },
          requestPermission: async ({ mode }: { mode: string }) => {
            asked.push([mode, true]);
            granted = true;
            return "granted";
          },
        }) as unknown as FileSystemDirectoryHandle;
      return { make, asked };
    };

    it("adds a new folder as a root", async () => {
      const { make } = handles();
      const f = new Folders(memoryKV(), async () => null);
      expect(await f.pick(new HandleDir(make("a")))).toBe("a");
      expect(await f.pick(new HandleDir(make("b")))).toBe("b");
      expect((await f.list()).map((r) => r.name).sort()).toEqual(["a", "b"]);
    });

    it("gives a folder that is a root already its permission again instead of adding it twice", async () => {
      const { make, asked } = handles();
      const kv = memoryKV();
      await new Folders(kv, async () => null).pick(new HandleDir(make("sess")));
      const again = new Folders(kv, async () => null); // a new page load: the stored handle needs reconnecting
      await expect(again.resolve("sess", "readwrite")).rejects.toMatchObject({ status: 403 });
      expect(asked.every(([, ask]) => !ask)).toBe(true); // resolving never prompts
      expect(await again.pick(new HandleDir(make("sess")))).toBe("sess");
      expect(asked.at(-1)).toEqual(["readwrite", true]);
      expect((await again.resolve("sess", "readwrite")).name).toBe("sess");
      expect((await again.list()).map((r) => r.name)).toEqual(["sess"]);
    });

    it("does not take a disk folder for a handle that is already a root", async () => {
      const { make } = handles();
      const f = new Folders(memoryKV(), async () => null);
      await f.pick(new HandleDir(make("x")));
      expect(await f.pick(tmp("x"))).toBe("x (2)");
    });
  });
});
