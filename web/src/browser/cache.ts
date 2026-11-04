import type { Dir } from "./dir";
import { readNpy, writeNpy } from "./npy";

/** Derived views of one run, kept outside the run folder (main's ~/.cache/minime/<id>). */
export class RunCache {
  constructor(private dir: Dir) {}
  has(name: string): Promise<boolean> {
    return this.dir.file(name).then(
      () => true,
      () => false,
    );
  }
  async loadJSON<T>(name: string, dflt: T): Promise<T> {
    return this.dir.file(name).then(
      async (f) => JSON.parse(await f.text()) as T,
      () => dflt,
    );
  }
  saveJSON(name: string, v: unknown): Promise<void> {
    return this.dir.write(name, JSON.stringify(v));
  }
  async loadNpy(name: string): Promise<ReturnType<typeof readNpy>> {
    return readNpy(new Uint8Array(await (await this.dir.file(name)).arrayBuffer()));
  }
  saveNpy(name: string, shape: number[], data: Float32Array): Promise<void> {
    return this.dir.write(name, writeNpy(shape, data));
  }
}
