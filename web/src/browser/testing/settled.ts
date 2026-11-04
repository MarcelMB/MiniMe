import { Computing } from "../../api";

/** Overviews are computed on the first request (a 202 until done): ask again until they are there. */
export async function settled<T>(get: () => Promise<T>): Promise<T> {
  for (;;)
    try {
      return await get();
    } catch (e) {
      if (!(e instanceof Computing)) throw e;
      await new Promise((r) => setTimeout(r, 20));
    }
}
