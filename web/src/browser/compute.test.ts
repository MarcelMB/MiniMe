import { beforeEach, expect, it } from "vitest";
import { computing, resetComputing } from "./compute";

beforeEach(resetComputing);

it("starts once, reports computing, then forgets failures with a 500", async () => {
  let calls = 0;
  let fail!: (e: Error) => void;
  const fn = () => {
    calls++;
    return new Promise<void>((_ok, no) => {
      fail = no;
    });
  };
  expect(computing("k", fn).computing).toBeGreaterThanOrEqual(0);
  expect(computing("k", fn).computing).toBeGreaterThanOrEqual(0);
  expect(calls).toBe(1);
  fail(new Error("no frames"));
  await new Promise((r) => setTimeout(r, 0));
  expect(() => computing("k", fn)).toThrow("no frames");
  computing("k", fn);
  expect(calls).toBe(2);
});
