import { vi } from "vitest";

// Browser tests talk to the in-page router through main's api.ts, as the browser build does.
vi.mock("../../transport", async () => {
  const { browserFetch } = await import("../fetch");
  const { openEvents } = await import("../events");
  return { transportFetch: browserFetch, openEvents };
});
