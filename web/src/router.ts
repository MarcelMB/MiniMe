import { useEffect, useState } from "react";

export const PAGES = ["overview", "frames", "motion", "compression", "seeds", "units", "traces", "tune"] as const;
export type Page = (typeof PAGES)[number];

export type Route =
  | { name: "home" }
  | { name: "new" }
  | { name: "job"; id: string }
  | { name: "run"; id: string; page: Page };

export function parseHash(h: string): Route {
  const parts = h.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (parts[0] === "new") return { name: "new" };
  if (parts[0] === "job" && parts[1]) return { name: "job", id: parts[1] };
  if (parts[0] === "run" && parts[1]) {
    const page = (PAGES as readonly string[]).includes(parts[2] ?? "") ? (parts[2] as Page) : "overview";
    return { name: "run", id: parts[1], page };
  }
  return { name: "home" };
}

export function href(r: Route): string {
  switch (r.name) {
    case "home":
      return "#/";
    case "new":
      return "#/new";
    case "job":
      return `#/job/${r.id}`;
    case "run":
      return `#/run/${r.id}/${r.page}`;
  }
}

export function go(r: Route): void {
  window.location.hash = href(r);
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
