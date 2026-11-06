import { useSyncExternalStore } from "react";

/** The colour scheme: the system's (which may follow the time of day) unless the page's toggle chose
 * the other one. A choice is kept in this browser; choosing the system's scheme again clears it, so
 * the page follows the system once more. */
export type Scheme = "light" | "dark";

const KEY = "minime-scheme";
const listeners = new Set<() => void>();
let chosen: Scheme | null = null;

function darkQuery(): MediaQueryList | null {
  return typeof window.matchMedia === "function" ? window.matchMedia("(prefers-color-scheme: dark)") : null;
}

export function systemScheme(): Scheme {
  return darkQuery()?.matches ? "dark" : "light";
}

export function currentScheme(): Scheme {
  return chosen ?? systemScheme();
}

function readChoice(): Scheme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

function keepChoice(s: Scheme | null) {
  try {
    if (s) localStorage.setItem(KEY, s);
    else localStorage.removeItem(KEY);
  } catch {
    // no storage: the choice holds until the page is reloaded
  }
}

function notify() {
  for (const f of listeners) f();
}

/** `data-theme` on <html> for a chosen scheme (theme.css reads it); none when following the system. */
function apply() {
  if (chosen) document.documentElement.dataset.theme = chosen;
  else delete document.documentElement.dataset.theme;
  notify();
}

/** Switch to the other scheme. */
export function toggleScheme() {
  const next: Scheme = currentScheme() === "dark" ? "light" : "dark";
  chosen = next === systemScheme() ? null : next;
  keepChoice(chosen);
  apply();
}

/** Once, before the first render: apply a kept choice and follow the system's changes. */
export function initScheme() {
  chosen = readChoice();
  apply();
  darkQuery()?.addEventListener("change", notify);
}

function subscribe(f: () => void) {
  listeners.add(f);
  return () => {
    listeners.delete(f);
  };
}

/** The scheme the page is shown in; re-renders when it changes. */
export function useScheme(): Scheme {
  return useSyncExternalStore(subscribe, currentScheme);
}
