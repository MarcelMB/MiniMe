import { beforeEach, describe, expect, it } from "vitest";
import { currentScheme, initScheme, toggleScheme } from "./scheme";

describe("scheme toggle", () => {
  beforeEach(() => {
    localStorage.clear();
    initScheme();
  });

  it("follows the system until the toggle picks the other scheme", () => {
    // jsdom has no matchMedia: the system scheme reads as light
    expect(currentScheme()).toBe("light");
    expect(document.documentElement.dataset.theme).toBeUndefined();
    toggleScheme();
    expect(currentScheme()).toBe("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.getItem("minime-scheme")).toBe("dark");
  });

  it("goes back to following the system when the system's scheme is chosen again", () => {
    toggleScheme();
    toggleScheme();
    expect(currentScheme()).toBe("light");
    expect(document.documentElement.dataset.theme).toBeUndefined();
    expect(localStorage.getItem("minime-scheme")).toBeNull();
  });

  it("keeps a choice across a reload", () => {
    toggleScheme();
    initScheme();
    expect(currentScheme()).toBe("dark");
  });
});
