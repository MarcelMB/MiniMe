import { expect, test, type Locator, type Page } from "@playwright/test";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL(".", import.meta.url)), ".data");
const pages: [string, (page: Page) => Locator][] = [
  ["Overview", (p) => p.getByText("From seeds to units")],
  ["Frames", (p) => p.getByRole("region", { name: "Reconstruction" })],
  ["Motion", (p) => p.locator(".uplot").first()],
  ["Compression", (p) => p.getByText("Total rank")],
  ["Seeds", (p) => p.getByText("Seeds per step")],
  ["Units", (p) => p.locator("table.data")],
  ["Traces", (p) => p.getByText("Add unit", { exact: true })],
];

test("opens a run and every stage page without errors", async ({ page }) => {
  const errors: string[] = [];
  const failed: string[] = [];
  const consoleErrors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
  page.on("response", (r) => {
    if (r.url().includes("/api/") && r.status() >= 400) failed.push(`${r.status()} ${r.url()}`);
  });
  await page.goto(`/?token=e2e&root=${encodeURIComponent(root)}`);
  await page.getByRole("link", { name: "Open run t" }).click();
  await expect(page.getByRole("heading", { name: "t", exact: true })).toBeVisible();
  await expect(page.getByRole("img", { name: /frame with unit outlines/ })).toBeVisible();
  for (const [name, marker] of pages) {
    const button = page.getByRole("group", { name: "Stage" }).getByRole("button", { name });
    await button.click();
    await expect(button).toHaveAttribute("aria-pressed", "true");
    await expect(marker(page).first()).toBeVisible({ timeout: 60_000 });
    await page.waitForLoadState("networkidle");
    await expect(page.getByRole("alert")).toHaveCount(0);
  }
  await page.getByRole("group", { name: "Image" }).getByRole("button", { name: "Reconstruction" }).click();
  await page.getByRole("slider", { name: "Scrub recording" }).fill("150");
  await expect(page.getByText(/0:00:05/)).toBeVisible();
  expect(errors).toEqual([]);
  expect(failed).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test("the launcher lists the parameters", async ({ page }) => {
  await page.goto(`/?token=e2e#/new`);
  await expect(page.getByLabel("workers")).toBeVisible();
  await expect(page.getByRole("button", { name: "Start run" })).toBeDisabled();
});
