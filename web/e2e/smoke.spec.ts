import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL(".", import.meta.url)), ".data");
const pages = ["Overview", "Frames", "Motion", "Compression", "Seeds", "Units", "Traces"];

test("opens a run and every stage page without errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`/?token=e2e&root=${encodeURIComponent(root)}`);
  await page.getByRole("link", { name: "Open run t" }).click();
  await expect(page.getByRole("heading", { name: "t", exact: true })).toBeVisible();
  await expect(page.getByRole("img", { name: /frame with unit outlines/ })).toBeVisible();
  for (const name of pages) {
    await page.getByRole("group", { name: "Stage" }).getByRole("button", { name }).click();
    await expect(page.getByRole("alert")).toHaveCount(0, { timeout: 60_000 });
  }
  await page.getByRole("group", { name: "Image" }).getByRole("button", { name: "Reconstruction" }).click();
  await page.getByRole("slider", { name: "Scrub recording" }).fill("150");
  await expect(page.getByText(/0:00:05/)).toBeVisible();
  expect(errors).toEqual([]);
});

test("the launcher lists the parameters", async ({ page }) => {
  await page.goto(`/?token=e2e#/new`);
  await expect(page.getByLabel("workers")).toBeVisible();
  await expect(page.getByRole("button", { name: "Start run" })).toBeDisabled();
});
