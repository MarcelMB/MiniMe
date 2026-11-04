// Build the e2e run (once) and dump main's answers on it: e2e/.data/contract-python/.
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";

const py = process.env.MINIME_PYTHON ?? resolve("../.venv/bin/python");
const env = { ...process.env, PYTHONPATH: resolve("../src") };
const run = (...args) => execFileSync(py, args, { stdio: "inherit", env });
run("e2e/make_run.py", "e2e/.data");
run("scripts/contract_dump.py", "e2e/.data/processed/t", "e2e/.data/contract-python", "--raw", "e2e/.data/raw");
run("scripts/params_snapshot.py", "src/browser/params.defaults.json");
// the snapshot is committed, so leave it in the repository's JSON format
execFileSync("npx", ["biome", "format", "--write", "src/browser/params.defaults.json"], { stdio: "inherit" });
