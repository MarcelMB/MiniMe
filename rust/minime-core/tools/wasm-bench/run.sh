#!/bin/bash
# Old vs new kernels in headless Chrome: tools/wasm-bench/run.sh RAW_DIR WORK_DIR PORT [kernel...]
# Builds the wasm with the `bench` feature (simd128, like `npm --prefix web run wasm`) into WORK_DIR/pkg.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
raw="$1"; work="$2"; port="$3"; shift 3
mkdir -p "$work"
cd "$here/../.."
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --release --target web --out-dir "$work/pkg" --features bench >/dev/null
cargo run --release --quiet --example dump_frames -- "$raw" "$work/frames.u8" 8
node "$here/run.mjs" "$work/pkg" "$work/frames.u8" "$port" "$@"
