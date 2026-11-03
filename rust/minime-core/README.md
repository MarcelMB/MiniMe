# minime-core

Rust core of the browser MiniMe (design: `docs/browser-app.md`). One crate, two builds:

- **native** check harness: `cargo build --release`, then
  `target/release/minime-core --raw RAW --inputs INPUTS --out OUT [--check-frames T.npy]`;
- **wasm** for the web worker: `npm --prefix web run wasm` (writes `web/src/stream/pkg/`).

**Inputs:** raw GREY AVIs + `INPUTS/` from `experiments/streaming/ref_dump.py`
(`params.json`, footprints `A_*.npy`, `shifts.npy`, `ymean.npy`, `sn.npy`).
**Outputs:** the `.npy` files of `ref_dump.py`'s `ref/` (traces `C`, `f`, the sums
`stats_*`, footprints `Asp_*` / `Amerged_*`, ...) + `summary.json`; check with
`experiments/streaming/check_rust.py REF_DIR OUT`.

| Module | Python reference |
|---|---|
| `avi` | `io.read_avi` (GREY AVI, `idx1` or chunk scan) |
| `preprocess` | `cv2.medianBlur`, `cv2.warpAffine` (OpenCV 5 NEON: fused lerp), tophat |
| `stream` | `stream.TraceModel`, `Support`, `stream_pass`, `spatial_from_stats`, `merge_from_stats` |
| `nnqp` | `spatial_kernel.solve_nnqp` |
| `butter` | `scipy.signal.butter(..., output="sos")` + `sosfilt` |
| `linalg` | `init._solve_normal` (Cholesky / pinv) |

`cargo test` runs the unit tests; `cargo run --release --example bench_prep` times the
per-frame preprocessing.
