# MiniMe wgpu GPU vs CPU benchmark: Windows PC, 2026-11-02

Branch `perf/gpu-wgpu` at d627a07, `params_d12.json` from the handoff, 20 min V4 demo session
(`minime demo`, 24 files, 8.54 GB, all MD5s verified), `--workers 16` (physical core count;
the Mac used 8).

## Machine

| | |
|---|---|
| CPU | AMD Ryzen 9 9955HX, 16 cores / 32 threads |
| RAM | 61.7 GB (49.7 GB free at idle before the runs) |
| OS | Windows 11 Education 25H2, build 26200.9168 |
| Discrete GPU | NVIDIA GeForce RTX 4070, 12 GB, driver 32.0.15.9186 (NVIDIA 591.86) |
| Integrated GPU | AMD Radeon 610M ("AMD Radeon(TM) Graphics"), driver 32.0.21030.2001 |
| Data disk | D: Samsung SSD 990 PRO 4TB (NVMe); raw and `processed\` both on D: |
| Python | CPython 3.11.18 (uv-managed), wgpu 0.32.0 |

Before the runs: power mode "Best performance", Defender exclusion for `D:\minime-demo`,
Windows Update paused, Discord/Slack/ChatGPT/Notion/Ollama/Steam/OneDrive closed.
Idle CPU 1.0 % (max 2.9 % over 20 s), GPU idle.

Adapter lines from step 1 (GeForce plugged in):

```
high: NVIDIA GeForce RTX 4070 (DiscreteGPU) via Vulkan
low: AMD Radeon(TM) 610M (IntegratedGPU) via Vulkan
```

## Results (seconds)

(a) GeForce, `geforce_20min.json`:

| device | rep | wall | pass1 | pass2 | cnmf | units |
|---|---|---|---|---|---|---|
| gpu (RTX 4070) | 1 | 90.4 | 39.0 | 34.2 | 10.7 | 479 |
| cpu | 1 | 131.1 | 49.2 | 62.2 | 10.7 | 479 |
| gpu (RTX 4070) | 2 | 104.8 | 49.8 | 35.1 | 10.7 | 479 |
| cpu | 2 | 132.2 | 50.8 | 61.7 | 10.7 | 479 |

(b) iGPU, `igpu_20min.json`:

| device | rep | wall | pass1 | pass2 | cnmf | units |
|---|---|---|---|---|---|---|
| gpu (Radeon 610M) | 1 | 129.3 | 50.6 | 59.6 | 10.4 | 479 |
| gpu (Radeon 610M) | 2 | 128.4 | 49.3 | 60.1 | 10.3 | 479 |

Pass 2: CPU 62.0 s mean; RTX 4070 34.7 s (1.79x); Radeon 610M 59.9 s (1.03x).
Whole run: CPU 131.7 s; RTX 4070 97.6 s (1.35x); Radeon 610M 128.9 s (1.02x).
GeForce gpu r1's pass 1 (39 s vs 49-51 s in every other run) is an outlier, not a GPU
effect (pass 1 is CPU-only), so compare pass 2.

Unit counts: identical at every stage on all six runs (seeds_init 14999, seeds_pnr 9540,
seeds_merged 482, units_initA 482, units after spatial/merge 479, units_final 479), so no
`units.csv` comparison was needed.

## How the iGPU was measured

Not by unplugging the GeForce: wgpu-py 0.32 picks an adapter by name substring when
`WGPUPY_WGPU_ADAPTER_NAME` is set (`wgpu/backends/wgpu_native/_api.py`, `request_adapter`),
which also covers MiniMe's `request_adapter_sync(power_preference="high-performance")` call.
The iGPU series ran with `WGPUPY_WGPU_ADAPTER_NAME="Radeon(TM) 610M"`; both `report.json`
files record `"gpu": "AMD Radeon(TM) 610M (IntegratedGPU) via Vulkan"`. This could serve as
the `MINIME_GPU=integrated|discrete` TODO without new selection code.

## Issues found on Windows

1. `minime demo` fails with `URLError: [SSL: CERTIFICATE_VERIFY_FAILED] unable to get local
   issuer certificate` under uv's Python on Windows. Python's default verify paths do not find
   the chain for `dataverse.ucla.edu` (issuer InCommon Intermediate CA - OVG2C); with
   `certifi.where()` as `cafile` it verifies. Worked around with
   `SSL_CERT_FILE=<certifi bundle>`; a fix would be an `ssl.create_default_context(cafile=certifi.where())`
   passed to both `urlopen` calls in `src/minime/demo.py`.
2. `report.json` has `"peak_rss_children": null` on Windows, which crashed the handoff's
   `bench_gpu_cpu.py` (`None / 1e9`). The driver here uses `(R.get("peak_rss_children") or 0)`;
   the Mac copy should get the same change. `peak_rss_children_gb` is therefore 0 in the
   Windows results.
3. Not MiniMe: under the Claude desktop app, uv's default Python dir in `%APPDATA%` is
   redirected by app packaging and `uv venv` fails; `UV_PYTHON_INSTALL_DIR` outside AppData
   fixes it.

The data download took ~70 min (Dataverse at ~0.3-0.45 MB/s per connection; fetched 8 files
in parallel with curl, then verified with `minime demo`).
