"""Check a browser window loop that ends in the streaming pass (``ref_stitch.py --stream``).

    python experiments/streaming/check_chain.py REF_DIR BROWSER_DIR RAW_DIR

``REF_DIR`` is ``ref_stitch.py --session-inputs --stream``'s folder. Checks the session mean
image and noise the browser built from its windows against ``stream.window_mean_noise`` of
the stored Python windows, then runs the native streaming pass (``minime-core``) on exactly
the browser's stream inputs (its stitched footprints, mean image and noise, the session's
shifts) and compares the browser's streaming outputs with it (``check_rust.py``). The native
pass is checked against Python's prototype in slice 1, so this closes the chain. The stitch
itself is ``check_stitch.py --own-windows``'s. Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter

MINIME_CORE = Path(__file__).parents[2] / "rust/minime-core/target/release/minime-core"
TOL = {"ymean": 1e-6, "sn": 1e-5}  # relative; pass 2's mean is exact, its noise ~1e-6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", type=Path)
    ap.add_argument("browser", type=Path)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--minime-core", type=Path, default=MINIME_CORE)
    a = ap.parse_args()
    ok = True
    for name, tol in TOL.items():
        r, g = np.load(a.ref / "ref" / f"{name}.npy"), np.load(a.browser / f"{name}.npy")
        d = float((np.abs(g - r) / np.maximum(np.abs(r), 1e-12)).max())
        good = d <= tol
        print(f"{name:6s} {'ok  ' if good else 'FAIL'} max rel diff {d:.3g} (tol {tol:g})")
        ok &= good

    session = json.loads((a.ref / "session" / "session.json").read_text())
    params = dict(session["stream"])
    params["T"] = int(np.load(a.browser / "f.npy").size)
    nat = a.browser / "native_chain"
    inp = nat / "inputs"
    shutil.rmtree(nat, ignore_errors=True)
    inp.mkdir(parents=True)
    (inp / "params.json").write_text(json.dumps(params, indent=1))
    for n in ("A_data", "A_indices", "A_indptr", "ymean", "sn"):
        shutil.copy2(a.browser / f"{n}.npy", inp / f"{n}.npy")
    sh = np.load(a.ref / "session" / "shifts.npy")[: params["T"]]
    np.save(inp / "shifts.npy", sh)
    np.save(inp / "sos.npy", butter(4, 2 * params["noise_freq"], output="sos"))
    (nat / "ref").mkdir()
    subprocess.run(
        [a.minime_core, "--raw", a.raw, "--inputs", inp, "--out", nat / "ref"], check=True
    )
    rc = subprocess.run(
        [sys.executable, Path(__file__).with_name("check_rust.py"), nat, a.browser]
    ).returncode
    ok &= rc == 0
    print("chain PASS" if ok else "chain FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
