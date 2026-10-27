import shutil

import numpy as np
import scipy.sparse as sps

from minime.serve import derived
from minime.serve.runs import RunView


def _snapshot(d):
    return {
        p.relative_to(d): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


def test_saved_seeds_and_init_are_read_from_the_run(tiny_run, cache_dir):
    rv = RunView(tiny_run, cache_dir)
    s = derived.seeds(rv)
    assert set(s) == {"init", "pnr", "merged"}
    assert derived.init_A(rv).shape[0] == rv.report["counts"]["units_init_merge"]


def test_older_run_gets_seeds_and_init_computed_into_the_cache(tiny_run, run_copy, cache_dir):
    shutil.rmtree(run_copy / "seeds")
    shutil.rmtree(run_copy / "init")
    before = _snapshot(run_copy)
    rv = RunView(run_copy, cache_dir)
    assert derived.seeds(rv) is None and derived.init_A(rv) is None
    derived.compute_seeds_and_init(rv)
    got = derived.seeds(rv)
    for k in derived.SEED_STAGES:
        np.testing.assert_array_equal(got[k], np.load(tiny_run / "seeds" / f"{k}.npy"))
    want = sps.load_npz(tiny_run / "init" / "A.npz")
    assert (derived.init_A(rv) != want).nnz == 0
    assert _snapshot(run_copy) == before


def test_residual_and_overview(tiny_run, cache_dir):
    rv = RunView(tiny_run, cache_dir)
    assert derived.residual(rv) is None
    derived.compute_residual(rv)
    r = derived.residual(rv)
    assert r.dtype == np.float32 and r.shape == (len(rv.factors.block_max), 3)
    assert derived.traces_overview(rv, "C", 100) is None
    derived.compute_overview(rv, "C", 100)
    ov = derived.traces_overview(rv, "C", 100)
    assert ov.shape == (rv.A.shape[0], 2, 100)
