"""What the pipeline does with each optional 1p effect of the synthetic session.

One tiny pipeline run (300 frames, 128 x 128, ~8 s) per effect, scored against the truth.
Recall is over ``synth.detectable`` cells (centred inside the frame, at least two spikes),
since the scorer's plain recall also counts cells in the motion margin or silent ones.
``xfail`` marks document a current weakness; the reason names it with the measured number.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from minime import pipeline, score
from minime.params import Params
from minime.testing import synth

T, SIDE = 300, 128
# the tiny-session parameters of tests/conftest.py (copied: conftest is being reworked)
PARAMS = dict(
    tile=32,
    seeds_wnd=100,
    seeds_stp=50,
    seeds_max_wnd=8,
    template_frames=50,
    inita_wnd=8,
    dl_wnd=5,
    size_thres=(9, None),
    svd_k0=16,
    motion_before_tophat=False,
)


def _session(root: Path, options: synth.SynthOptions, side: int = SIDE) -> Path:
    raw = root / "raw"
    synth.write_session(
        raw, minutes=T / 30 / 60, fps=30, segment=100, seed=0, side=side, options=options
    )
    return raw


def _unit_centroids(run: Path) -> np.ndarray:
    meta = json.loads((run / "factors" / "meta.json").read_text(encoding="utf-8"))
    A = sparse.load_npz(run / "cnmf" / "A.npz")
    uc = score.footprint_centroids(A, int(meta["H"]), int(meta["W"]))
    return uc[np.isfinite(uc).all(axis=1)]


class Result:
    """A scored run: the scorer's dict plus recall over detectable cells."""

    def __init__(self, out: dict, run: Path, s: synth.SynthSetup):
        self.out, self.run, self.setup = out, run, s
        self.offset = np.array(out["align_offset_px"])
        _, ti = score.match_units(_unit_centroids(run), synth.true_centroids(s) + self.offset)
        self.matched = set(ti.tolist())
        self.detectable = set(np.flatnonzero(synth.detectable(s, SIDE, SIDE)).tolist())
        self.recall = len(self.matched & self.detectable) / len(self.detectable)
        self.precision = out["precision"]
        self.corr = out["trace_corr_median"]
        self.n_units = out["n_units"]

    def motion_error(self) -> np.ndarray:
        """(T,) px error of the estimated rigid shifts after removing the template offset."""
        est = np.load(self.run / "motion" / "arrays.npz")["shifts"].astype(float)
        c, _, sign = score.motion_fit(est, self.setup.shifts.astype(float))
        return np.hypot(*(est - sign * self.setup.shifts - c).T)


def _run(root: Path, options: synth.SynthOptions) -> Result:
    raw = _session(root, options)
    run = pipeline.run(raw, Params(**PARAMS), run_name="t", workers=1)
    return Result(score.score_pipeline_dir(run), run, synth.synth_setup(SIDE, SIDE, T, 0, options))


# --- scorer ----------------------------------------------------------------------------


def test_score_truth_against_itself_with_dropped_frames(tmp_path):
    o = synth.SynthOptions(dropped=5, duplicated=5)
    raw = _session(tmp_path, o, side=64)
    s = synth.synth_setup(64, 64, T, 0, o)
    truth, ok = score._truth_traces(s)
    assert truth.shape == (s.ys.size, T) and ok.sum() == T - 5
    cents = np.rint(synth.true_centroids(s)).astype(int)
    keep = np.flatnonzero((cents >= 0).all(axis=1) & (cents < 64).all(axis=1))
    A = sparse.lil_matrix((keep.size, 64 * 64), dtype=np.float32)
    for i, k in enumerate(keep):
        A[i, cents[k, 0] * 64 + cents[k, 1]] = 1.0
    meta = raw.parent / "meta.json"
    out = score.score_run(sparse.csr_matrix(A), truth[keep], 64, 64, meta, s.shifts)
    assert out["n_matched"] == keep.size and out["trace_corr_median"] > 0.999


def test_score_reads_truth_npz_like_the_regenerated_truth(tmp_path):
    o = synth.SynthOptions(overlap=2, out_of_focus=2, dropped=3)
    raw = _session(tmp_path, o, side=64)
    s = synth.synth_setup(64, 64, T, 0, o)
    truth, _ = score._truth_traces(s)
    cents = np.rint(synth.true_centroids(s)).astype(int)
    keep = np.flatnonzero((cents >= 0).all(axis=1) & (cents < 64).all(axis=1))
    A = sparse.lil_matrix((keep.size + 1, 64 * 64), dtype=np.float32)
    for i, k in enumerate(keep):
        A[i, cents[k, 0] * 64 + cents[k, 1]] = 1.0
    oc = np.rint(synth.oof_centroids(s)[0]).clip(0, 63).astype(int)
    A[keep.size, oc[0] * 64 + oc[1]] = 1.0  # a false unit on an out-of-focus cell
    traces = np.vstack([truth[keep], np.zeros((1, T), np.float32)])
    meta = raw.parent / "meta.json"
    from_npz = score.score_run(sparse.csr_matrix(A), traces, 64, 64, meta, s.shifts)
    (raw.parent / "truth.npz").unlink()
    regen = score.score_run(sparse.csr_matrix(A), traces, 64, 64, meta, s.shifts)
    assert from_npz == regen
    assert from_npz["recall_detectable"] == 1.0
    assert from_npz["recall_by_category"]["overlap"]["recall"] == 1.0
    assert from_npz["n_matched"] == keep.size and from_npz["units_on_out_of_focus"] == 1


def test_score_meta_without_options(tmp_path):
    raw = _session(tmp_path, synth.SynthOptions(), side=64)
    meta_path = raw.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["options"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    s = synth.synth_setup(64, 64, T, 0)
    A = sparse.csr_matrix(np.zeros((1, 64 * 64), np.float32))
    out = score.score_run(A, np.zeros((1, T), np.float32), 64, 64, meta_path, s.shifts)
    assert out["n_true"] == s.ys.size


# --- one pipeline run per effect ---------------------------------------------------------


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("clean"), synth.SynthOptions())


def test_clean_baseline(clean):
    assert clean.out["alignment_ok"]
    assert clean.recall >= 0.9 and clean.precision >= 0.9 and clean.corr > 0.9


def test_background_drift(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["background"])
    assert r.recall >= clean.recall - 0.1
    assert r.precision >= clean.precision - 0.1


def test_neuropil(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["neuropil"])
    assert r.recall >= clean.recall - 0.1
    assert r.corr > clean.corr - 0.1


@pytest.mark.xfail(
    strict=True,
    reason="overlapping pairs 6-7 px apart with trace correlation 0.7-0.9 merge into one "
    "unit: 1 of 3 pairs split on 2026-10-27",
)
def test_overlapping_cells_are_split(tmp_path):
    o = synth.SCENARIOS["overlap"]
    r = _run(tmp_path, o)
    n = r.setup.ys.size - o.overlap
    pairs = [(int(np.flatnonzero(r.setup.group[:n] == k)[0]), n + k) for k in range(o.overlap)]
    split = sum(int(a in r.matched and b in r.matched) for a, b in pairs)
    assert split >= o.overlap - 1


def test_late_active_cells_are_found(tmp_path):
    o = synth.SCENARIOS["late_active"]
    r = _run(tmp_path, o)
    start = int(o.late_active_start * T)
    late = set(np.flatnonzero((r.setup.first_active >= start) & (r.setup.first_active < T)))
    assert late and late <= r.detectable
    assert late <= r.matched


def test_bleaching(tmp_path, clean):
    r = _run(tmp_path, synth.SynthOptions(bleach_tau=150.0))
    assert r.recall >= clean.recall - 0.1
    assert r.corr > clean.corr - 0.1


def test_vessels_keep_recall(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["vessels"])
    assert r.recall >= clean.recall - 0.1


def test_vessels_no_unit_on_a_vessel(tmp_path):
    r = _run(tmp_path, synth.SCENARIOS["vessels"])
    m = r.setup.margin
    inner = r.setup.vessel_mask[m:-m, m:-m]
    c = np.rint(_unit_centroids(r.run) - r.offset).astype(int)
    c = c[(c >= 0).all(axis=1) & (c < SIDE).all(axis=1)]
    assert int((inner[c[:, 0], c[:, 1]] < 0.7).sum()) == 0


# thin hard-edged vessel lines gave precision 0.57; branching soft vessels give 0.94 = clean
def test_vessels_keep_precision(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["vessels"])
    assert r.precision >= clean.precision - 0.1


def test_large_rigid_motion(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["large_motion"])
    err = r.motion_error()
    assert np.median(err) < 0.5 and (err > 1).sum() <= 3
    assert r.recall >= clean.recall - 0.1


def test_nonrigid_motion(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["nonrigid"])
    assert r.recall >= clean.recall - 0.15


def test_dropped_and_duplicated_frames(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["dropped"])
    assert r.recall >= clean.recall - 0.1
    assert r.precision >= clean.precision - 0.1
    assert r.corr > clean.corr - 0.1


def test_cell_size_and_brightness_variation_finds_big_bright_cells(tmp_path):
    r = _run(tmp_path, synth.SCENARIOS["cell_variation"])
    s = r.setup
    big_bright = set(np.flatnonzero((s.sigma >= 2.5) & (s.amp >= 30)).tolist()) & r.detectable
    assert big_bright and big_bright <= r.matched


@pytest.mark.xfail(
    strict=True,
    reason="cells of varying size and brightness produce extra units: precision 0.59 vs "
    "0.94 clean (32 units for 23 cells) on 2026-10-27",
)
def test_cell_size_and_brightness_variation_keeps_precision(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["cell_variation"])
    assert r.precision >= clean.precision - 0.1


def test_low_snr(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["low_snr"])
    assert r.recall >= clean.recall - 0.15


def test_bursting_spikes_with_rise(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["bursts"])
    assert r.recall >= clean.recall - 0.1
    assert r.corr > clean.corr - 0.1


@pytest.mark.xfail(
    strict=True,
    reason="bursting cells with a rise time produce extra units: precision 0.61 vs 0.94 "
    "clean (28 units for 23 cells) on 2026-10-27",
)
def test_bursting_spikes_keep_precision(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["bursts"])
    assert r.precision >= clean.precision - 0.1


def test_hemodynamic_blob_is_not_a_unit(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["hemo"])
    assert r.recall >= clean.recall - 0.1
    assert r.precision >= clean.precision - 0.1


def test_edge_band(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["edge"])
    assert r.recall >= clean.recall - 0.1


@pytest.mark.xfail(
    strict=True,
    reason="out-of-focus cells add false units: precision 0.75 vs 0.94 clean (24 units for "
    "17, 4 of them on out-of-focus cells) on 2026-10-28",
)
def test_out_of_focus_cells_keep_precision(tmp_path, clean):
    r = _run(tmp_path, synth.SCENARIOS["out_of_focus"])
    assert r.recall >= clean.recall - 0.1
    assert r.precision >= clean.precision - 0.1 and r.out["units_on_out_of_focus"] == 0


def test_realistic_scenario_floor(tmp_path):
    r = _run(tmp_path, synth.SCENARIOS["realistic"])
    # regression floor; measured on 2026-10-27: recall 0.58, precision 0.92, corr 0.98
    assert r.recall >= 0.5 and r.precision >= 0.8 and r.corr > 0.9
