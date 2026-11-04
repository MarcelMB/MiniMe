/** The browser window loop's `session.json` from MiniMe's defaults (`src/minime/params.py`),
 * laid out as `experiments/streaming/ref_stitch.py --session-inputs --stream` writes it: one
 * object per stage, each with the frame size, plus the window length and the stitch and
 * streaming-pass parameters. */

/** `ref_stitch.py --length`'s default: 2.5-minute windows at 20 fps. */
export const WINDOW_FRAMES = 3000;

/** MiniMe's `Params()` defaults, as `Params.asdict()` spells them. */
export const DEFAULTS = {
  median_ksize: 7,
  tophat_wnd: 15,
  motion_max_shift: 20,
  motion_down: 2,
  template_frames: 200,
  motion_before_tophat: true,
  motion_bandpass: [1.0, 6.0] as [number, number],
  tile: 64,
  svd_k0: 40,
  svd_power_iter: 1,
  rank_rho_min: 0.2,
  rank_z: 4.0,
  noise_frames: 0,
  seeds_wnd: 1000,
  seeds_stp: 500,
  seeds_max_wnd: 15,
  seeds_diff_thres: 3,
  noise_freq: 0.06,
  pnr_thres: "auto" as number | "auto",
  gmm_refine: false,
  merge_dist: 10,
  merge_corr: 0.8,
  inita_corr: 0.8,
  inita_wnd: 10,
  init_merge_corr: 0.8,
  loop_merge_corr: 0.9,
  merge_min_dist: 0.0,
  dl_wnd: 10,
  sparse_snr: 1.0,
  size_thres: [25, null] as [number | null, number | null],
  n_iter: 2,
  background_clip: false,
};

/** `t1_discovery.py`'s stitch threshold (`cos`). */
export const STITCH_COS = 0.7;
/** `ref_dump.py`'s streaming-pass layout: sums per 6000-frame segment, 100 frames per batch. */
export const STREAM_SEGMENT_FRAMES = 6000;
export const STREAM_BATCH = 100;

export type Session = ReturnType<typeof defaultSession>;

/** The parameters the browser pipeline reads (everything `defaultSessionFrom` takes from DEFAULTS). */
export const SESSION_PARAMS = Object.keys(DEFAULTS) as readonly string[];

/** The motion sidecar's parameters (`Params.motion_params()`), the ones that fix the shifts. */
export const MOTION_PARAMS = [
  "median_ksize",
  "tophat_wnd",
  "motion_max_shift",
  "motion_down",
  "template_frames",
  "motion_before_tophat",
  "motion_bandpass",
] as const;

/** `session.json` for `H` x `W` frames with every parameter at its default. */
export function defaultSession(H: number, W: number, windowFrames = WINDOW_FRAMES) {
  return defaultSessionFrom(DEFAULTS, H, W, windowFrames);
}

/** `session.json` for `H` x `W` frames with the given parameters over the defaults. */
export function sessionFor(H: number, W: number, params: Record<string, unknown>, windowFrames = WINDOW_FRAMES) {
  return defaultSessionFrom({ ...DEFAULTS, ...params } as typeof DEFAULTS, H, W, windowFrames);
}

/** Changed parameters the browser pipeline would ignore (`workers`: it always runs on one thread). */
export function unusedParams(params: Record<string, unknown>): string[] {
  return Object.keys(params)
    .filter((k) => !SESSION_PARAMS.includes(k) && k !== "workers")
    .sort();
}

function defaultSessionFrom(p: typeof DEFAULTS, H: number, W: number, windowFrames: number) {
  const size = { H, W };
  return {
    window_frames: windowFrames,
    pass1: {
      ...size,
      median_ksize: p.median_ksize,
      tophat_wnd: p.tophat_wnd,
      motion_max_shift: p.motion_max_shift,
      motion_down: p.motion_down,
      template_frames: p.template_frames,
      motion_before_tophat: p.motion_before_tophat,
      motion_bandpass: p.motion_bandpass,
      tile: p.tile,
      seeds_wnd: p.seeds_wnd,
      seeds_stp: p.seeds_stp,
    },
    pass2: {
      ...size,
      tile: p.tile,
      svd_k0: p.svd_k0,
      svd_power_iter: p.svd_power_iter,
      rank_rho_min: p.rank_rho_min,
      rank_z: p.rank_z,
      noise_freq: p.noise_freq,
      noise_frames: p.noise_frames,
    },
    seeds: {
      ...size,
      tile: p.tile,
      seeds_wnd: p.seeds_wnd,
      seeds_stp: p.seeds_stp,
      seeds_max_wnd: p.seeds_max_wnd,
      seeds_diff_thres: p.seeds_diff_thres,
      noise_freq: p.noise_freq,
      pnr_thres: p.pnr_thres,
      gmm_refine: p.gmm_refine,
      merge_dist: p.merge_dist,
      merge_corr: p.merge_corr,
    },
    init: {
      ...size,
      tile: p.tile,
      noise_freq: p.noise_freq,
      inita_corr: p.inita_corr,
      inita_wnd: p.inita_wnd,
      init_merge_corr: p.init_merge_corr,
      merge_min_dist: p.merge_min_dist,
    },
    cnmf: {
      ...size,
      tile: p.tile,
      noise_freq: p.noise_freq,
      dl_wnd: p.dl_wnd,
      sparse_snr: p.sparse_snr,
      size_thres: p.size_thres,
      n_iter: p.n_iter,
      background_clip: p.background_clip,
      loop_merge_corr: p.loop_merge_corr,
      merge_min_dist: p.merge_min_dist,
    },
    stitch: { cos_thres: STITCH_COS },
    stream: {
      ...size,
      segment_frames: STREAM_SEGMENT_FRAMES,
      batch: STREAM_BATCH,
      median_ksize: p.median_ksize,
      tophat_wnd: p.tophat_wnd,
      motion_before_tophat: p.motion_before_tophat,
      dl_wnd: p.dl_wnd,
      sparse_snr: p.sparse_snr,
      size_thres: p.size_thres,
      noise_freq: p.noise_freq,
      loop_merge_corr: p.loop_merge_corr,
    },
  };
}
