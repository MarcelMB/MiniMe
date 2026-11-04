//! Rigid motion as in `minime.motion` and `minime.pass1`: the registration image, the
//! session template and the per-frame FFT cross-correlation estimate, plus frame quality.
//!
//! Everything up to the FFT is bit-exact with Python on arm64 (`filters`, numpy's pairwise
//! float32 mean). numpy's float32 pocketfft is not reproduced, so the correlation, and with
//! it the sub-pixel shift, agrees to float32 rounding (the integer peak should match).

use crate::filters::{area_down2_f32, area_down2_u8, gauss_f32, gauss_u8, resize_up2_f32};
use crate::preprocess::{check_median_ksize, median_blur, shift_f32, tophat};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use serde::Deserialize;
use std::sync::Arc;

/// The motion subset of `Params` (`Params.motion_params()`) plus the frame size.
#[derive(Debug, Clone, Deserialize)]
pub struct MotionParams {
    #[serde(rename = "H")]
    pub h: usize,
    #[serde(rename = "W")]
    pub w: usize,
    pub median_ksize: usize,
    pub tophat_wnd: usize,
    pub motion_max_shift: usize,
    pub motion_down: usize,
    pub template_frames: usize,
    pub motion_before_tophat: bool,
    pub motion_bandpass: [f64; 2],
}

// ---------------------------------------------------------------------------
// Registration image
// ---------------------------------------------------------------------------

/// The frame motion is estimated on: band-passed float32, or the uint8 tophat.
#[derive(Debug, Clone)]
pub enum Reg {
    F32(Vec<f32>),
    U8(Vec<u8>),
}

impl Reg {
    /// `motion.apply_shifts` on one frame.
    pub fn shifted(&self, h: usize, w: usize, dy: f32, dx: f32) -> Reg {
        match self {
            Reg::F32(f) => Reg::F32(shift_f32(f, h, w, dy, dx)),
            Reg::U8(f) => Reg::U8(crate::preprocess::shift(f, h, w, dy, dx)),
        }
    }
}

/// `preprocess.bandpass(den, lo, hi)` on one uint8 frame (even `h`, `w`): the sigma-`lo`
/// blur minus the sigma-`hi` blur, the wide one taken at half resolution and upsampled.
pub fn bandpass(den: &[u8], h: usize, w: usize, lo: f64, hi: f64) -> Vec<f32> {
    let half: Vec<f32> = area_down2_u8(den, h, w).iter().map(|&v| f32::from(v)).collect();
    let wide = resize_up2_f32(&gauss_f32(&half, h / 2, w / 2, hi / 2.0), h / 2, w / 2);
    gauss_u8(den, h, w, lo)
        .iter()
        .zip(&wide)
        .map(|(&n, &b)| f32::from(n) - b)
        .collect()
}

/// `pass1.registration_image` of one raw frame (median first, as `Pass1.process`).
pub fn registration_image(raw: &[u8], p: &MotionParams) -> Reg {
    registration_from_denoised(&median_blur(raw, p.h, p.w, p.median_ksize), p)
}

/// `pass1.registration_image` of one median-filtered frame.
pub fn registration_from_denoised(den: &[u8], p: &MotionParams) -> Reg {
    if p.motion_before_tophat {
        let [lo, hi] = p.motion_bandpass;
        Reg::F32(bandpass(den, p.h, p.w, lo, hi))
    } else {
        Reg::U8(tophat(den, p.h, p.w, p.tophat_wnd))
    }
}

// ---------------------------------------------------------------------------
// numpy helpers
// ---------------------------------------------------------------------------

/// numpy's float32 `pairwise_sum` (8 accumulators per 128-block, halves above that).
fn pairwise_sum(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(-0.0, |s, &v| s + v);
    }
    if n <= 128 {
        let mut r = [0f32; 8];
        r.copy_from_slice(&a[..8]);
        let full = n - n % 8;
        let mut i = 8;
        while i < full {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[full..] {
            res += v;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// `x.mean()` of a contiguous float32 array.
fn mean_f32(x: &[f32]) -> f32 {
    (0.0 + pairwise_sum(x)) / x.len() as f32
}

/// `np.hanning(m)`.
fn hanning(m: usize) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    (0..m)
        .map(|i| {
            let n = 1 - m as i64 + 2 * i as i64;
            0.5 + 0.5 * (std::f64::consts::PI * n as f64 / (m - 1) as f64).cos()
        })
        .collect()
}

/// `|np.fft.fftfreq(n) * n| <= m` per index (computed as numpy does, in float64).
fn in_window(n: usize, m: usize) -> Vec<bool> {
    let val = 1.0 / n as f64;
    let half = (n - 1) / 2 + 1;
    (0..n)
        .map(|i| {
            let k = if i < half { i as i64 } else { i as i64 - n as i64 };
            (k as f64 * val * n as f64).abs() <= m as f64
        })
        .collect()
}

/// `motion._parabolic`: sub-pixel peak offset from three samples (float32), clipped.
fn parabolic(c_m: f32, c_0: f32, c_p: f32) -> f32 {
    let den = c_m - 2.0 * c_0 + c_p;
    let off = if den.abs() > 1e-12 {
        0.5 * (c_m - c_p) / den
    } else {
        0.0
    };
    off.clamp(-0.5, 0.5)
}

// ---------------------------------------------------------------------------
// FFT correlation
// ---------------------------------------------------------------------------

/// Prepared-image geometry, the Hann window and the 2-D real FFT plans.
///
/// Spectra are stored column-major, `(pw / 2 + 1) x ph`, so the column FFTs run over
/// contiguous memory.
pub struct Correlator {
    down: usize,
    h: usize,
    w: usize,
    ph: usize,
    pw: usize,
    nc: usize,
    hann: Vec<f32>,
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
}

impl Correlator {
    pub fn new(h: usize, w: usize, down: usize) -> Result<Self, String> {
        if down != 1 && down != 2 {
            return Err(format!("motion_down must be 1 or 2, not {down}"));
        }
        if down == 2 && (!h.is_multiple_of(2) || !w.is_multiple_of(2)) {
            return Err(format!("{h} x {w} frames: 2x downsampling needs an even size"));
        }
        let (ph, pw) = (h / down, w / down);
        let (hy, hx) = (hanning(ph), hanning(pw));
        let hann = hy
            .iter()
            .flat_map(|&a| hx.iter().map(move |&b| (a * b) as f32))
            .collect();
        let mut rp = RealFftPlanner::<f32>::new();
        let mut cp = FftPlanner::<f32>::new();
        Ok(Self {
            down,
            h,
            w,
            ph,
            pw,
            nc: pw / 2 + 1,
            hann,
            r2c: rp.plan_fft_forward(pw),
            c2r: rp.plan_fft_inverse(pw),
            fwd: cp.plan_fft_forward(ph),
            inv: cp.plan_fft_inverse(ph),
        })
    }

    /// `(ph, pw)`: the prepared (downsampled) image size.
    pub fn prep_shape(&self) -> (usize, usize) {
        (self.ph, self.pw)
    }

    /// `motion._prep` of one frame: downsample, remove the mean, Hann-window (float32).
    pub fn prep(&self, reg: &Reg) -> Vec<f32> {
        let mut f: Vec<f32> = match (reg, self.down) {
            (Reg::F32(x), 2) => area_down2_f32(x, self.h, self.w),
            (Reg::F32(x), _) => x.clone(),
            (Reg::U8(x), 2) => area_down2_u8(x, self.h, self.w)
                .iter()
                .map(|&v| f32::from(v))
                .collect(),
            (Reg::U8(x), _) => x.iter().map(|&v| f32::from(v)).collect(),
        };
        let m = mean_f32(&f);
        for (v, &hw) in f.iter_mut().zip(&self.hann) {
            *v = (*v - m) * hw;
        }
        f
    }

    /// `np.fft.rfft2` (column-major spectrum).
    fn rfft2(&self, img: &[f32]) -> Vec<Complex32> {
        let (ph, pw, nc) = (self.ph, self.pw, self.nc);
        let mut spec = vec![Complex32::new(0.0, 0.0); nc * ph];
        let mut row = vec![0f32; pw];
        let mut out = self.r2c.make_output_vec();
        for y in 0..ph {
            row.copy_from_slice(&img[y * pw..(y + 1) * pw]);
            self.r2c.process(&mut row, &mut out).expect("rfft sizes");
            for (c, &v) in out.iter().enumerate() {
                spec[c * ph + y] = v;
            }
        }
        self.fwd.process(&mut spec);
        spec
    }

    /// `np.fft.irfft2(spec, s=(ph, pw))` of a column-major spectrum (consumed).
    fn irfft2(&self, mut spec: Vec<Complex32>) -> Vec<f32> {
        let (ph, pw, nc) = (self.ph, self.pw, self.nc);
        self.inv.process(&mut spec);
        let scale = 1.0 / (ph * pw) as f32;
        let mut img = vec![0f32; ph * pw];
        let mut line = self.c2r.make_input_vec();
        let mut out = vec![0f32; pw];
        for y in 0..ph {
            for (c, v) in line.iter_mut().enumerate() {
                *v = spec[c * ph + y];
            }
            // numpy's irfft ignores the imaginary parts of the DC and Nyquist bins
            line[0].im = 0.0;
            if pw % 2 == 0 {
                line[nc - 1].im = 0.0;
            }
            self.c2r.process(&mut line, &mut out).expect("irfft sizes");
            for (d, &v) in img[y * pw..(y + 1) * pw].iter_mut().zip(&out) {
                *d = v * scale;
            }
        }
        img
    }
}

// ---------------------------------------------------------------------------
// Template, estimate, quality
// ---------------------------------------------------------------------------

/// `motion.make_template` fed one registration frame at a time: the same `n` frames are
/// pushed twice, in order. Round 1 averages the prepared frames into a first template;
/// round 2 aligns each frame to it and averages again (numpy's sequential float32 mean).
pub struct TemplateBuilder {
    corr: Correlator,
    n: usize,
    max_shift: usize,
    acc: Vec<f32>,
    pushed: usize,
    first: Option<Motion>,
}

impl TemplateBuilder {
    pub fn new(p: &MotionParams, n: usize) -> Result<Self, String> {
        check_median_ksize(p.median_ksize)?;
        if n == 0 {
            return Err("no frames for the motion template".into());
        }
        let corr = Correlator::new(p.h, p.w, p.motion_down)?;
        let acc = vec![0f32; corr.ph * corr.pw];
        Ok(Self {
            corr,
            n,
            max_shift: p.motion_max_shift,
            acc,
            pushed: 0,
            first: None,
        })
    }

    /// Frames pushed so far, over both rounds (done at `2 n`).
    pub fn pushed(&self) -> usize {
        self.pushed
    }

    pub fn push(&mut self, reg: &Reg) {
        assert!(
            self.pushed < 2 * self.n,
            "template already has 2 x {} frames",
            self.n
        );
        let prep = match &self.first {
            None => self.corr.prep(reg),
            Some(m) => {
                let [dy, dx] = m.estimate(reg);
                let (h, w) = (self.corr.h, self.corr.w);
                self.corr.prep(&reg.shifted(h, w, dy as f32, dx as f32))
            }
        };
        for (a, v) in self.acc.iter_mut().zip(prep) {
            *a += v;
        }
        self.pushed += 1;
        if self.pushed == self.n {
            let corr = Correlator::new(self.corr.h, self.corr.w, self.corr.down).expect("checked in new");
            self.first = Some(Motion::from_image(corr, self.mean(), self.max_shift));
            self.acc.fill(0.0);
        }
    }

    fn mean(&self) -> Vec<f32> {
        let nf = self.n as f32;
        self.acc.iter().map(|&v| v / nf).collect()
    }

    pub fn finish(self) -> Result<Motion, String> {
        if self.pushed != 2 * self.n {
            return Err(format!("template got {} of 2 x {} frames", self.pushed, self.n));
        }
        let img = self.mean();
        Ok(Motion::from_image(self.corr, img, self.max_shift))
    }
}

/// A built template with what `estimate` and `quality` need.
pub struct Motion {
    corr: Correlator,
    spec: Vec<Complex32>, // conj(rfft2(template)), column-major
    image: Vec<f32>,      // `motion.template_image`
    src: Vec<f32>,        // the image `spec` was made from (before `image`'s FFT round trip)
    max_shift: usize,
    win_y: Vec<bool>,
    win_x: Vec<bool>,
}

impl Motion {
    pub(crate) fn from_image(corr: Correlator, img: Vec<f32>, max_shift: usize) -> Self {
        let spec = corr.rfft2(&img).iter().map(|v| v.conj()).collect();
        let image = corr.irfft2(corr.rfft2(&img));
        let m = max_shift / corr.down;
        let (win_y, win_x) = (in_window(corr.ph, m), in_window(corr.pw, m));
        Self {
            corr,
            spec,
            image,
            src: img,
            max_shift,
            win_y,
            win_x,
        }
    }

    /// The prepared template before its FFT round trip: `from_template_image` of it rebuilds
    /// this `Motion` exactly (of `template_image()` it does not: that one is round-tripped once more).
    pub fn source_image(&self) -> &[f32] {
        &self.src
    }

    /// A template from a stored prepared template image (`motion.template_image`, as in
    /// a sidecar), like `pipeline._template_from_sidecar`.
    pub fn from_template_image(p: &MotionParams, img: Vec<f32>) -> Result<Self, String> {
        check_median_ksize(p.median_ksize)?;
        let corr = Correlator::new(p.h, p.w, p.motion_down)?;
        if img.len() != corr.ph * corr.pw {
            return Err(format!(
                "template holds {} pixels, expected {} x {}",
                img.len(),
                corr.ph,
                corr.pw
            ));
        }
        Ok(Self::from_image(corr, img, p.motion_max_shift))
    }

    /// `motion.make_template` from `n` registration frames, fetched twice by `get_reg`
    /// (see `TemplateBuilder`), so only one frame is held at a time.
    pub fn build(
        p: &MotionParams,
        n: usize,
        get_reg: &mut dyn FnMut(usize) -> Result<Reg, String>,
    ) -> Result<Self, String> {
        let mut b = TemplateBuilder::new(p, n)?;
        for _round in 0..2 {
            for t in 0..n {
                b.push(&get_reg(t)?);
            }
        }
        b.finish()
    }

    /// `motion.template_image`: the prepared template, `prep_shape()` float32.
    pub fn template_image(&self) -> &[f32] {
        &self.image
    }

    pub fn prep_shape(&self) -> (usize, usize) {
        self.corr.prep_shape()
    }

    pub fn max_shift(&self) -> usize {
        self.max_shift
    }

    /// `motion.estimate` of one frame: `(dy, dx)` in full-resolution pixels.
    pub fn estimate(&self, reg: &Reg) -> [f64; 2] {
        let c = &self.corr;
        let (ph, pw) = (c.ph, c.pw);
        let mut spec = c.rfft2(&c.prep(reg));
        for (v, t) in spec.iter_mut().zip(&self.spec) {
            *v *= t;
        }
        let corr = c.irfft2(spec);
        let at = |y: usize, x: usize| -> f32 {
            if self.win_y[y] && self.win_x[x] {
                corr[y * pw + x]
            } else {
                0.0
            }
        };
        let (mut iy, mut ix, mut best) = (0, 0, f32::NEG_INFINITY);
        for y in (0..ph).filter(|&y| self.win_y[y]) {
            for x in (0..pw).filter(|&x| self.win_x[x]) {
                if corr[y * pw + x] > best {
                    (iy, ix, best) = (y, x, corr[y * pw + x]);
                }
            }
        }
        let c0 = at(iy, ix);
        let sy = parabolic(at((iy + ph - 1) % ph, ix), c0, at((iy + 1) % ph, ix));
        let sx = parabolic(at(iy, (ix + pw - 1) % pw), c0, at(iy, (ix + 1) % pw));
        let wrap = |i: usize, n: usize| if i > n / 2 { i as f64 - n as f64 } else { i as f64 };
        let down = c.down as f64;
        [
            down * (wrap(iy, ph) + f64::from(sy)),
            down * (wrap(ix, pw) + f64::from(sx)),
        ]
    }

    /// `motion.frame_quality` of one motion-corrected registration frame: Pearson
    /// correlation over pixels with the template.
    pub fn quality(&self, warped: &Reg) -> f32 {
        let f = self.corr.prep(warped);
        let (mf, mt) = (mean_f32(&f), mean_f32(&self.image));
        let (mut ft, mut ff, mut tt) = (0f64, 0f64, 0f64);
        for (&a, &b) in f.iter().zip(&self.image) {
            let (a, b) = (f64::from(a - mf), f64::from(b - mt));
            ft += a * b;
            ff += a * a;
            tt += b * b;
        }
        let den = ff.sqrt() * tt.sqrt();
        if den > 0.0 {
            (ft / den) as f32
        } else {
            0.0
        }
    }
}

/// `serve.derived.block_residual` of `n` block maxima (`block_max`, `n * h * w` float32, frame
/// size from `p`): each block's `(dy, dx)` and Pearson correlation against the session's max
/// projection, `(n, 3)` flat. The reference is `motion._mean_template(max_proj, down=1)` (the
/// prepared max projection, no downsampling) and the shifts search `p.motion_max_shift` px (main
/// calls `motion.estimate` with its default of 20, which a run with the default parameters
/// shares); like main, the correlation is of the block as it is, not shifted. A registration that
/// held gives ~0 px for every block.
pub fn block_residual(
    block_max: &[f32],
    n: usize,
    max_proj: &[f32],
    p: &MotionParams,
) -> Result<Vec<f32>, String> {
    let px = p.h * p.w;
    if block_max.len() != n * px {
        return Err(format!(
            "{n} blocks of {} x {} need {} values, got {}",
            p.h,
            p.w,
            n * px,
            block_max.len()
        ));
    }
    if max_proj.len() != px {
        return Err(format!(
            "the max projection holds {} values, not {} x {} = {px}",
            max_proj.len(),
            p.h,
            p.w
        ));
    }
    let corr = Correlator::new(p.h, p.w, 1)?;
    let img = corr.prep(&Reg::F32(max_proj.to_vec()));
    let reference = Motion::from_image(corr, img, p.motion_max_shift);
    let mut out = Vec::with_capacity(3 * n);
    for block in block_max.chunks_exact(px.max(1)).take(n) {
        let reg = Reg::F32(block.to_vec());
        let [dy, dx] = reference.estimate(&reg);
        out.extend([dy as f32, dx as f32, reference.quality(&reg)]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_window_and_hann() {
        let win = in_window(300, 10);
        assert_eq!(win.iter().filter(|&&v| v).count(), 21);
        assert!(win[10] && !win[11] && win[290] && !win[289]);
        let hn = hanning(5);
        assert_eq!(hn, vec![0.0, 0.5, 1.0, 0.5, 0.0]);
    }

    /// A smooth test image with two blobs: `shift_f32(img, .., dy, dx)` is its warp.
    fn blobs(h: usize, w: usize) -> Vec<f32> {
        (0..h * w)
            .map(|i| {
                let (y, x) = ((i / w) as f32, (i % w) as f32);
                let r2 = (y - 30.0).powi(2) + (x - 41.0).powi(2);
                100.0 * (-r2 / 50.0).exp() + 40.0 * (-((y - 12.0).powi(2) + (x - 60.0).powi(2)) / 20.0).exp()
            })
            .collect()
    }

    fn residual_params() -> MotionParams {
        MotionParams {
            motion_down: 2, // block_residual compares at full resolution whatever this says
            ..params(7)
        }
    }

    #[test]
    fn block_residual_finds_the_offset_of_each_block_from_the_session_max() {
        // blocks that are the session's max shifted by (dy, dx) need the shift (-dy, -dx)
        let (h, w) = (64, 80);
        let img = blobs(h, w);
        let shifts = [(0.0f32, 0.0f32), (4.0, -6.0), (-3.0, 2.0)];
        let blocks: Vec<f32> = shifts
            .iter()
            .flat_map(|&(dy, dx)| shift_f32(&img, h, w, dy, dx))
            .collect();
        let p = MotionParams {
            h,
            w,
            ..residual_params()
        };
        let out = block_residual(&blocks, 3, &img, &p).unwrap();
        assert_eq!(out.len(), 9);
        for (b, &(dy, dx)) in shifts.iter().enumerate() {
            let (ry, rx, c) = (out[3 * b], out[3 * b + 1], out[3 * b + 2]);
            assert!(
                (ry + dy).abs() < 0.5 && (rx + dx).abs() < 0.5,
                "block {b}: {ry} {rx}"
            );
            // the correlation is that of the block with the session max, not of the aligned block
            assert!(c > 0.5 && c <= 1.0 + 1e-6, "block {b}: {c}");
        }
        assert!(out[2] > 0.999, "an unshifted block matches the max: {}", out[2]);
    }

    #[test]
    fn block_residual_looks_no_further_than_the_max_shift() {
        let (h, w) = (64, 80);
        let img = blobs(h, w);
        let block = shift_f32(&img, h, w, 5.0, 0.0);
        let near = MotionParams {
            h,
            w,
            motion_max_shift: 2,
            ..residual_params()
        };
        let [ry, _, _] = block_residual(&block, 1, &img, &near).unwrap()[..] else {
            panic!()
        };
        assert!(ry.abs() <= 2.5, "{ry}");
        let far = MotionParams {
            motion_max_shift: 20,
            ..near
        };
        let [ry, _, _] = block_residual(&block, 1, &img, &far).unwrap()[..] else {
            panic!()
        };
        // the Hann window pulls a large offset in a little (-4.5 for 5 px here)
        assert!((ry + 5.0).abs() < 1.0, "{ry}");
    }

    #[test]
    fn block_residual_checks_its_sizes() {
        let p = MotionParams {
            h: 8,
            w: 8,
            ..residual_params()
        };
        let img = vec![0f32; 64];
        assert_eq!(
            block_residual(&[0f32; 100], 2, &img, &p).err().as_deref(),
            Some("2 blocks of 8 x 8 need 128 values, got 100")
        );
        assert_eq!(
            block_residual(&[0f32; 64], 1, &img[..10], &p).err().as_deref(),
            Some("the max projection holds 10 values, not 8 x 8 = 64")
        );
        assert_eq!(block_residual(&[], 0, &img, &p).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn estimate_recovers_a_known_shift() {
        // f(y, x) = t(y + 4, x - 6) is aligned back by the shift (-4, 6)
        let (h, w) = (64, 80);
        let img: Vec<f32> = (0..h * w)
            .map(|i| {
                let (y, x) = ((i / w) as f32, (i % w) as f32);
                let r2 = (y - 30.0).powi(2) + (x - 41.0).powi(2);
                100.0 * (-r2 / 50.0).exp() + 40.0 * (-((y - 12.0).powi(2) + (x - 60.0).powi(2)) / 20.0).exp()
            })
            .collect();
        let corr = Correlator::new(h, w, 2).unwrap();
        let tmpl = corr.prep(&Reg::F32(img.clone()));
        let m = Motion::from_image(corr, tmpl, 20);
        let [dy, dx] = m.estimate(&Reg::F32(shift_f32(&img, h, w, 4.0, -6.0)));
        assert!((dy + 4.0).abs() < 0.5 && (dx - 6.0).abs() < 0.5, "{dy} {dx}");
    }

    fn params(median_ksize: usize) -> MotionParams {
        MotionParams {
            h: 64,
            w: 64,
            median_ksize,
            tophat_wnd: 15,
            motion_max_shift: 20,
            motion_down: 2,
            template_frames: 200,
            motion_before_tophat: true,
            motion_bandpass: [1.0, 6.0],
        }
    }

    #[test]
    fn the_template_and_the_stored_template_refuse_an_even_median_size() {
        // `median_blur` (cv2.medianBlur) takes odd sizes only; a run must fail before it filters
        for k in [4, 6, 8] {
            let want = format!("median_ksize must be odd, got {k}");
            assert_eq!(
                TemplateBuilder::new(&params(k), 10).err().as_deref(),
                Some(want.as_str())
            );
            let img = vec![0f32; 32 * 32];
            assert_eq!(
                Motion::from_template_image(&params(k), img).err().as_deref(),
                Some(want.as_str())
            );
        }
        for k in [0, 1, 3, 7] {
            assert!(TemplateBuilder::new(&params(k), 10).is_ok(), "{k}");
            assert!(
                Motion::from_template_image(&params(k), vec![0f32; 32 * 32]).is_ok(),
                "{k}"
            );
        }
    }
}
