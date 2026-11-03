//! Digital Butterworth low-pass as second-order sections, `scipy.signal.butter(N, Wn,
//! output="sos")` (analog prototype -> prewarp -> bilinear -> pole pairs), and
//! `sosfilt` run per sample with its state carried over.

#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    fn add(self, o: C) -> C {
        C {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
    fn sub(self, o: C) -> C {
        C {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
    fn mul(self, o: C) -> C {
        C {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    fn div(self, o: C) -> C {
        let d = o.re * o.re + o.im * o.im;
        C {
            re: (self.re * o.re + self.im * o.im) / d,
            im: (self.im * o.re - self.re * o.im) / d,
        }
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

/// `(n_sections, 6)` rows `[b0, b1, b2, 1, a1, a2]`; `wn` is the cutoff as a fraction of
/// Nyquist (`2 * noise_freq` for a cutoff in cycles per frame). Even `order` only.
pub fn butter_lowpass_sos(order: usize, wn: f64) -> Vec<[f64; 6]> {
    assert!(order.is_multiple_of(2) && order > 0, "even order only");
    let n = order as f64;
    let fs2 = 4.0; // 2 * fs with fs = 2
    let warped = 2.0 * 2.0 * (std::f64::consts::PI * wn / 2.0).tan();
    // analog prototype poles, scaled by the warped cutoff
    let poles: Vec<C> = (0..order)
        .map(|i| {
            let m = -(n - 1.0) + 2.0 * i as f64;
            let a = std::f64::consts::PI * m / (2.0 * n);
            C {
                re: -a.cos() * warped,
                im: -a.sin() * warped,
            }
        })
        .collect();
    let k_analog = warped.powi(order as i32);
    let f = C { re: fs2, im: 0.0 };
    let pz: Vec<C> = poles.iter().map(|&p| f.add(p).div(f.sub(p))).collect();
    let mut den = C { re: 1.0, im: 0.0 };
    for &p in &poles {
        den = den.mul(f.sub(p));
    }
    let k = k_analog * C { re: 1.0, im: 0.0 }.div(den).re;
    // conjugate pairs (upper half plane), farthest from the unit circle first
    let mut up: Vec<C> = pz.into_iter().filter(|p| p.im > 0.0).collect();
    up.sort_by(|a, b| (1.0 - a.abs()).abs().total_cmp(&(1.0 - b.abs()).abs()).reverse());
    up.iter()
        .enumerate()
        .map(|(i, p)| {
            let g = if i == 0 { k } else { 1.0 };
            [g, 2.0 * g, g, 1.0, -(p.re + p.re), p.re * p.re + p.im * p.im]
        })
        .collect()
}

/// `sosfilt` with state per channel, fed one sample per channel at a time.
pub struct SosFilter {
    sos: Vec<[f64; 6]>,
    zi: Vec<[f64; 2]>, // (n_sections * channels)
    channels: usize,
}

impl SosFilter {
    pub fn new(sos: Vec<[f64; 6]>, channels: usize) -> Self {
        let zi = vec![[0.0; 2]; sos.len() * channels];
        Self { sos, zi, channels }
    }

    pub fn sos(&self) -> &[[f64; 6]] {
        &self.sos
    }

    /// Filter one sample of every channel in place.
    pub fn step(&mut self, x: &mut [f64]) {
        for (c, xc) in x.iter_mut().enumerate().take(self.channels) {
            let mut cur = *xc;
            for (s, sec) in self.sos.iter().enumerate() {
                let z = &mut self.zi[s * self.channels + c];
                let new = sec[0] * cur + z[0];
                z[0] = sec[1] * cur - sec[4] * new + z[1];
                z[1] = sec[2] * cur - sec[5] * new;
                cur = new;
            }
            *xc = cur;
        }
    }
}
