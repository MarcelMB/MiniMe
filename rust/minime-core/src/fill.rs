//! The template fill of the motion border (`motion_fill = "template"`): `motion._valid_rects`,
//! `motion.fill_outside` and `motion.fill_image`.
//!
//! A frame warped by `(dy, dx)` reads the pixels outside `[y0, y1) x [x0, x1)` from outside the
//! frame; after the tophat they get the session's fill image (the template frames' mean corrected
//! image over the frames that hold data at each pixel). Integer arithmetic throughout, so the fill
//! sums of any split of the frames add up to the same image.

/// Rows `[y0, y1)` and columns `[x0, x1)` a warp by `(dy, dx)` reads from inside the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub y0: usize,
    pub y1: usize,
    pub x0: usize,
    pub x1: usize,
}

/// `motion._valid_rects` of one float64 shift: `y0 = clip(ceil(-dy), 0, h)`,
/// `y1 = clip(floor(h - 1 - dy) + 1, 0, h)`, the same for x.
pub fn valid_rect(dy: f64, dx: f64, h: usize, w: usize) -> Rect {
    let clip = |v: f64, n: usize| v.clamp(0.0, n as f64) as usize;
    Rect {
        y0: clip((-dy).ceil(), h),
        y1: clip(((h - 1) as f64 - dy).floor() + 1.0, h),
        x0: clip((-dx).ceil(), w),
        x1: clip(((w - 1) as f64 - dx).floor() + 1.0, w),
    }
}

/// `motion.fill_outside` of one frame (in place): the rows above `y0` and from `y1` on, then the
/// columns left of `x0` and from `x1` on, from `fill`.
pub fn fill_outside(frame: &mut [u8], h: usize, w: usize, r: Rect, fill: &[u8]) {
    assert_eq!(frame.len(), h * w);
    assert_eq!(fill.len(), h * w);
    let y0 = r.y0.min(h);
    let y1 = r.y1.clamp(y0, h);
    frame[..y0 * w].copy_from_slice(&fill[..y0 * w]);
    frame[y1 * w..].copy_from_slice(&fill[y1 * w..]);
    let x0 = r.x0.min(w);
    let x1 = r.x1.clamp(x0, w);
    for y in 0..h {
        let row = y * w;
        frame[row..row + x0].copy_from_slice(&fill[row..row + x0]);
        frame[row + x1..row + w].copy_from_slice(&fill[row + x1..row + w]);
    }
}

/// The sums `motion.fill_image` divides: per pixel the sum and count over the frames that hold
/// data there, and the sum over every frame. Add frames in any order, merge partial sums.
#[derive(Debug, Clone)]
pub struct FillSum {
    pub total: Vec<u32>,
    pub count: Vec<u32>,
    pub every: Vec<u32>,
    pub n: u32,
}

impl FillSum {
    pub fn new(px: usize) -> Self {
        Self {
            total: vec![0; px],
            count: vec![0; px],
            every: vec![0; px],
            n: 0,
        }
    }

    /// One corrected template frame (`h * w` uint8, warped by the shift `r` came from).
    pub fn add(&mut self, top: &[u8], h: usize, w: usize, r: Rect) {
        assert_eq!(top.len(), h * w);
        for (e, &v) in self.every.iter_mut().zip(top) {
            *e += u32::from(v);
        }
        for y in r.y0..r.y1.min(h) {
            let row = y * w;
            for x in r.x0..r.x1.min(w) {
                self.total[row + x] += u32::from(top[row + x]);
                self.count[row + x] += 1;
            }
        }
        self.n += 1;
    }

    /// Add another worker's sums.
    pub fn merge(&mut self, o: &FillSum) {
        for (a, b) in [
            (&mut self.total, &o.total),
            (&mut self.count, &o.count),
            (&mut self.every, &o.every),
        ] {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        self.n += o.n;
    }

    /// `[total, count, every, [n]]` flat, for a message.
    pub fn to_flat(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(3 * self.total.len() + 1);
        v.extend_from_slice(&self.total);
        v.extend_from_slice(&self.count);
        v.extend_from_slice(&self.every);
        v.push(self.n);
        v
    }

    pub fn from_flat(v: &[u32]) -> Result<Self, String> {
        if v.is_empty() || !(v.len() - 1).is_multiple_of(3) {
            return Err(format!("fill sums of {} values", v.len()));
        }
        let px = (v.len() - 1) / 3;
        Ok(Self {
            total: v[..px].to_vec(),
            count: v[px..2 * px].to_vec(),
            every: v[2 * px..3 * px].to_vec(),
            n: v[3 * px],
        })
    }

    /// `motion.fill_image`: `total // count` where a frame holds data, else `every // n`.
    pub fn image(&self) -> Vec<u8> {
        let n = self.n.max(1);
        self.total
            .iter()
            .zip(&self.count)
            .zip(&self.every)
            .map(|((&t, &c), &e)| if c > 0 { t / c } else { e / n } as u8)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::{shift, shift_replicate, shift_replicate_scalar};

    fn noise(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 56) as u8
            })
            .collect()
    }

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 40) as f64 / (1u64 << 24) as f64
    }

    #[test]
    fn valid_rect_matches_numpy() {
        // frame 1 of tests/test_motion.py's example: (3, -4) on 20 x 30 reads rows 17-19 and
        // columns 0-3 from outside
        assert_eq!(valid_rect(3.0, -4.0, 20, 30), Rect { y0: 0, y1: 17, x0: 4, x1: 30 });
        assert_eq!(valid_rect(-2.0, 2.0, 20, 30), Rect { y0: 2, y1: 20, x0: 0, x1: 28 });
        assert_eq!(valid_rect(0.0, 0.0, 5, 6), Rect { y0: 0, y1: 5, x0: 0, x1: 6 });
        // fractional: a pixel whose source leaves the frame by any amount is outside
        assert_eq!(valid_rect(0.25, -0.25, 5, 6), Rect { y0: 0, y1: 4, x0: 1, x1: 6 });
        assert_eq!(valid_rect(99.0, -99.0, 5, 6), Rect { y0: 0, y1: 0, x0: 6, x1: 6 });
    }

    #[test]
    fn fill_outside_sets_exactly_what_a_zero_warp_of_white_leaves_dark() {
        let (h, w) = (37, 53);
        let mut s = 5u64;
        for _ in 0..60 {
            let dy = (lcg(&mut s) - 0.5) * 18.0;
            let dx = (lcg(&mut s) - 0.5) * 18.0;
            let (dy32, dx32) = (dy as f32, dx as f32);
            let dark: Vec<bool> = shift(&vec![255; h * w], h, w, dy32, dx32)
                .iter()
                .map(|&v| v < 255)
                .collect();
            let warped = shift_replicate(&noise(h * w, 9), h, w, dy32, dx32);
            let mut out = warped.clone();
            let r = valid_rect(f64::from(dy32), f64::from(dx32), h, w);
            fill_outside(&mut out, h, w, r, &vec![7; h * w]);
            for i in 0..h * w {
                let want = if dark[i] { 7 } else { warped[i] };
                assert_eq!(out[i], want, "{dy} {dx} px {i}");
            }
        }
    }

    #[test]
    fn replicate_warp_matches_its_scalar_reference() {
        let shifts = [
            (0.0, 0.0),
            (1.0, -3.0),
            (0.37, -1.62),
            (-2.5, 4.25),
            (5.5, -17.75),
            (-20.4, 19.6),
            (100.0, 1.0),
            (0.9999999, 2.0000002),
        ];
        for (h, w) in [(1, 1), (2, 3), (9, 17), (31, 40), (64, 63), (60, 100)] {
            let f = noise(h * w, (h * 7 + w) as u64);
            for (dy, dx) in shifts {
                assert_eq!(
                    shift_replicate(&f, h, w, dy, dx),
                    shift_replicate_scalar(&f, h, w, dy, dx),
                    "{h}x{w} {dy} {dx}"
                );
            }
        }
    }

    #[test]
    fn fill_sum_averages_the_frames_holding_data_in_any_order() {
        let (h, w) = (20, 30);
        let sh = [(0.0, 0.0), (3.0, -4.0), (-2.0, 2.0)];
        let frames: Vec<Vec<u8>> = [10u8, 200, 31].iter().map(|&v| vec![v; h * w]).collect();
        let mut a = FillSum::new(h * w);
        for (f, &(dy, dx)) in frames.iter().zip(&sh) {
            a.add(f, h, w, valid_rect(dy, dx, h, w));
        }
        let img = a.image();
        assert_eq!(img[10 * w + 10], ((10 + 200 + 31) / 3) as u8);
        assert_eq!(img[19 * w + 10], ((10 + 31) / 2) as u8);
        assert_eq!(img[10], ((10 + 200) / 2) as u8);
        assert_eq!(img[19 * w + 29], 10);
        let (mut b, mut c) = (FillSum::new(h * w), FillSum::new(h * w));
        b.add(&frames[2], h, w, valid_rect(-2.0, 2.0, h, w));
        c.add(&frames[0], h, w, valid_rect(0.0, 0.0, h, w));
        c.add(&frames[1], h, w, valid_rect(3.0, -4.0, h, w));
        b.merge(&FillSum::from_flat(&c.to_flat()).unwrap());
        assert_eq!(b.image(), img);
        let mut one = FillSum::new(h * w);
        one.add(&frames[1], h, w, valid_rect(30.0, 0.0, h, w));
        assert!(one.image().iter().all(|&v| v == 200)); // no frame holds data: every frame's mean
    }
}
