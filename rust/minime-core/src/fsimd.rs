//! Four `f32` lanes and the exact emulation of `preprocess::fma` on them, for the filters
//! that fuse multiply-adds in OpenCV's order (`gauss_f32`, `resize_up2_f32`, the sub-pixel
//! shift).
//!
//! `fma(a, b, c) = ((a as f64) * (b as f64) + (c as f64)) as f32` is not the hardware
//! instruction (wasm has none that is guaranteed fused, and the double rounding differs from a
//! true fma in rare corner cases), so the lanes do exactly that sum in `f64` pairs, with the
//! same IEEE operations in the same order as the scalar code: the results are bit for bit
//! those of `preprocess::fma`. wasm `simd128` and aarch64 NEON have it natively; any other
//! target runs the same arithmetic one lane at a time.

#![allow(unused_unsafe)]

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use core::arch::wasm32 as w;
    pub type F4 = w::v128;
    pub type D2 = w::v128;

    /// # Safety
    /// `p` must point to 4 readable `f32`s (no alignment needed).
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> F4 {
        unsafe { (p as *const F4).read_unaligned() }
    }

    /// # Safety
    /// `p` must point to 4 writable `f32`s (no alignment needed).
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: F4) {
        unsafe { (p as *mut F4).write_unaligned(v) }
    }

    #[inline(always)]
    pub fn splat(x: f32) -> F4 {
        w::f32x4_splat(x)
    }

    #[inline(always)]
    pub fn add(a: F4, b: F4) -> F4 {
        w::f32x4_add(a, b)
    }

    #[inline(always)]
    pub fn sub(a: F4, b: F4) -> F4 {
        w::f32x4_sub(a, b)
    }

    #[inline(always)]
    pub fn mul(a: F4, b: F4) -> F4 {
        w::f32x4_mul(a, b)
    }

    #[inline(always)]
    pub fn lo(v: F4) -> D2 {
        w::f64x2_promote_low_f32x4(v)
    }

    #[inline(always)]
    pub fn hi(v: F4) -> D2 {
        w::f64x2_promote_low_f32x4(w::i32x4_shuffle::<2, 3, 2, 3>(v, v))
    }

    #[inline(always)]
    pub fn d_splat(x: f64) -> D2 {
        w::f64x2_splat(x)
    }

    #[inline(always)]
    pub fn d_add(a: D2, b: D2) -> D2 {
        w::f64x2_add(a, b)
    }

    #[inline(always)]
    pub fn d_mul(a: D2, b: D2) -> D2 {
        w::f64x2_mul(a, b)
    }

    /// The two pairs rounded to `f32`, as four lanes.
    #[inline(always)]
    pub fn pack(lo: D2, hi: D2) -> F4 {
        w::i32x4_shuffle::<0, 1, 4, 5>(w::f32x4_demote_f64x2_zero(lo), w::f32x4_demote_f64x2_zero(hi))
    }

    /// Each lane rounded to `f32` and widened again.
    #[inline(always)]
    pub fn d_round(x: D2) -> D2 {
        w::f64x2_promote_low_f32x4(w::f32x4_demote_f64x2_zero(x))
    }

    /// The even-indexed lanes of `a b` (eight consecutive floats): `a0 a2 b0 b2`.
    #[inline(always)]
    pub fn even(a: F4, b: F4) -> F4 {
        w::i32x4_shuffle::<0, 2, 4, 6>(a, b)
    }

    /// The odd-indexed lanes of `a b`: `a1 a3 b1 b3`.
    #[inline(always)]
    pub fn odd(a: F4, b: F4) -> F4 {
        w::i32x4_shuffle::<1, 3, 5, 7>(a, b)
    }

    /// Interleaves the low halves: `a0 b0 a1 b1`.
    #[inline(always)]
    pub fn zip_lo(a: F4, b: F4) -> F4 {
        w::i32x4_shuffle::<0, 4, 1, 5>(a, b)
    }

    /// Interleaves the high halves: `a2 b2 a3 b3`.
    #[inline(always)]
    pub fn zip_hi(a: F4, b: F4) -> F4 {
        w::i32x4_shuffle::<2, 6, 3, 7>(a, b)
    }

    /// The sixteen bytes as floats.
    #[inline(always)]
    pub fn widen16(b: &[u8; 16]) -> [F4; 4] {
        // SAFETY: `b` is 16 readable bytes.
        let v = unsafe { (b.as_ptr() as *const w::v128).read_unaligned() };
        let (l, h) = (w::u16x8_extend_low_u8x16(v), w::u16x8_extend_high_u8x16(v));
        [
            w::f32x4_convert_u32x4(w::u32x4_extend_low_u16x8(l)),
            w::f32x4_convert_u32x4(w::u32x4_extend_high_u16x8(l)),
            w::f32x4_convert_u32x4(w::u32x4_extend_low_u16x8(h)),
            w::f32x4_convert_u32x4(w::u32x4_extend_high_u16x8(h)),
        ]
    }

    /// Sixteen floats rounded half to even and clamped to `0..=255` (NaN is 0), as bytes.
    #[inline(always)]
    pub fn narrow16(v: [F4; 4]) -> [u8; 16] {
        let i = |x: F4| w::i32x4_trunc_sat_f32x4(w::f32x4_nearest(x));
        let n = w::u8x16_narrow_i16x8(
            w::i16x8_narrow_i32x4(i(v[0]), i(v[1])),
            w::i16x8_narrow_i32x4(i(v[2]), i(v[3])),
        );
        let mut out = [0u8; 16];
        // SAFETY: `out` is 16 writable bytes.
        unsafe { (out.as_mut_ptr() as *mut w::v128).write_unaligned(n) };
        out
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use core::arch::aarch64 as a;
    pub type F4 = a::float32x4_t;
    pub type D2 = a::float64x2_t;

    /// # Safety
    /// `p` must point to 4 readable `f32`s (no alignment needed).
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> F4 {
        unsafe { a::vld1q_f32(p) }
    }

    /// # Safety
    /// `p` must point to 4 writable `f32`s (no alignment needed).
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: F4) {
        unsafe { a::vst1q_f32(p, v) }
    }

    #[inline(always)]
    pub fn splat(x: f32) -> F4 {
        unsafe { a::vdupq_n_f32(x) }
    }

    #[inline(always)]
    pub fn add(x: F4, y: F4) -> F4 {
        unsafe { a::vaddq_f32(x, y) }
    }

    #[inline(always)]
    pub fn sub(x: F4, y: F4) -> F4 {
        unsafe { a::vsubq_f32(x, y) }
    }

    #[inline(always)]
    pub fn mul(x: F4, y: F4) -> F4 {
        unsafe { a::vmulq_f32(x, y) }
    }

    #[inline(always)]
    pub fn lo(v: F4) -> D2 {
        unsafe { a::vcvt_f64_f32(a::vget_low_f32(v)) }
    }

    #[inline(always)]
    pub fn hi(v: F4) -> D2 {
        unsafe { a::vcvt_high_f64_f32(v) }
    }

    #[inline(always)]
    pub fn d_splat(x: f64) -> D2 {
        unsafe { a::vdupq_n_f64(x) }
    }

    #[inline(always)]
    pub fn d_add(x: D2, y: D2) -> D2 {
        unsafe { a::vaddq_f64(x, y) }
    }

    #[inline(always)]
    pub fn d_mul(x: D2, y: D2) -> D2 {
        unsafe { a::vmulq_f64(x, y) }
    }

    /// The two pairs rounded to `f32`, as four lanes.
    #[inline(always)]
    pub fn pack(l: D2, h: D2) -> F4 {
        unsafe { a::vcvt_high_f32_f64(a::vcvt_f32_f64(l), h) }
    }

    /// Each lane rounded to `f32` and widened again.
    #[inline(always)]
    pub fn d_round(x: D2) -> D2 {
        unsafe { a::vcvt_f64_f32(a::vcvt_f32_f64(x)) }
    }

    /// The even-indexed lanes of `x y` (eight consecutive floats): `x0 x2 y0 y2`.
    #[inline(always)]
    pub fn even(x: F4, y: F4) -> F4 {
        unsafe { a::vuzp1q_f32(x, y) }
    }

    /// The odd-indexed lanes of `x y`: `x1 x3 y1 y3`.
    #[inline(always)]
    pub fn odd(x: F4, y: F4) -> F4 {
        unsafe { a::vuzp2q_f32(x, y) }
    }

    /// Interleaves the low halves: `a0 b0 a1 b1`.
    #[inline(always)]
    pub fn zip_lo(x: F4, y: F4) -> F4 {
        unsafe { a::vzip1q_f32(x, y) }
    }

    /// Interleaves the high halves: `a2 b2 a3 b3`.
    #[inline(always)]
    pub fn zip_hi(x: F4, y: F4) -> F4 {
        unsafe { a::vzip2q_f32(x, y) }
    }

    /// The sixteen bytes as floats.
    #[inline(always)]
    pub fn widen16(b: &[u8; 16]) -> [F4; 4] {
        // SAFETY: `b` is 16 readable bytes.
        unsafe {
            let v = a::vld1q_u8(b.as_ptr());
            let (l, h) = (a::vmovl_u8(a::vget_low_u8(v)), a::vmovl_high_u8(v));
            [
                a::vcvtq_f32_u32(a::vmovl_u16(a::vget_low_u16(l))),
                a::vcvtq_f32_u32(a::vmovl_high_u16(l)),
                a::vcvtq_f32_u32(a::vmovl_u16(a::vget_low_u16(h))),
                a::vcvtq_f32_u32(a::vmovl_high_u16(h)),
            ]
        }
    }

    /// Sixteen floats rounded half to even and clamped to `0..=255` (NaN is 0), as bytes.
    #[inline(always)]
    pub fn narrow16(v: [F4; 4]) -> [u8; 16] {
        unsafe {
            let i = |x: F4| a::vcvtq_s32_f32(a::vrndnq_f32(x));
            let lo = a::vcombine_s16(a::vqmovn_s32(i(v[0])), a::vqmovn_s32(i(v[1])));
            let hi = a::vcombine_s16(a::vqmovn_s32(i(v[2])), a::vqmovn_s32(i(v[3])));
            let n = a::vcombine_u8(a::vqmovun_s16(lo), a::vqmovun_s16(hi));
            let mut out = [0u8; 16];
            a::vst1q_u8(out.as_mut_ptr(), n);
            out
        }
    }
}

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64"
)))]
mod imp {
    pub type F4 = [f32; 4];
    pub type D2 = [f64; 2];

    /// # Safety
    /// `p` must point to 4 readable `f32`s.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> F4 {
        unsafe { (p as *const F4).read_unaligned() }
    }

    /// # Safety
    /// `p` must point to 4 writable `f32`s.
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: F4) {
        unsafe { (p as *mut F4).write_unaligned(v) }
    }

    #[inline(always)]
    pub fn splat(x: f32) -> F4 {
        [x; 4]
    }

    #[inline(always)]
    pub fn add(a: F4, b: F4) -> F4 {
        std::array::from_fn(|i| a[i] + b[i])
    }

    #[inline(always)]
    pub fn sub(a: F4, b: F4) -> F4 {
        std::array::from_fn(|i| a[i] - b[i])
    }

    #[inline(always)]
    pub fn mul(a: F4, b: F4) -> F4 {
        std::array::from_fn(|i| a[i] * b[i])
    }

    #[inline(always)]
    pub fn lo(v: F4) -> D2 {
        [f64::from(v[0]), f64::from(v[1])]
    }

    #[inline(always)]
    pub fn hi(v: F4) -> D2 {
        [f64::from(v[2]), f64::from(v[3])]
    }

    #[inline(always)]
    pub fn d_splat(x: f64) -> D2 {
        [x; 2]
    }

    #[inline(always)]
    pub fn d_add(a: D2, b: D2) -> D2 {
        [a[0] + b[0], a[1] + b[1]]
    }

    #[inline(always)]
    pub fn d_mul(a: D2, b: D2) -> D2 {
        [a[0] * b[0], a[1] * b[1]]
    }

    #[inline(always)]
    pub fn pack(lo: D2, hi: D2) -> F4 {
        [lo[0] as f32, lo[1] as f32, hi[0] as f32, hi[1] as f32]
    }

    #[inline(always)]
    pub fn d_round(x: D2) -> D2 {
        [f64::from(x[0] as f32), f64::from(x[1] as f32)]
    }

    #[inline(always)]
    pub fn even(a: F4, b: F4) -> F4 {
        [a[0], a[2], b[0], b[2]]
    }

    #[inline(always)]
    pub fn odd(a: F4, b: F4) -> F4 {
        [a[1], a[3], b[1], b[3]]
    }

    #[inline(always)]
    pub fn zip_lo(a: F4, b: F4) -> F4 {
        [a[0], b[0], a[1], b[1]]
    }

    #[inline(always)]
    pub fn zip_hi(a: F4, b: F4) -> F4 {
        [a[2], b[2], a[3], b[3]]
    }

    /// The sixteen bytes as floats.
    #[inline(always)]
    pub fn widen16(b: &[u8; 16]) -> [F4; 4] {
        std::array::from_fn(|i| std::array::from_fn(|j| f32::from(b[4 * i + j])))
    }

    /// Sixteen floats rounded half to even and clamped to `0..=255` (NaN is 0), as bytes.
    #[inline(always)]
    pub fn narrow16(v: [F4; 4]) -> [u8; 16] {
        std::array::from_fn(|i| v[i / 4][i % 4].round_ties_even().clamp(0.0, 255.0) as u8)
    }
}

pub use imp::{
    add, d_add, d_mul, d_round, d_splat, even, hi, lo, load, mul, narrow16, odd, pack, splat, store, sub,
    widen16, zip_hi, zip_lo, D2, F4,
};

/// Four `f32` values held as two `f64` pairs (every lane exactly an `f32`'s value when it comes
/// from `from_f4` or `round32`).
#[derive(Clone, Copy)]
pub struct D4(D2, D2);

impl D4 {
    #[inline(always)]
    pub fn from_f4(v: F4) -> D4 {
        D4(lo(v), hi(v))
    }

    #[inline(always)]
    pub fn splat(x: f32) -> D4 {
        D4(d_splat(f64::from(x)), d_splat(f64::from(x)))
    }

    /// Each lane rounded to `f32` (kept as an `f64`).
    #[inline(always)]
    pub fn round32(self) -> D4 {
        D4(d_round(self.0), d_round(self.1))
    }

    /// The four lanes rounded to `f32`.
    #[inline(always)]
    pub fn to_f4(self) -> F4 {
        pack(self.0, self.1)
    }
}

/// The `f64` product (exact for two `f32`s).
impl std::ops::Mul for D4 {
    type Output = D4;

    #[inline(always)]
    fn mul(self, o: D4) -> D4 {
        D4(d_mul(self.0, o.0), d_mul(self.1, o.1))
    }
}

/// The `f64` sum (one rounding, to 53 bits).
impl std::ops::Add for D4 {
    type Output = D4;

    #[inline(always)]
    fn add(self, o: D4) -> D4 {
        D4(d_add(self.0, o.0), d_add(self.1, o.1))
    }
}

/// `preprocess::fma` on four lanes.
#[inline(always)]
pub fn fma4(a: F4, b: F4, c: F4) -> F4 {
    (D4::from_f4(a) * D4::from_f4(b) + D4::from_f4(c)).to_f4()
}

/// `a[i..i + 4]` as a vector. The caller guarantees `i + 4 <= a.len()`.
#[inline(always)]
pub fn ld(a: &[f32], i: usize) -> F4 {
    assert!(i + 4 <= a.len());
    // SAFETY: bounds asserted just above (the check is one compare per vector).
    unsafe { load(a.as_ptr().add(i)) }
}

/// Stores the four lanes at `a[i..i + 4]`.
#[inline(always)]
pub fn st(a: &mut [f32], i: usize, v: F4) {
    assert!(i + 4 <= a.len());
    // SAFETY: bounds asserted just above.
    unsafe { store(a.as_mut_ptr().add(i), v) }
}
