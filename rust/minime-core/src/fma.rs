//! Fused multiply-add where the target has none. wasm32 has no scalar FMA, so
//! `f32::mul_add` / `f64::mul_add` there are calls into a software routine, tens of times
//! slower than a multiply and an add. `cnmf` uses them for sums that must follow
//! Accelerate's FMA order bit for bit.

/// `a * b + c` with one rounding, bit for bit `f32::mul_add`.
///
/// On wasm32 the product is formed in float64 (exact: 24 + 24 bits) and added to `c`
/// there; if that sum is exact (checked with Knuth's TwoSum error term) one rounding to
/// float32 is the fused result. Otherwise (a gap of more than 28 binary orders between the
/// product and `c`, overflow, NaN) the software routine decides.
#[inline(always)]
pub fn fma32(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        let p = f64::from(a) * f64::from(b);
        let c64 = f64::from(c);
        let s = p + c64;
        let bb = s - p;
        let e = (p - (s - bb)) + (c64 - bb);
        if e == 0.0 {
            return s as f32;
        }
        a.mul_add(b, c)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        a.mul_add(b, c)
    }
}
