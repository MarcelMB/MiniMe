//! Branch-free float32 helpers that LLVM vectorises (wasm simd128, NEON): the natural
//! logarithm of `compress`' noise estimate.

/// `ln(x)` for normal positive `x`, Cephes' `logf` (reduction to `[sqrt(1/2), sqrt 2)`, a
/// degree-8 polynomial; peak relative error 7.6e-8). Float32 in and out, so a loop over a
/// slice compiles to vector code, which `f32::ln` (a libm call) never does.
#[inline(always)]
pub fn ln_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let mut e = ((bits >> 23) as i32 - 127) as f32;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000); // [1, 2)
    let big = m > std::f32::consts::SQRT_2;
    let m = if big { m * 0.5 } else { m };
    e += if big { 1.0 } else { 0.0 };
    let f = m - 1.0;
    let z = f * f;
    let mut y = 7.037_683_6e-2_f32;
    y = y * f - 1.151_461_0e-1;
    y = y * f + 1.167_699_9e-1;
    y = y * f - 1.242_014_1e-1;
    y = y * f + 1.424_932_3e-1;
    y = y * f - 1.666_805_8e-1;
    y = y * f + 2.000_071_5e-1;
    y = y * f - 2.499_999_4e-1;
    y = y * f + 3.333_333_1e-1;
    y = y * f * z;
    y += e * -2.121_944_4e-4;
    y -= 0.5 * z;
    f + y + e * 0.693_359_4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_matches_libm() {
        let mut worst = 0f64;
        let mut x = f32::EPSILON;
        while x < 1e7 {
            let a = f64::from(ln_f32(x));
            let b = f64::from(x).ln();
            // relative to the value, and absolute near 1 where ln crosses zero
            worst = worst.max((a - b).abs() / b.abs().max(1.0));
            x *= 1.0001;
        }
        assert!(worst < 2e-7, "{worst}");
    }
}
