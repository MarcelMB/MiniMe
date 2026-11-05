//! Sixteen `u8` lanes with exactly what the median and tophat kernels need (load, store, min, max,
//! saturating subtract):
//! wasm `simd128` (the browser build is made with `+simd128`), NEON on aarch64, SSE2 on
//! x86_64, and a plain array that LLVM vectorises (or not) everywhere else. All four give the
//! same bits; the lane type only decides the speed.

#![allow(unused_unsafe)]

/// Whether the lanes are real vector registers. Without them (a wasm build without `simd128`,
/// or a target this module has no intrinsics for) the sorting-network median is slower than the
/// scalar one, so `preprocess::median_blur` keeps Huang's filter there.
pub const NATIVE_LANES: bool = cfg!(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64",
    target_arch = "x86_64"
));

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use core::arch::wasm32 as w;
    pub type V = w::v128;

    /// # Safety
    /// `p` must point to 16 readable bytes (no alignment needed).
    #[inline(always)]
    pub unsafe fn load(p: *const u8) -> V {
        unsafe { (p as *const V).read_unaligned() }
    }

    /// # Safety
    /// `p` must point to 16 writable bytes (no alignment needed).
    #[inline(always)]
    pub unsafe fn store(p: *mut u8, v: V) {
        unsafe { (p as *mut V).write_unaligned(v) }
    }

    #[inline(always)]
    pub fn min(a: V, b: V) -> V {
        w::u8x16_min(a, b)
    }

    #[inline(always)]
    pub fn max(a: V, b: V) -> V {
        w::u8x16_max(a, b)
    }

    /// `a - b`, saturating at 0.
    #[inline(always)]
    pub fn subs(a: V, b: V) -> V {
        w::u8x16_sub_sat(a, b)
    }

    /// Sixteen rounded 2 x 2 block means `(a + b + c + d + 2) >> 2` of two rows: `a0 a1` are
    /// 32 consecutive bytes of the upper row, `b0 b1` of the lower one.
    #[inline(always)]
    pub fn mean2x2(a0: V, a1: V, b0: V, b1: V) -> V {
        let two = w::u16x8_splat(2);
        let m = |a: V, b: V| {
            let s = w::u16x8_add(
                w::u16x8_extadd_pairwise_u8x16(a),
                w::u16x8_extadd_pairwise_u8x16(b),
            );
            w::u16x8_shr(w::u16x8_add(s, two), 2)
        };
        w::u8x16_narrow_i16x8(m(a0, b0), m(a1, b1))
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use core::arch::aarch64 as a;
    pub type V = a::uint8x16_t;

    /// # Safety
    /// `p` must point to 16 readable bytes (no alignment needed).
    #[inline(always)]
    pub unsafe fn load(p: *const u8) -> V {
        unsafe { a::vld1q_u8(p) }
    }

    /// # Safety
    /// `p` must point to 16 writable bytes (no alignment needed).
    #[inline(always)]
    pub unsafe fn store(p: *mut u8, v: V) {
        unsafe { a::vst1q_u8(p, v) }
    }

    #[inline(always)]
    pub fn min(a: V, b: V) -> V {
        unsafe { core::arch::aarch64::vminq_u8(a, b) }
    }

    #[inline(always)]
    pub fn max(a: V, b: V) -> V {
        unsafe { core::arch::aarch64::vmaxq_u8(a, b) }
    }

    /// `a - b`, saturating at 0.
    #[inline(always)]
    pub fn subs(a: V, b: V) -> V {
        unsafe { core::arch::aarch64::vqsubq_u8(a, b) }
    }

    /// Sixteen rounded 2 x 2 block means `(a + b + c + d + 2) >> 2` of two rows: `a0 a1` are
    /// 32 consecutive bytes of the upper row, `b0 b1` of the lower one.
    #[inline(always)]
    pub fn mean2x2(a0: V, a1: V, b0: V, b1: V) -> V {
        unsafe {
            let m = |a: V, b: V| {
                let s = a::vaddq_u16(a::vpaddlq_u8(a), a::vpaddlq_u8(b));
                a::vshrq_n_u16::<2>(a::vaddq_u16(s, a::vdupq_n_u16(2)))
            };
            a::vcombine_u8(a::vmovn_u16(m(a0, b0)), a::vmovn_u16(m(a1, b1)))
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::x86_64 as x;
    pub type V = x::__m128i;

    /// # Safety
    /// `p` must point to 16 readable bytes (no alignment needed).
    #[inline(always)]
    pub unsafe fn load(p: *const u8) -> V {
        unsafe { x::_mm_loadu_si128(p as *const V) }
    }

    /// # Safety
    /// `p` must point to 16 writable bytes (no alignment needed).
    #[inline(always)]
    pub unsafe fn store(p: *mut u8, v: V) {
        unsafe { x::_mm_storeu_si128(p as *mut V, v) }
    }

    #[inline(always)]
    pub fn min(a: V, b: V) -> V {
        unsafe { x::_mm_min_epu8(a, b) }
    }

    #[inline(always)]
    pub fn max(a: V, b: V) -> V {
        unsafe { x::_mm_max_epu8(a, b) }
    }

    /// `a - b`, saturating at 0.
    #[inline(always)]
    pub fn subs(a: V, b: V) -> V {
        unsafe { x::_mm_subs_epu8(a, b) }
    }

    /// Sixteen rounded 2 x 2 block means `(a + b + c + d + 2) >> 2` of two rows: `a0 a1` are
    /// 32 consecutive bytes of the upper row, `b0 b1` of the lower one.
    #[inline(always)]
    pub fn mean2x2(a0: V, a1: V, b0: V, b1: V) -> V {
        let (mut a, mut b) = ([0u8; 32], [0u8; 32]);
        unsafe {
            store(a.as_mut_ptr(), a0);
            store(a.as_mut_ptr().add(16), a1);
            store(b.as_mut_ptr(), b0);
            store(b.as_mut_ptr().add(16), b1);
        }
        let out: [u8; 16] = std::array::from_fn(|i| {
            let s =
                u16::from(a[2 * i]) + u16::from(a[2 * i + 1]) + u16::from(b[2 * i]) + u16::from(b[2 * i + 1]);
            ((s + 2) >> 2) as u8
        });
        unsafe { load(out.as_ptr()) }
    }
}

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "aarch64",
    target_arch = "x86_64"
)))]
mod imp {
    pub type V = [u8; 16];

    /// # Safety
    /// `p` must point to 16 readable bytes.
    #[inline(always)]
    pub unsafe fn load(p: *const u8) -> V {
        unsafe { (p as *const V).read_unaligned() }
    }

    /// # Safety
    /// `p` must point to 16 writable bytes.
    #[inline(always)]
    pub unsafe fn store(p: *mut u8, v: V) {
        unsafe { (p as *mut V).write_unaligned(v) }
    }

    #[inline(always)]
    pub fn min(a: V, b: V) -> V {
        std::array::from_fn(|i| a[i].min(b[i]))
    }

    #[inline(always)]
    pub fn max(a: V, b: V) -> V {
        std::array::from_fn(|i| a[i].max(b[i]))
    }

    /// `a - b`, saturating at 0.
    #[inline(always)]
    pub fn subs(a: V, b: V) -> V {
        std::array::from_fn(|i| a[i].saturating_sub(b[i]))
    }

    /// Sixteen rounded 2 x 2 block means `(a + b + c + d + 2) >> 2` of two rows: `a0 a1` are
    /// 32 consecutive bytes of the upper row, `b0 b1` of the lower one.
    #[inline(always)]
    pub fn mean2x2(a0: V, a1: V, b0: V, b1: V) -> V {
        let (mut a, mut b) = ([0u8; 32], [0u8; 32]);
        unsafe {
            store(a.as_mut_ptr(), a0);
            store(a.as_mut_ptr().add(16), a1);
            store(b.as_mut_ptr(), b0);
            store(b.as_mut_ptr().add(16), b1);
        }
        let out: [u8; 16] = std::array::from_fn(|i| {
            let s =
                u16::from(a[2 * i]) + u16::from(a[2 * i + 1]) + u16::from(b[2 * i]) + u16::from(b[2 * i + 1]);
            ((s + 2) >> 2) as u8
        });
        unsafe { load(out.as_ptr()) }
    }
}

pub use imp::*;
