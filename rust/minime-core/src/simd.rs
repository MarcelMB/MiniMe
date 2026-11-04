//! Sixteen `u8` lanes with exactly what the median and tophat kernels need (load, store, min, max,
//! saturating subtract):
//! wasm `simd128` (the browser build is made with `+simd128`), NEON on aarch64, SSE2 on
//! x86_64, and a plain array that LLVM vectorises (or not) everywhere else. All four give the
//! same bits; the lane type only decides the speed.

#![allow(unused_unsafe)]

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
}

pub use imp::*;
