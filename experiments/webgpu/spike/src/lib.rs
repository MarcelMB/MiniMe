//! Throwaway spike: MiniMe's per-frame preprocessing on the CPU (the code the app runs, included
//! as is) and on the GPU through the `wgpu` crate, in a browser worker (WebGPU backend) and
//! natively (Metal), with the same WGSL as the hand-written TypeScript driver.

#[path = "../../../../rust/minime-core/src/preprocess.rs"]
#[allow(dead_code)]
pub mod preprocess;

#[path = "../../../../rust/minime-core/src/compress.rs"]
#[allow(dead_code)]
pub mod compress;
#[path = "../../../../rust/minime-core/src/linalg.rs"]
#[allow(dead_code)]
pub mod linalg;
#[path = "../../../../rust/minime-core/src/npy.rs"]
#[allow(dead_code)]
pub mod npy;
pub mod p2;

#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(target_arch = "wasm32")]
mod wasm;

/// Median 7, shift, tophat 15 of `n` frames (`h * w` each) on the CPU, like `Pass1::push`.
pub fn cpu_chain_into(frames: &[u8], shifts: &[f32], n: usize, h: usize, w: usize, out: &mut [u8]) {
    let prep = preprocess::Prep {
        median_ksize: 7,
        tophat_wnd: 15,
        motion_before_tophat: true,
    };
    for i in 0..n {
        let f = prep.frame(
            &frames[i * h * w..(i + 1) * h * w],
            h,
            w,
            shifts[2 * i],
            shifts[2 * i + 1],
        );
        out[i * h * w..(i + 1) * h * w].copy_from_slice(&f);
    }
}
