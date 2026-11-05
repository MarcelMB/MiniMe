//! The fast per-frame kernels against their scalar references on real frames (>= 200 in a
//! release build, 9 in a debug one: the references are slow there): every output must be bit for
//! bit the same. The frames come from `$MINIME_RAW` or `~/minian_testing_local/synth_v4_5min/raw`
//! (a folder of AVIs, 600 x 600 GREY); without it the test does nothing.

use minime_core::avi::{self, FileReader, ReadAt};
use minime_core::filters::{
    area_down2_u8, gauss_f32, gauss_f32_scalar, gauss_u8, gauss_u8_scalar, resize_up2_f32,
    resize_up2_f32_scalar,
};
use minime_core::preprocess::{
    median_blur, median_blur_huang, shift, shift_f32, shift_f32_scalar, shift_scalar, tophat, tophat_scalar,
};
use std::path::PathBuf;

fn raw_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("MINIME_RAW").map(PathBuf::from).or_else(|| {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("minian_testing_local/synth_v4_5min/raw"))
    })?;
    dir.is_dir().then_some(dir)
}

/// `per_file` frames spread over each AVI of `dir`.
fn load_frames(dir: &PathBuf, per_file: usize) -> Vec<(Vec<u8>, usize, usize)> {
    let mut avis: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "avi"))
        .collect();
    avis.sort();
    let mut out = Vec::new();
    for p in avis {
        let mut r = FileReader::open(&p).unwrap();
        let ix = avi::index(&mut r).unwrap();
        for i in 0..per_file {
            let (off, size) = ix.frames[(i * ix.frames.len() / per_file + 7) % ix.frames.len()];
            out.push((
                ix.unpad(&r.read_at(off, size as usize).unwrap()),
                ix.height,
                ix.width,
            ));
        }
    }
    out
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Shifts like the ones a session produces (a few pixels, sub-pixel) plus whole, tiny, border
/// and out-of-image ones.
fn shifts(i: usize) -> [(f32, f32); 4] {
    let f = |k: usize| ((i * 7919 + k * 104729) % 4001) as f32 / 4000.0;
    [
        ((f(1) - 0.5) * 30.0, (f(2) - 0.5) * 30.0),
        ((f(3) - 0.5) * 3.0, (f(4) - 0.5) * 3.0),
        [
            (0.0, 0.0),
            (3.0, -2.0),
            (1e-6, -1e-6),
            (19.99, -20.01),
            (0.5, 700.0),
            (-640.0, 0.25),
        ][i % 6],
        ((f(5) - 0.5) * 80.0, (f(6) - 0.5) * 80.0),
    ]
}

#[test]
fn fast_kernels_are_bit_identical_to_the_scalar_ones_on_real_frames() {
    let Some(dir) = raw_dir() else {
        eprintln!("no raw AVIs, skipped");
        return;
    };
    let per_file = if cfg!(debug_assertions) { 1 } else { 25 };
    let frames = load_frames(&dir, per_file);
    assert!(
        cfg!(debug_assertions) || frames.len() >= 200,
        "{} frames",
        frames.len()
    );
    for (i, (raw, h, w)) in frames.iter().enumerate() {
        let (h, w) = (*h, *w);
        let den = median_blur(raw, h, w, 7);
        assert_eq!(den, median_blur_huang(raw, h, w, 7), "median, frame {i}");
        for wnd in [15, 5] {
            assert_eq!(
                tophat(&den, h, w, wnd),
                tophat_scalar(&den, h, w, wnd),
                "tophat {wnd}, frame {i}"
            );
        }
        assert_eq!(
            gauss_u8(&den, h, w, 1.0),
            gauss_u8_scalar(&den, h, w, 1.0),
            "gauss_u8, frame {i}"
        );
        // the registration image's steps
        let half: Vec<f32> = area_down2_u8(&den, h, w).iter().map(|&v| f32::from(v)).collect();
        let g = gauss_f32(&half, h / 2, w / 2, 3.0);
        assert!(
            same_bits(&g, &gauss_f32_scalar(&half, h / 2, w / 2, 3.0)),
            "gauss_f32, frame {i}"
        );
        let up = resize_up2_f32(&g, h / 2, w / 2);
        assert!(
            same_bits(&up, &resize_up2_f32_scalar(&g, h / 2, w / 2)),
            "resize, frame {i}"
        );
        let reg: Vec<f32> = gauss_u8(&den, h, w, 1.0)
            .iter()
            .zip(&up)
            .map(|(&n, &b)| f32::from(n) - b)
            .collect();
        for (dy, dx) in shifts(i) {
            assert_eq!(
                shift(&den, h, w, dy, dx),
                shift_scalar(&den, h, w, dy, dx),
                "shift {dy} {dx}, frame {i}"
            );
            assert!(
                same_bits(
                    &shift_f32(&reg, h, w, dy, dx),
                    &shift_f32_scalar(&reg, h, w, dy, dx)
                ),
                "shift_f32 {dy} {dx}, frame {i}"
            );
        }
    }
}
