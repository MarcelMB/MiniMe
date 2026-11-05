//! The auto-tune's data-driven start: the seed-contrast histogram and its knee
//! (`seeds.contrast_hist`, `seeds.contrast_knee`, `sweep.start_diff_thres`).

use crate::seeds::{rings, rolling_windows, window_block_ranges};

/// `seeds.window_ranges` on one window's own frames: the block ranges of its seed windows.
pub fn seed_window_ranges(
    edges: &[usize],
    t: usize,
    wnd: usize,
    stp: usize,
) -> Result<Vec<(usize, usize)>, String> {
    let windows = rolling_windows(t, wnd, stp);
    if windows.is_empty() {
        return Err(format!("no {wnd}-frame seed window fits in {t} frames"));
    }
    window_block_ranges(edges, &windows)
}

/// `ranges[::max(1, len(ranges) // n)]`.
pub fn sample_ranges(ranges: &[(usize, usize)], n: usize) -> Vec<(usize, usize)> {
    let step = (ranges.len() / n).max(1);
    ranges.iter().copied().step_by(step).collect()
}

/// Histogram (whole grey levels) of the local-max contrast `local_max_union` compares with
/// `diff`: per pixel and window, the largest `max - min` over the disk kernels of radius
/// `k0..k1` in which the pixel is the local maximum. A pixel is a seed candidate at `diff`
/// iff its contrast > `diff`.
pub fn contrast_hist(
    block_max: &[u8],
    h: usize,
    w: usize,
    ranges: &[(usize, usize)],
    k0: usize,
    k1: usize,
) -> Vec<i64> {
    let mut hist = vec![0i64; 256];
    let rg = rings(k1 - 1);
    let mut fm = vec![0u8; h * w];
    for &(i0, i1) in ranges {
        fm.copy_from_slice(&block_max[i0 * h * w..(i0 + 1) * h * w]);
        for b in i0 + 1..i1 {
            for (m, &v) in fm.iter_mut().zip(&block_max[b * h * w..(b + 1) * h * w]) {
                *m = (*m).max(v);
            }
        }
        let ring_range = |y: usize, x: usize, ring: &[(isize, isize)]| {
            let (mut mx, mut mn) = (0u8, 255u8);
            for &(dy, dx) in ring {
                let (yy, xx) = (y as isize + dy, x as isize + dx);
                if yy >= 0 && xx >= 0 && (yy as usize) < h && (xx as usize) < w {
                    let v = fm[yy as usize * w + xx as usize];
                    mx = mx.max(v);
                    mn = mn.min(v);
                }
            }
            (mx, mn)
        };
        for y in 0..h {
            for x in 0..w {
                let v = fm[y * w + x];
                let mut mn = v;
                let mut is_max = true;
                for ring in &rg[1..=k0] {
                    let (rmx, rmn) = ring_range(y, x, ring);
                    if rmx > v {
                        is_max = false;
                        break;
                    }
                    mn = mn.min(rmn);
                }
                if !is_max {
                    continue;
                }
                for ring in &rg[k0 + 1..k1] {
                    let (rmx, rmn) = ring_range(y, x, ring);
                    if rmx > v {
                        break;
                    }
                    mn = mn.min(rmn);
                }
                hist[(v - mn) as usize] += 1;
            }
        }
    }
    while hist.len() > 1 && *hist.last().unwrap() == 0 {
        hist.pop();
    }
    hist
}

/// The lowest contrast `c` from which the count of local maxima falls by less than `ratio`
/// per grey level for two levels in a row; None if the histogram never flattens.
pub fn contrast_knee(hist: &[i64], ratio: f64) -> Option<usize> {
    let h: Vec<f64> = hist.iter().map(|&v| v as f64).collect();
    (1..h.len().saturating_sub(2))
        .find(|&c| h[c + 1] > 0.0 && h[c + 2] > 0.0 && h[c] / h[c + 1] < ratio && h[c + 1] / h[c + 2] < ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seeds::local_max_union;

    #[test]
    fn contrast_knee_finds_where_noise_stops_dominating() {
        // noise: count falls 4x per grey level up to 5; from 5 on it falls < 1.5x (cells)
        let hist = [
            4i64.pow(8),
            4i64.pow(7),
            4i64.pow(6),
            4i64.pow(5),
            4i64.pow(4),
            4i64.pow(3),
            50,
            40,
            32,
            26,
            21,
        ];
        assert_eq!(contrast_knee(&hist, 1.5), Some(5));
        assert_eq!(contrast_knee(&[1000, 100, 10, 1], 1.5), None);
    }

    #[test]
    fn contrast_hist_matches_the_seed_rule() {
        // a pixel is a seed candidate at diff iff its contrast > diff: the histogram's tail above
        // diff counts the local maxima local_max_union keeps in the same windows
        let (h, w) = (24, 24);
        let mut bm = vec![0u8; 3 * h * w];
        let mut s = 7u32;
        for v in bm.iter_mut() {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((s >> 16) % 40) as u8;
        }
        bm[5 * w + 5] = 200;
        bm[h * w + 15 * w + 15] = 120;
        let ranges = [(0usize, 2usize), (1, 3)];
        let hist = contrast_hist(&bm, h, w, &ranges, 2, 6);
        for diff in [1.0, 3.0, 10.0] {
            let mut n = 0;
            for &(i0, i1) in &ranges {
                let mut fm = bm[i0 * h * w..(i0 + 1) * h * w].to_vec();
                for b in i0 + 1..i1 {
                    for (m, &v) in fm.iter_mut().zip(&bm[b * h * w..(b + 1) * h * w]) {
                        *m = (*m).max(v);
                    }
                }
                n += local_max_union(&fm, h, w, 2, 6, diff)
                    .iter()
                    .filter(|&&m| m)
                    .count() as i64;
            }
            let tail: i64 = hist.iter().skip(diff as usize + 1).sum();
            assert_eq!(tail, n, "diff {diff}");
        }
    }

    #[test]
    fn sample_ranges_takes_every_nth() {
        let r: Vec<(usize, usize)> = (0..40).map(|i| (i, i + 1)).collect();
        assert_eq!(sample_ranges(&r, 16).len(), 20); // 40 // 16 = 2 -> every second
        assert_eq!(sample_ranges(&r[..5], 16).len(), 5);
    }
}
