//! Small dense linear algebra (row-major `n x n` `f64`): Cholesky, a cyclic Jacobi
//! eigensolver for symmetric matrices, and the normal-equation solver rule of
//! `init._solve_normal` (Cholesky unless `cond > 1e12`, else the pseudo-inverse).

/// `s + x y`, fused where the target has an FMA instruction. wasm32 has none, so
/// `f64::mul_add` there is a software routine (15.6 ns, 17 times a multiply and an add, on
/// M1 Max in Chrome); `cnmf` summed 54 M of them in one Gram matrix, 0.84 s of a 1.1 s
/// spatial update. The unfused sum differs from the fused one by rounding (1e-16 relative
/// per term), far below any decision's margin.
#[inline(always)]
pub fn madd64(x: f64, y: f64, s: f64) -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        s + x * y
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        x.mul_add(y, s)
    }
}

/// `x.madd(y, s)` for float64: `madd64` as a method, for the places that read `x.mul_add(y, s)`.
pub trait Madd {
    fn madd(self, y: f64, s: f64) -> f64;
}

impl Madd for f64 {
    #[inline(always)]
    fn madd(self, y: f64, s: f64) -> f64 {
        madd64(self, y, s)
    }
}

pub const COND_MAX: f64 = 1e12;
pub const PINV_RCOND: f64 = 1e-12;

/// Lower Cholesky factor of `a`, or `None` if `a` is not positive definite.
pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d <= 0.0 || !d.is_finite() {
            return None;
        }
        let d = d.sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / d;
        }
    }
    Some(l)
}

/// Solve `L L^T x = b` in place.
pub fn cho_solve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// Eigenvalues (ascending) and eigenvectors (column `j` of the row-major result is the
/// vector of eigenvalue `j`) of the symmetric matrix `a`, by cyclic Jacobi rotations.
pub fn sym_eig(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let norm: f64 = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| m[i * n + j] * m[i * n + j])
            .sum::<f64>()
            .sqrt();
        if off <= 1e-15 * norm || off == 0.0 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let mkp = m[k * n + p];
                    let mkq = m[k * n + q];
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p * n + k];
                    let mqk = m[q * n + k];
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| m[i * n + i].total_cmp(&m[j * n + j]));
    let evals = order.iter().map(|&i| m[i * n + i]).collect();
    let mut evecs = vec![0.0; n * n];
    for (jn, &jo) in order.iter().enumerate() {
        for k in 0..n {
            evecs[k * n + jn] = v[k * n + jo];
        }
    }
    (evals, evecs)
}

/// How far below `COND_MAX` a bound must be for `NormalSolver::certified` to skip the eigensolve.
pub const CERTIFY_MARGIN: f64 = 100.0;

/// An upper bound on the 2-norm condition number of the symmetric positive definite `g` from its
/// Cholesky factor `l`: for a symmetric matrix the 2-norm is at most the infinity norm, and
/// `G^-1 = L^-T L^-1`, so `cond(G) <= |G|_inf |L^-1|_1 |L^-1|_inf` (rounding in `L^-1` is
/// `cond * eps` relative, far inside the margin it is used with).
pub fn cond_bound(g: &[f64], l: &[f64], n: usize) -> f64 {
    let g_inf = (0..n)
        .map(|i| g[i * n..(i + 1) * n].iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    // L^-1, lower triangular, column by column (forward substitution on e_j)
    let mut inv = vec![0.0; n * n];
    for j in 0..n {
        inv[j * n + j] = 1.0 / l[j * n + j];
        for i in j + 1..n {
            let mut s = 0.0;
            for k in j..i {
                s += l[i * n + k] * inv[k * n + j];
            }
            inv[i * n + j] = -s / l[i * n + i];
        }
    }
    let row_inf = (0..n)
        .map(|i| inv[i * n..i * n + i + 1].iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let col_1 = (0..n)
        .map(|j| (j..n).map(|i| inv[i * n + j].abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let b = g_inf * row_inf * col_1;
    if b.is_finite() {
        b
    } else {
        f64::INFINITY
    }
}

/// `B -> G^-1 B` from one factorisation of the symmetric `G` (`init._solve_normal`).
pub enum NormalSolver {
    Chol { l: Vec<f64>, n: usize, cond: f64 },
    Pinv { gi: Vec<f64>, n: usize, cond: f64 },
}

impl NormalSolver {
    pub fn new(g: &[f64], n: usize) -> Self {
        let (evals, evecs) = sym_eig(g, n);
        let amax = evals.iter().fold(0.0f64, |m, &e| m.max(e.abs()));
        let amin = evals.iter().fold(f64::INFINITY, |m, &e| m.min(e.abs()));
        let cond = if amin > 0.0 { amax / amin } else { f64::INFINITY };
        if cond <= COND_MAX {
            if let Some(l) = cholesky(g, n) {
                return Self::Chol { l, n, cond };
            }
        }
        let cut = PINV_RCOND * amax;
        let mut gi = vec![0.0; n * n];
        for (j, &e) in evals.iter().enumerate() {
            if e.abs() > cut {
                let inv = 1.0 / e;
                for a in 0..n {
                    let va = evecs[a * n + j] * inv;
                    for b in 0..n {
                        gi[a * n + b] += va * evecs[b * n + j];
                    }
                }
            }
        }
        Self::Pinv { gi, n, cond }
    }

    /// `new`'s solver without its eigensolve when that cannot change the outcome: if `G` has a
    /// Cholesky factor `L` and the rigorous bound `cond_bound` on `cond(G)` is at most
    /// `COND_MAX / CERTIFY_MARGIN`, `new` would also take Cholesky (its `cond` is `cond(G)` to
    /// rounding) and returns the same `L`, so the solver is the same bit for bit; otherwise this
    /// is `new`. The Jacobi eigensolve is O(n^3) per sweep (16 s at n = 827 natively); the bound is
    /// one triangular inverse (n^3 / 6). The stored `cond` is then the bound, not `cond(G)`.
    pub fn certified(g: &[f64], n: usize) -> Self {
        if let Some(l) = cholesky(g, n) {
            let bound = cond_bound(g, &l, n);
            if bound <= COND_MAX / CERTIFY_MARGIN {
                return Self::Chol { l, n, cond: bound };
            }
        }
        Self::new(g, n)
    }

    /// `np.linalg.cond(G)`: largest over smallest absolute eigenvalue.
    pub fn cond(&self) -> f64 {
        match self {
            Self::Chol { cond, .. } | Self::Pinv { cond, .. } => *cond,
        }
    }

    pub fn is_cholesky(&self) -> bool {
        matches!(self, Self::Chol { .. })
    }

    /// Solve in place.
    pub fn solve(&self, b: &mut [f64]) {
        match self {
            Self::Chol { l, n, .. } => cho_solve(l, *n, b),
            Self::Pinv { gi, n, .. } => {
                let x: Vec<f64> = (0..*n)
                    .map(|a| (0..*n).map(|c| gi[a * n + c] * b[c]).sum())
                    .collect();
                b.copy_from_slice(&x);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eig_and_solve() {
        let g = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let (e, v) = sym_eig(&g, 3);
        for j in 0..3 {
            for a in 0..3 {
                let gv: f64 = (0..3).map(|b| g[a * 3 + b] * v[b * 3 + j]).sum();
                assert!((gv - e[j] * v[a * 3 + j]).abs() < 1e-12);
            }
        }
        let s = NormalSolver::new(&g, 3);
        assert!(s.is_cholesky());
        let mut b = [1.0, 2.0, 3.0];
        s.solve(&mut b);
        for a in 0..3 {
            let gx: f64 = (0..3).map(|c| g[a * 3 + c] * b[c]).sum();
            assert!((gx - [1.0, 2.0, 3.0][a]).abs() < 1e-12);
        }
    }

    /// A random SPD matrix `B B^T + d I` (an LCG).
    fn spd(n: usize, d: f64, seed: u32) -> Vec<f64> {
        let mut x = seed;
        let b: Vec<f64> = (0..n * n)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f64::from(x >> 8) / f64::from(1u32 << 24) - 0.5
            })
            .collect();
        let mut g = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                g[i * n + j] =
                    (0..n).map(|k| b[i * n + k] * b[j * n + k]).sum::<f64>() + if i == j { d } else { 0.0 };
            }
        }
        g
    }

    #[test]
    fn the_certified_solver_is_the_jacobi_ruled_one_bit_for_bit_or_falls_back() {
        for (n, d, seed) in [(5, 1.0, 1), (40, 0.5, 2), (120, 0.1, 3)] {
            let g = spd(n, d, seed);
            let (a, b) = (NormalSolver::new(&g, n), NormalSolver::certified(&g, n));
            assert!(a.is_cholesky() && b.is_cholesky());
            // the bound is above the 2-norm condition number
            assert!(b.cond() >= a.cond() * (1.0 - 1e-9), "{} < {}", b.cond(), a.cond());
            let rhs: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
            let (mut x, mut y) = (rhs.clone(), rhs.clone());
            a.solve(&mut x);
            b.solve(&mut y);
            assert_eq!(
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        // a nearly singular matrix: no certificate, so `new`'s eigensolve decides (pinv here)
        let mut g = spd(6, 0.0, 4);
        let v: Vec<f64> = (0..6).map(|i| g[i * 6]).collect();
        for i in 0..6 {
            for j in 0..6 {
                g[i * 6 + j] = v[i] * v[j] + if i == j { 1e-15 } else { 0.0 };
            }
        }
        let (a, b) = (NormalSolver::new(&g, 6), NormalSolver::certified(&g, 6));
        assert_eq!(a.is_cholesky(), b.is_cholesky());
        assert_eq!(a.cond().to_bits(), b.cond().to_bits());
    }
}
