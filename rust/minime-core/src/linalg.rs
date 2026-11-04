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
}
