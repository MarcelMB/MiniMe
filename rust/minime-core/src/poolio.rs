//! Byte layout of a pass-2 tile's factors, for handing a tile compressed by a pool worker to
//! the thread that owns the window's `Pass2Engine`: a header of eight little-endian u32
//! (rank, k, and the lengths of `w`, `v`, `s`, `mean`, `sigma`, `energy`) and the arrays.

use crate::compress::TileFactors;

fn put32(out: &mut Vec<u8>, a: &[f32]) {
    for v in a {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

impl TileFactors {
    pub fn to_bytes(&self) -> Vec<u8> {
        let lens = [
            self.rank,
            self.k,
            self.w.len(),
            self.v.len(),
            self.s.len(),
            self.mean.len(),
            self.sigma.len(),
            self.energy.len(),
        ];
        let mut out = Vec::with_capacity(
            32 + 4 * (self.w.len() + self.v.len() + self.s.len() + self.mean.len() + self.sigma.len())
                + 8 * self.energy.len(),
        );
        for l in lens {
            out.extend_from_slice(&(l as u32).to_le_bytes());
        }
        for a in [&self.w, &self.v, &self.s, &self.mean, &self.sigma] {
            put32(&mut out, a);
        }
        for v in &self.energy {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        let head = |i: usize| -> Result<usize, String> {
            let s = b.get(4 * i..4 * i + 4).ok_or("tile factors: short header")?;
            Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
        };
        let lens: Vec<usize> = (0..8).map(head).collect::<Result<_, _>>()?;
        // the five float32 arrays, then energy as float64
        let total = 32 + 4 * (lens[2] + lens[3] + lens[4] + lens[5] + lens[6]) + 8 * lens[7];
        if b.len() != total {
            return Err(format!("tile factors: {} bytes, expected {total}", b.len()));
        }
        let mut at = 32;
        let mut f32s = |n: usize| -> Vec<f32> {
            let v = b[at..at + 4 * n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            at += 4 * n;
            v
        };
        let (w, v, s, mean, sigma) = (
            f32s(lens[2]),
            f32s(lens[3]),
            f32s(lens[4]),
            f32s(lens[5]),
            f32s(lens[6]),
        );
        let energy = b[at..at + 8 * lens[7]]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
        Ok(Self {
            w,
            v,
            s,
            mean,
            sigma,
            energy,
            rank: lens[0],
            k: lens[1],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_factors_round_trip_bit_for_bit() {
        let tf = TileFactors {
            w: vec![1.5, -0.0, f32::MIN_POSITIVE, 3.25],
            v: vec![0.1, 0.2, 0.3],
            s: vec![9.0, 4.0],
            mean: vec![7.0; 5],
            sigma: vec![0.5; 5],
            energy: vec![1e300, 2.5, f64::from_bits(0x7ff8_0000_0000_0001)],
            rank: 2,
            k: 40,
        };
        let back = TileFactors::from_bytes(&tf.to_bytes()).unwrap();
        let bits32 = |a: &[f32]| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits32(&back.w), bits32(&tf.w));
        assert_eq!(bits32(&back.v), bits32(&tf.v));
        assert_eq!(bits32(&back.s), bits32(&tf.s));
        assert_eq!(bits32(&back.mean), bits32(&tf.mean));
        assert_eq!(bits32(&back.sigma), bits32(&tf.sigma));
        let bits64 = |a: &[f64]| a.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits64(&back.energy), bits64(&tf.energy));
        assert_eq!((back.rank, back.k), (2, 40));
        assert!(TileFactors::from_bytes(&tf.to_bytes()[1..]).is_err());
    }
}
