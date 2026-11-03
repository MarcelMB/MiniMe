//! `minime.factors.Factors` of one window: pass 2's per-tile factors (`Y_b ~ W_b V_b^T +
//! mean_b`) plus pass 1's max projection and block maxima, which every later stage reads.

use crate::compress::TileFactors;
use crate::pass1::{tile_grid, Tile};

pub struct Factors {
    pub h: usize,
    pub w: usize,
    pub t: usize,
    pub grid: Vec<Tile>,
    pub tiles: Vec<TileFactors>,
    pub max_proj: Vec<u8>,  // (h, w)
    pub block_max: Vec<u8>, // (edges.len() - 1, h, w)
    pub edges: Vec<usize>,
    tile_of: Vec<u32>,
    loc_of: Vec<u32>,
}

impl Factors {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        h: usize,
        w: usize,
        t: usize,
        tile: usize,
        tiles: Vec<TileFactors>,
        max_proj: Vec<u8>,
        block_max: Vec<u8>,
        edges: Vec<usize>,
    ) -> Result<Self, String> {
        let grid = tile_grid(h, w, tile);
        if tiles.len() != grid.len() {
            return Err(format!(
                "{} tiles of factors, the grid has {}",
                tiles.len(),
                grid.len()
            ));
        }
        for (k, (g, tf)) in grid.iter().zip(&tiles).enumerate() {
            let (p, r) = (g.pixels(), tf.rank);
            if tf.w.len() != p * r || tf.v.len() != t * r || tf.mean.len() != p {
                return Err(format!(
                    "tile {k}: factor sizes do not match {p} pixels, rank {r}"
                ));
            }
        }
        if max_proj.len() != h * w || edges.len() < 2 || block_max.len() != (edges.len() - 1) * h * w {
            return Err("max projection or block maxima do not match the frame size".into());
        }
        let mut tile_of = vec![0u32; h * w];
        let mut loc_of = vec![0u32; h * w];
        for (b, g) in grid.iter().enumerate() {
            let mut loc = 0;
            for y in g.y0..g.y1 {
                for x in g.x0..g.x1 {
                    tile_of[y * w + x] = b as u32;
                    loc_of[y * w + x] = loc;
                    loc += 1;
                }
            }
        }
        Ok(Self {
            h,
            w,
            t,
            grid,
            tiles,
            max_proj,
            block_max,
            edges,
            tile_of,
            loc_of,
        })
    }

    /// Split pass 2's flat outputs (`ref_pass2.py`'s `ref/`: `W`, `V`, `mean`, `sigma`,
    /// `energy` concatenated tile after tile, `rank` per tile) into tiles.
    #[allow(clippy::too_many_arguments)]
    pub fn split_flat(
        grid: &[Tile],
        t: usize,
        w: &[f32],
        v: &[f32],
        mean: &[f32],
        sigma: &[f32],
        energy: &[f64],
        rank: &[usize],
    ) -> Result<Vec<TileFactors>, String> {
        if rank.len() != grid.len() {
            return Err(format!("{} ranks for {} tiles", rank.len(), grid.len()));
        }
        let (mut ow, mut ov, mut op) = (0, 0, 0);
        let mut out = Vec::with_capacity(grid.len());
        for (g, &r) in grid.iter().zip(rank) {
            let p = g.pixels();
            if ow + p * r > w.len() || ov + t * r > v.len() || op + p > mean.len() {
                return Err("flat factors are shorter than the ranks say".into());
            }
            out.push(TileFactors {
                w: w[ow..ow + p * r].to_vec(),
                v: v[ov..ov + t * r].to_vec(),
                s: Vec::new(),
                mean: mean[op..op + p].to_vec(),
                sigma: sigma[op..op + p].to_vec(),
                energy: energy[op..op + p].to_vec(),
                rank: r,
                k: 0,
            });
            ow += p * r;
            ov += t * r;
            op += p;
        }
        Ok(out)
    }

    pub fn p(&self) -> usize {
        self.h * self.w
    }

    /// Tile and position inside the tile of flat pixel `px`.
    pub fn locate(&self, px: usize) -> (usize, usize) {
        (self.tile_of[px] as usize, self.loc_of[px] as usize)
    }

    /// `Factors.traces` of one pixel: `W[loc] V^T + mean[loc]` in float32, each frame's sum
    /// as sequential fused multiply-adds over the components (Accelerate's sgemm order on
    /// the window's factors), then the mean added.
    pub fn trace(&self, px: usize, out: &mut [f32]) {
        let (b, l) = self.locate(px);
        let tf = &self.tiles[b];
        let r = tf.rank;
        let w = &tf.w[l * r..(l + 1) * r];
        let m = tf.mean[l];
        for (o, v) in out.iter_mut().zip(tf.v.chunks_exact(r)) {
            let mut acc = 0f32;
            for (&a, &b) in w.iter().zip(v) {
                acc = a.mul_add(b, acc);
            }
            *o = acc + m;
        }
    }

    pub fn sigma(&self, px: usize) -> f32 {
        let (b, l) = self.locate(px);
        self.tiles[b].sigma[l]
    }

    /// `np.median(sigma_flat())` in float32 (the mean of the two middle values for an even
    /// count).
    pub fn sigma_median(&self) -> f32 {
        let mut s: Vec<f32> = self
            .tiles
            .iter()
            .flat_map(|tf| tf.sigma.iter().copied())
            .collect();
        s.sort_unstable_by(f32::total_cmp);
        let n = s.len();
        if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        }
    }
}
