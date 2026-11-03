//! Native check harness over the raw AVIs on disk, writing the same `.npy` files as the
//! Python reference scripts in `experiments/streaming/`:
//!
//!     minime-core --raw RAW_DIR --inputs INPUTS_DIR --out OUT_DIR [--check-frames T.npy]
//!     minime-core motion --raw RAW_DIR --params PARAMS.json --out OUT_DIR [--check-frames T.npy]
//!     minime-core pass1 --raw RAW_DIR --inputs INPUTS_DIR --out OUT_DIR [--estimate]
//!     minime-core pass2 --tiles TILE_DIR --inputs INPUTS_DIR --out OUT_DIR [--own-rng]
//!     minime-core seeds --inputs INPUTS_DIR --out OUT_DIR
//!     minime-core init --inputs INPUTS_DIR --out OUT_DIR
//!
//! The first is one streaming pass (`ref_dump.py`), the second the motion stage on one
//! window (`ref_motion.py`), the third pass 1 on one window (`ref_pass1.py`), with the
//! tile store written to `OUT_DIR/tiles/`, the fourth pass 2 on that tile store
//! (`ref_pass2.py`), replaying numpy's draws unless `--own-rng`, the fifth the seed stage
//! on Python's factors of that window (`ref_seeds.py`'s `fac_*.npy`), the sixth init on
//! those factors and Python's merged seeds (`ref_init.py`'s `init_seeds.npy`).

use minime_core::avi::{self, AviIndex, FileReader, ReadAt};
use minime_core::compress::{self, Gaussian, Normals, Pass2Params, Recorded};
use minime_core::corr::FactorCorr;
use minime_core::factors::Factors;
use minime_core::init::{self, InitParams};
use minime_core::motion::{self, Motion, MotionParams, Reg};
use minime_core::pass1::{self, Pass1, Pass1Params};
use minime_core::seeds::{self, SeedParams};
use minime_core::{finish, npy, Inputs};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// `.avi` files whose name starts with a number, in that number's order (`io.list_segments`).
fn segments(raw: &Path) -> Result<Vec<PathBuf>, String> {
    let mut segs: Vec<(u64, PathBuf)> = std::fs::read_dir(raw)
        .map_err(|e| format!("{}: {e}", raw.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("avi")))
        .filter_map(|p| {
            let name = p.file_name()?.to_str()?.to_string();
            let digits: String = name.chars().take_while(char::is_ascii_digit).collect();
            Some((digits.parse().ok()?, p))
        })
        .collect();
    segs.sort();
    Ok(segs.into_iter().map(|(_, p)| p).collect())
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn check_frames(args: &[String]) -> Result<Vec<usize>, String> {
    Ok(match arg(args, "--check-frames") {
        Some(p) => npy::read_i64(&std::fs::read(&p).map_err(|e| e.to_string())?)?
            .data
            .iter()
            .map(|&v| v as usize)
            .collect(),
        None => Vec::new(),
    })
}

/// Random access to the frames of all segments by session frame index.
struct Frames {
    segs: Vec<(FileReader, AviIndex)>,
    starts: Vec<usize>, // first session frame of each segment, plus the total
}

impl Frames {
    fn open(raw: &Path, h: usize, w: usize) -> Result<Self, String> {
        let mut segs = Vec::new();
        let mut starts = vec![0];
        for p in segments(raw)? {
            let mut r = FileReader::open(&p)?;
            let ix = avi::index(&mut r)?;
            if (ix.height, ix.width) != (h, w) {
                return Err(format!(
                    "{}: {}x{} frames, expected {h}x{w}",
                    p.display(),
                    ix.height,
                    ix.width
                ));
            }
            starts.push(starts.last().unwrap() + ix.frames.len());
            segs.push((r, ix));
        }
        Ok(Self { segs, starts })
    }

    fn get(&mut self, t: usize) -> Result<Vec<u8>, String> {
        let s = self.starts.partition_point(|&v| v <= t) - 1;
        let (r, ix) = self.segs.get_mut(s).ok_or(format!("frame {t} is past the end"))?;
        let (off, size) = ix.frames[t - self.starts[s]];
        Ok(ix.unpad(&r.read_at(off, size as usize)?))
    }
}

/// The motion stage on frames `[t0, t1)`: template from the first `template_frames` frames
/// of segment 0, then shift and quality per frame (`ref_motion.py`).
fn run_motion(args: &[String]) -> Result<(), String> {
    let raw = PathBuf::from(arg(args, "--raw").ok_or("--raw RAW_DIR")?);
    let out = PathBuf::from(arg(args, "--out").ok_or("--out DIR")?);
    let pj = arg(args, "--params").ok_or("--params PARAMS.json")?;
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&pj).map_err(|e| format!("{pj}: {e}"))?)
            .map_err(|e| e.to_string())?;
    let p: MotionParams = serde_json::from_value(meta.clone()).map_err(|e| format!("{pj}: {e}"))?;
    let t0 = meta["t0"].as_u64().ok_or("params: t0")? as usize;
    let t1 = meta["t1"].as_u64().ok_or("params: t1")? as usize;
    let check = check_frames(args)?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut frames = Frames::open(&raw, p.h, p.w)?;
    if t1 > *frames.starts.last().unwrap() {
        return Err(format!("t1 {t1} is past the last frame"));
    }

    let s = Instant::now();
    let n_tmpl = p.template_frames.min(frames.starts[1]);
    let m = Motion::build(&p, n_tmpl, &mut |t| {
        Ok(motion::registration_image(&frames.get(t)?, &p))
    })?;
    let t_template = s.elapsed().as_secs_f64();

    let (mut t_read, mut t_reg, mut t_est, mut t_qual) = (0.0, 0.0, 0.0, 0.0);
    let (mut shifts, mut quality) = (Vec::new(), Vec::new());
    let (mut reg_chk, mut prep_chk, mut chk_t) = (Vec::new(), Vec::new(), Vec::new());
    let corr = motion::Correlator::new(p.h, p.w, p.motion_down)?;
    for t in t0..t1 {
        let s0 = Instant::now();
        let raw = frames.get(t)?;
        let s1 = Instant::now();
        let reg = motion::registration_image(&raw, &p);
        let s2 = Instant::now();
        let [dy, dx] = m.estimate(&reg);
        let s3 = Instant::now();
        quality.push(m.quality(&reg.shifted(p.h, p.w, dy as f32, dx as f32)));
        let s4 = Instant::now();
        shifts.extend([dy, dx]);
        if check.contains(&t) {
            chk_t.push(t as i64);
            if let Reg::F32(f) = &reg {
                reg_chk.extend_from_slice(f);
            }
            prep_chk.extend(corr.prep(&reg));
        }
        t_read += (s1 - s0).as_secs_f64();
        t_reg += (s2 - s1).as_secs_f64();
        t_est += (s3 - s2).as_secs_f64();
        t_qual += (s4 - s3).as_secs_f64();
    }
    let (ph, pw) = m.prep_shape();
    let n = t1 - t0;
    let mut files = vec![
        (
            "template.npy".to_string(),
            npy::write(&[ph, pw], m.template_image()),
        ),
        ("shifts.npy".into(), npy::write(&[n, 2], &shifts)),
        ("quality.npy".into(), npy::write(&[n], &quality)),
        ("frames_t.npy".into(), npy::write(&[chk_t.len()], &chk_t)),
        ("prep.npy".into(), npy::write(&[chk_t.len(), ph, pw], &prep_chk)),
    ];
    if !reg_chk.is_empty() {
        files.push(("reg.npy".into(), npy::write(&[chk_t.len(), p.h, p.w], &reg_chk)));
    }
    let timing = serde_json::json!({
        "frames": n, "template": t_template, "read": t_read, "registration": t_reg,
        "estimate": t_est, "quality": t_qual,
    });
    files.push(("timing.json".into(), serde_json::to_vec_pretty(&timing).unwrap()));
    for (name, bytes) in &files {
        std::fs::write(out.join(name), bytes).map_err(|e| e.to_string())?;
    }
    eprintln!("{timing}");
    Ok(())
}

/// Pass 1 on frames `[t0, t1)` with the inputs' shifts (or, with `--estimate`, shifts
/// estimated against the inputs' template), writing the tile store as `ref_pass1.py` does.
fn run_pass1(args: &[String]) -> Result<(), String> {
    let raw = PathBuf::from(arg(args, "--raw").ok_or("--raw RAW_DIR")?);
    let inputs = PathBuf::from(arg(args, "--inputs").ok_or("--inputs DIR")?);
    let out = PathBuf::from(arg(args, "--out").ok_or("--out DIR")?);
    let estimate = args.iter().any(|a| a == "--estimate");
    let read = |name: &str| std::fs::read(inputs.join(name)).map_err(|e| format!("{name}: {e}"));
    let p: Pass1Params =
        serde_json::from_slice(&read("params.json")?).map_err(|e| format!("params.json: {e}"))?;
    let shifts = npy::read_f32(&read("shifts.npy")?)?.data;
    let template = npy::read_f32(&read("template.npy")?)?.data;
    let (h, w, n) = (p.motion.h, p.motion.w, p.t1 - p.t0);
    if !estimate && shifts.len() != 2 * n {
        return Err(format!("{} shifts for {n} frames", shifts.len() / 2));
    }
    let tiles = out.join("tiles");
    let _ = std::fs::remove_dir_all(&tiles);
    std::fs::create_dir_all(&tiles).map_err(|e| e.to_string())?;
    let mut frames = Frames::open(&raw, h, w)?;
    let m = Motion::from_template_image(&p.motion, template)?;
    let mut p1 = Pass1::new(&p, Some(m));
    let mut files: Vec<std::fs::File> = (0..p1.grid.len())
        .map(|k| std::fs::File::create(tiles.join(format!("tile_{k:03}.part00"))))
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let (mut t_read, mut t_push, mut t_write) = (0.0, 0.0, 0.0);
    let mut write = |p1: &mut Pass1, t_write: &mut f64| -> Result<(), String> {
        let s = Instant::now();
        for (f, buf) in files.iter_mut().zip(p1.take_tiles()) {
            f.write_all(&buf).map_err(|e| e.to_string())?;
        }
        *t_write += s.elapsed().as_secs_f64();
        Ok(())
    };
    for t in p.t0..p.t1 {
        let s0 = Instant::now();
        let fr = frames.get(t)?;
        let s1 = Instant::now();
        let i = t - p.t0;
        p1.push(&fr, (!estimate).then(|| [shifts[2 * i], shifts[2 * i + 1]]))?;
        t_read += (s1 - s0).as_secs_f64();
        t_push += s1.elapsed().as_secs_f64();
        if p1.buffered() == 100 {
            write(&mut p1, &mut t_write)?;
        }
    }
    write(&mut p1, &mut t_write)?;
    let crc: Vec<u32> = (0..p1.grid.len())
        .map(|k| std::fs::read(tiles.join(format!("tile_{k:03}.part00"))).map(|b| pass1::crc32(&b, 0)))
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let summary = serde_json::json!({
        "frames": p1.t, "blocks": p1.n_blocks(), "tiles": p1.grid.len(), "estimate": estimate,
        "read": t_read, "process": t_push, "write_tiles": t_write,
    });
    eprintln!("{summary}");
    for (name, bytes) in pass1::outputs(&p1, &crc, summary) {
        std::fs::write(out.join(name), bytes).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Pass 2 on the tile store `--tiles` (`tile_NNN.partPP`, parts in order), one tile at a
/// time, writing the files `ref_pass2.py` writes under `ref/`.
fn run_pass2(args: &[String]) -> Result<(), String> {
    let tiles = PathBuf::from(arg(args, "--tiles").ok_or("--tiles DIR")?);
    let inputs = PathBuf::from(arg(args, "--inputs").ok_or("--inputs DIR")?);
    let out = PathBuf::from(arg(args, "--out").ok_or("--out DIR")?);
    let own_rng = args.iter().any(|a| a == "--own-rng");
    let read = |name: &str| std::fs::read(inputs.join(name)).map_err(|e| format!("{name}: {e}"));
    let p: Pass2Params =
        serde_json::from_slice(&read("pass2.json")?).map_err(|e| format!("pass2.json: {e}"))?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let grid = pass1::tile_grid(p.h, p.w, p.tile);
    let (mut t_read, mut t_compress) = (0.0, 0.0);
    let mut res = Vec::with_capacity(grid.len());
    for (k, tile) in grid.iter().enumerate() {
        let s0 = Instant::now();
        let mut raw = Vec::new();
        for part in 0.. {
            match std::fs::read(tiles.join(format!("tile_{k:03}.part{part:02}"))) {
                Ok(b) => raw.extend_from_slice(&b),
                Err(_) if part > 0 => break,
                Err(e) => return Err(format!("tile {k}: {e}")),
            }
        }
        if raw.len() != p.t * tile.pixels() {
            return Err(format!(
                "tile {k}: {} bytes, expected {}",
                raw.len(),
                p.t * tile.pixels()
            ));
        }
        let mut rng: Box<dyn Normals> = if own_rng {
            Box::new(Gaussian::new(k as u64))
        } else {
            Box::new(Recorded::new(
                npy::read_f32(&read(&format!("omega_{k:03}.npy"))?)?.data,
            ))
        };
        let s1 = Instant::now();
        res.push(compress::compress_tile(
            &raw,
            p.t,
            tile.pixels(),
            &p,
            rng.as_mut(),
        )?);
        t_read += (s1 - s0).as_secs_f64();
        t_compress += s1.elapsed().as_secs_f64();
    }
    let summary = serde_json::json!({
        "frames": p.t, "tiles": grid.len(), "own_rng": own_rng,
        "rank_total": res.iter().map(|r| r.rank).sum::<usize>(),
        "read": t_read, "compress": t_compress,
    });
    eprintln!("{summary}");
    for (name, bytes) in compress::outputs(&res, summary) {
        std::fs::write(out.join(name), bytes).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The window's `Factors` from `ref_seeds.py`'s flat `fac_*.npy` files.
fn load_factors(inputs: &Path, h: usize, w: usize, t: usize, tile: usize) -> Result<Factors, String> {
    let read = |name: &str| std::fs::read(inputs.join(name)).map_err(|e| format!("{name}: {e}"));
    let f32s =
        |n: &str| -> Result<Vec<f32>, String> { Ok(npy::read_f32(&read(&format!("fac_{n}.npy"))?)?.data) };
    let ints = |n: &str| -> Result<Vec<usize>, String> {
        Ok(npy::read_i64(&read(&format!("fac_{n}.npy"))?)?
            .data
            .iter()
            .map(|&v| v as usize)
            .collect())
    };
    let grid = pass1::tile_grid(h, w, tile);
    let tiles = Factors::split_flat(
        &grid,
        t,
        &f32s("W")?,
        &f32s("V")?,
        &f32s("mean")?,
        &f32s("sigma")?,
        &npy::read_f64(&read("fac_energy.npy")?)?.data,
        &ints("rank")?,
    )?;
    Factors::new(
        h,
        w,
        t,
        tile,
        tiles,
        npy::read_u8(&read("fac_max_proj.npy")?)?.data,
        npy::read_u8(&read("fac_block_max.npy")?)?.data,
        ints("block_edges")?,
    )
}

fn write_outputs(out: &Path, files: &[(String, Vec<u8>)]) -> Result<(), String> {
    for (name, bytes) in files {
        if name == "summary.json" {
            eprintln!("{}", String::from_utf8_lossy(bytes));
        }
        std::fs::write(out.join(name), bytes).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn run_seeds(args: &[String]) -> Result<(), String> {
    let inputs = PathBuf::from(arg(args, "--inputs").ok_or("--inputs DIR")?);
    let out = PathBuf::from(arg(args, "--out").ok_or("--out DIR")?);
    let p: SeedParams =
        serde_json::from_slice(&std::fs::read(inputs.join("seeds.json")).map_err(|e| e.to_string())?)
            .map_err(|e| format!("seeds.json: {e}"))?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let s0 = Instant::now();
    let f = load_factors(&inputs, p.h, p.w, p.t, p.tile)?;
    let load = s0.elapsed().as_secs_f64();
    let s1 = Instant::now();
    let res = seeds::run(&f, &p, &|| s1.elapsed().as_secs_f64())?;
    let seconds = s1.elapsed().as_secs_f64();
    let summary = serde_json::json!({"frames": p.t, "load": load, "seconds": seconds});
    write_outputs(&out, &seeds::outputs(&res, summary))
}

fn run_init(args: &[String]) -> Result<(), String> {
    let inputs = PathBuf::from(arg(args, "--inputs").ok_or("--inputs DIR")?);
    let out = PathBuf::from(arg(args, "--out").ok_or("--out DIR")?);
    let read = |name: &str| std::fs::read(inputs.join(name)).map_err(|e| format!("{name}: {e}"));
    let p: InitParams = serde_json::from_slice(&read("init.json")?).map_err(|e| format!("init.json: {e}"))?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let s0 = Instant::now();
    let f = load_factors(&inputs, p.h, p.w, p.t, p.tile)?;
    let seeds: Vec<[i64; 2]> = npy::read_i64(&read("init_seeds.npy")?)?
        .data
        .as_chunks::<2>()
        .0
        .to_vec();
    let load = s0.elapsed().as_secs_f64();
    let s1 = Instant::now();
    let mut fc = FactorCorr::new(&f, p.noise_freq);
    let corr_setup = s1.elapsed().as_secs_f64();
    let s2 = Instant::now();
    let res = init::run(&mut fc, &f, &seeds, &p, &|| s2.elapsed().as_secs_f64())?;
    let seconds = s2.elapsed().as_secs_f64();
    let summary =
        serde_json::json!({"frames": p.t, "load": load, "corr_setup": corr_setup, "seconds": seconds});
    write_outputs(&out, &init::outputs(&res, &p, summary))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("motion") => return run_motion(&args),
        Some("pass1") => return run_pass1(&args),
        Some("pass2") => return run_pass2(&args),
        Some("seeds") => return run_seeds(&args),
        Some("init") => return run_init(&args),
        _ => {}
    }
    let raw = PathBuf::from(arg(&args, "--raw").ok_or("--raw RAW_DIR")?);
    let inputs = PathBuf::from(arg(&args, "--inputs").ok_or("--inputs DIR")?);
    let out = PathBuf::from(arg(&args, "--out").ok_or("--out DIR")?);
    let check = check_frames(&args)?;
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let inp = Inputs::load(&mut |name| std::fs::read(inputs.join(name)).map_err(|e| format!("{name}: {e}")))?;
    let (h, w, t_total) = (inp.params.h, inp.params.w, inp.params.t);
    let prep = inp.prep();
    let t0 = Instant::now();
    let mut pass = minime_core::stream::StreamPass::new(inp.params.clone(), &inp.a, &inp.ymean);
    let setup = t0.elapsed().as_secs_f64();
    let (mut t_read, mut t_prep, mut t_push) = (0.0, 0.0, 0.0);
    let mut checked: Vec<u8> = Vec::new();
    let mut t = 0usize;
    'outer: for seg in segments(&raw)? {
        let mut r = FileReader::open(&seg)?;
        let ix = avi::index(&mut r)?;
        if (ix.height, ix.width) != (h, w) {
            return Err(format!(
                "{}: {}x{} frames, expected {h}x{w}",
                seg.display(),
                ix.height,
                ix.width
            ));
        }
        for &(off, size) in &ix.frames {
            if t == t_total {
                break 'outer;
            }
            let s = Instant::now();
            let frame = ix.unpad(&r.read_at(off, size as usize)?);
            let s2 = Instant::now();
            let y8 = prep.frame(&frame, h, w, inp.shifts[2 * t], inp.shifts[2 * t + 1]);
            let y: Vec<f32> = y8.iter().map(|&v| f32::from(v)).collect();
            let s3 = Instant::now();
            pass.push(&y);
            t_push += s3.elapsed().as_secs_f64();
            t_prep += (s3 - s2).as_secs_f64();
            t_read += (s2 - s).as_secs_f64();
            if check.contains(&t) {
                checked.extend_from_slice(&y8);
            }
            t += 1;
        }
    }
    if t != t_total {
        return Err(format!("read {t} frames, params say {t_total}"));
    }
    let s = Instant::now();
    let timing = serde_json::json!({"setup": setup, "read": t_read, "prep": t_prep, "push": t_push});
    let mut files = finish(&pass, &inp, timing)?;
    let update = s.elapsed().as_secs_f64();
    if !check.is_empty() {
        let n = checked.len() / (h * w);
        files.push(("frames.npy".into(), npy::write(&[n, h, w], &checked)));
    }
    for (name, bytes) in &files {
        std::fs::write(out.join(name), bytes).map_err(|e| e.to_string())?;
    }
    eprintln!(
        "{t} frames: setup {setup:.1} s, read {t_read:.1} s, prep {t_prep:.1} s, traces+sums {t_push:.1} s, update {update:.1} s"
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
