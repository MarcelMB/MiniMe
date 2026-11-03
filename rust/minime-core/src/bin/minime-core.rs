//! Native check harness: one streaming pass over the raw AVIs on disk, writing the same
//! `.npy` files as `experiments/streaming/ref_dump.py`'s `ref/`.
//!
//!     minime-core --raw RAW_DIR --inputs INPUTS_DIR --out OUT_DIR [--check-frames T.npy]

use minime_core::avi::{self, FileReader, ReadAt};
use minime_core::{finish, npy, Inputs};
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

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let raw = PathBuf::from(arg(&args, "--raw").ok_or("--raw RAW_DIR")?);
    let inputs = PathBuf::from(arg(&args, "--inputs").ok_or("--inputs DIR")?);
    let out = PathBuf::from(arg(&args, "--out").ok_or("--out DIR")?);
    let check: Vec<usize> = match arg(&args, "--check-frames") {
        Some(p) => npy::read_i64(&std::fs::read(&p).map_err(|e| e.to_string())?)?
            .data
            .iter()
            .map(|&v| v as usize)
            .collect(),
        None => Vec::new(),
    };
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
