//! Writes `N` raw frames (600 x 600 u8, spread over the first AVI of `RAW_DIR`) back to back:
//!
//!     cargo run --release --example dump_frames -- RAW_DIR OUT.u8 [N]
//!
//! The wasm micro-benchmark (`tools/wasm-bench/`) feeds them to the browser.

use minime_core::avi::{self, FileReader, ReadAt};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let mut avis: Vec<_> = std::fs::read_dir(&args[1])
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "avi"))
        .collect();
    avis.sort();
    let mut r = FileReader::open(&avis[0]).unwrap();
    let ix = avi::index(&mut r).unwrap();
    let mut out = Vec::new();
    for i in 0..n {
        let (off, size) = ix.frames[i * 113 % ix.frames.len()];
        out.extend(ix.unpad(&r.read_at(off, size as usize).unwrap()));
    }
    std::fs::write(&args[2], out).unwrap();
    println!("{n} frames of {} x {}", ix.height, ix.width);
}
