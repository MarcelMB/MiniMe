//! A few real 64 x 64 pass-2 tiles of the first window (3000 frames) of the 5-min synth: median,
//! the session's shifts, tophat, then the tile cut out as `(T, 4096)` uint8, for the GPU GEMM checks.
//! cargo run --release --example make_tiles -- <raw dir> <shifts_local.npy> <out dir>
use minime_core::avi::{index, FileReader, ReadAt};
use minime_core::npy;
use minime_core::preprocess::Prep;
use std::io::Write;

const T: usize = 3000;
const IDS: [usize; 8] = [11, 25, 34, 44, 55, 66, 77, 88];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (raw, out) = (std::path::Path::new(&a[1]), std::path::Path::new(&a[3]));
    let shifts = npy::read_f32(&std::fs::read(&a[2]).unwrap()).unwrap().data;
    std::fs::create_dir_all(out).unwrap();
    let (h, w) = (600usize, 600usize);
    let prep = Prep {
        median_ksize: 7,
        tophat_wnd: 15,
        motion_before_tophat: true,
    };
    let threads = 8;
    let per = T.div_ceil(threads);
    // each thread handles a contiguous range of frames and returns the tile cut-outs
    let results: Vec<Vec<Vec<u8>>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|k| {
                let (shifts, prep) = (&shifts, &prep);
                s.spawn(move || {
                    let mut tiles: Vec<Vec<u8>> = vec![Vec::new(); IDS.len()];
                    let mut files: Vec<Option<(FileReader, minime_core::avi::AviIndex)>> =
                        (0..3).map(|_| None).collect();
                    for t in (k * per)..((k + 1) * per).min(T) {
                        let (f, i) = (t / 1000, t % 1000);
                        if files[f].is_none() {
                            let mut rd = FileReader::open(&raw.join(format!("{f}.avi"))).unwrap();
                            let idx = index(&mut rd).unwrap();
                            files[f] = Some((rd, idx));
                        }
                        let (rd, idx) = files[f].as_mut().unwrap();
                        let (off, sz) = idx.frames[i];
                        let frame = idx.unpad(&rd.read_at(off, sz as usize).unwrap());
                        let o = prep.frame(&frame, h, w, shifts[2 * t], shifts[2 * t + 1]);
                        for (n, &id) in IDS.iter().enumerate() {
                            let (ty, tx) = (id / 10, id % 10);
                            for y in 0..64 {
                                let s0 = (ty * 64 + y) * w + tx * 64;
                                tiles[n].extend_from_slice(&o[s0..s0 + 64]);
                            }
                        }
                    }
                    tiles
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (n, &id) in IDS.iter().enumerate() {
        let mut f = std::fs::File::create(out.join(format!("tile_{id}.u8"))).unwrap();
        for r in &results {
            f.write_all(&r[n]).unwrap();
        }
    }
    println!("wrote {} tiles of {T} x 4096 to {}", IDS.len(), out.display());
}
