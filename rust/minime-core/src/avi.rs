//! Uncompressed 8-bit (GREY / Y800) AVI frame index over random-access reads.
//!
//! Only the headers and the `idx1` index are read; each frame is then one read of
//! `width * height` bytes at its offset (`File.slice` in the browser). Without an `idx1`
//! (or for OpenDML `AVIX` extensions) the `movi` lists are scanned chunk by chunk.

/// Random access to the bytes of one file.
pub trait ReadAt {
    fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>, String>;
    fn size(&self) -> u64;
}

#[derive(Debug, Clone)]
pub struct AviIndex {
    pub width: usize,
    pub height: usize,
    /// (offset of the frame's bytes, chunk size) per frame, in display order
    pub frames: Vec<(u64, u32)>,
    /// bytes per stored row (`width`, or `width` rounded up to 4 for padded DIB rows)
    pub stride: usize,
}

impl AviIndex {
    /// The frame's `height * width` pixels from its chunk bytes (row padding removed).
    pub fn unpad(&self, chunk: &[u8]) -> Vec<u8> {
        if self.stride == self.width {
            return chunk[..self.width * self.height].to_vec();
        }
        let mut out = Vec::with_capacity(self.width * self.height);
        for r in 0..self.height {
            out.extend_from_slice(&chunk[r * self.stride..r * self.stride + self.width]);
        }
        out
    }
}

fn u32le(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

fn is_video_chunk(id: &[u8]) -> bool {
    id.len() == 4
        && id[0].is_ascii_digit()
        && id[1].is_ascii_digit()
        && (&id[2..] == b"dc" || &id[2..] == b"db")
}

struct Header {
    width: usize,
    height: usize,
}

fn parse_hdrl(b: &[u8]) -> Result<Header, String> {
    // walk sub-chunks of the hdrl list looking for strh(vids) + strf (BITMAPINFOHEADER)
    let mut i = 0;
    let mut vids = false;
    let mut found: Option<Header> = None;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let size = u32le(b, i + 4) as usize;
        let body = i + 8;
        if id == b"LIST" {
            i = body + 4; // descend into strl
            continue;
        }
        if id == b"strh" && body + 8 <= b.len() {
            vids = &b[body..body + 4] == b"vids";
        }
        if id == b"strf" && vids && found.is_none() && body + 20 <= b.len() {
            let w = u32le(b, body + 4) as i32;
            let h = u32le(b, body + 8) as i32;
            let bits = u16::from_le_bytes([b[body + 14], b[body + 15]]);
            let comp = &b[body + 16..body + 20];
            let grey = comp == b"GREY" || comp == b"Y800" || comp == b"Y8  ";
            if !grey || bits != 8 {
                return Err(format!(
                    "unsupported video format {:?} ({bits} bit): only uncompressed 8-bit GREY AVIs",
                    String::from_utf8_lossy(comp)
                ));
            }
            found = Some(Header {
                width: w.unsigned_abs() as usize,
                height: h.unsigned_abs() as usize,
            });
        }
        i = body + size + (size & 1);
    }
    found.ok_or_else(|| "no video stream header (strf) in hdrl".into())
}

/// Build the frame index of an AVI.
pub fn index(r: &mut dyn ReadAt) -> Result<AviIndex, String> {
    let size = r.size();
    let head = r.read_at(0, 12)?;
    if &head[..4] != b"RIFF" || &head[8..12] != b"AVI " {
        return Err("not a RIFF AVI file".into());
    }
    let mut hdr: Option<Header> = None;
    let mut movi: Vec<(u64, u64)> = Vec::new(); // (offset of 'movi' fourcc, end)
    let mut idx1: Option<Vec<u8>> = None;
    // top-level RIFF chunks: 'RIFF AVI ' then optional 'RIFF AVIX' (OpenDML)
    let mut riff = 0u64;
    while riff + 12 <= size {
        let rh = r.read_at(riff, 12)?;
        if &rh[..4] != b"RIFF" {
            break;
        }
        let rsize = u64::from(u32le(&rh, 4));
        let rend = (riff + 8 + rsize).min(size);
        let mut i = riff + 12;
        while i + 8 <= rend {
            let ch = r.read_at(i, 12.min((rend - i) as usize))?;
            let id = &ch[..4];
            let csize = u64::from(u32le(&ch, 4));
            if id == b"LIST" && ch.len() >= 12 {
                let kind = &ch[8..12];
                if kind == b"hdrl" && hdr.is_none() {
                    hdr = Some(parse_hdrl(&r.read_at(i + 12, (csize - 4) as usize)?)?);
                } else if kind == b"movi" {
                    movi.push((i + 8, (i + 8 + csize).min(rend)));
                }
            } else if id == b"idx1" && idx1.is_none() {
                idx1 = Some(r.read_at(i + 8, csize as usize)?);
            }
            i += 8 + csize + (csize & 1);
        }
        riff = rend + (rend & 1);
    }
    let hdr = hdr.ok_or("no hdrl header list")?;
    let first = *movi.first().ok_or("no movi list")?;
    let mut frames = Vec::new();
    if let (Some(ix), 1) = (idx1.as_ref(), movi.len()) {
        let entries: Vec<(&[u8], u64, u32)> = ix
            .as_chunks::<16>()
            .0
            .iter()
            .map(|e| (&e[..4], u64::from(u32le(e, 8)), u32le(e, 12)))
            .filter(|(id, _, _)| is_video_chunk(id))
            .collect();
        if let Some(&(_, off0, _)) = entries.first() {
            // offsets are relative to the 'movi' fourcc (usual) or absolute
            let rel = r.read_at(first.0 + off0, 4)?;
            let base = if is_video_chunk(&rel) { first.0 } else { 0 };
            frames = entries.iter().map(|&(_, off, sz)| (base + off + 8, sz)).collect();
        }
    } else {
        for &(start, end) in &movi {
            let mut i = start + 4;
            while i + 8 <= end {
                let ch = r.read_at(i, 12.min((end - i) as usize))?;
                let csize = u32le(&ch, 4);
                if &ch[..4] == b"LIST" {
                    i += 12; // 'rec ' list: descend
                    continue;
                }
                if is_video_chunk(&ch[..4]) {
                    frames.push((i + 8, csize));
                }
                i += 8 + u64::from(csize) + u64::from(csize & 1);
            }
        }
    }
    // dropped frames are stored as empty chunks; the frame count follows the decoder,
    // which skips them
    frames.retain(|&(_, sz)| sz > 0);
    let n = hdr.width * hdr.height;
    let padded = ((hdr.width + 3) & !3) * hdr.height;
    let stride = match frames.first() {
        Some(&(_, sz)) if sz as usize == n => hdr.width,
        Some(&(_, sz)) if sz as usize == padded => (hdr.width + 3) & !3,
        Some(&(_, sz)) => {
            return Err(format!(
                "frame chunk of {sz} bytes for {}x{} GREY",
                hdr.width, hdr.height
            ))
        }
        None => hdr.width,
    };
    Ok(AviIndex {
        width: hdr.width,
        height: hdr.height,
        frames,
        stride,
    })
}

/// `ReadAt` over a file on disk (native builds).
#[cfg(not(target_arch = "wasm32"))]
pub struct FileReader {
    file: std::fs::File,
    len: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl FileReader {
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = file.metadata().map_err(|e| e.to_string())?.len();
        Ok(Self { file, len })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ReadAt for FileReader {
    fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>, String> {
        use std::io::{Read, Seek, SeekFrom};
        let mut buf = vec![0u8; len];
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|e| e.to_string())?;
        self.file.read_exact(&mut buf).map_err(|e| e.to_string())?;
        Ok(buf)
    }
    fn size(&self) -> u64 {
        self.len
    }
}
