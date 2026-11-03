//! Minimal `.npy` (NumPy format 1.0-3.0) reader and writer: little-endian, C order,
//! the dtypes the streaming pass exchanges with Python.

/// A decoded array: its shape and its values converted to the requested type.
pub struct Npy<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

fn header(bytes: &[u8]) -> Result<(String, Vec<usize>, usize), String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        v => return Err(format!("unsupported .npy version {v}")),
    };
    let h = std::str::from_utf8(&bytes[start..start + hlen]).map_err(|e| e.to_string())?;
    if h.contains("'fortran_order': True") {
        return Err("Fortran-order .npy not supported".into());
    }
    let descr = h
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .ok_or("no descr")?
        .to_string();
    let shape_s = h
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or("no shape")?;
    let shape = shape_s
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((descr, shape, start + hlen))
}

macro_rules! le {
    ($t:ty, $b:expr) => {
        $b.chunks_exact(std::mem::size_of::<$t>())
            .map(|c| <$t>::from_le_bytes(c.try_into().unwrap()))
    };
}

/// Read any numeric/bool `.npy` as `f64`.
pub fn read_f64(bytes: &[u8]) -> Result<Npy<f64>, String> {
    let (descr, shape, off) = header(bytes)?;
    let b = &bytes[off..];
    let data: Vec<f64> = match descr.as_str() {
        "<f8" => le!(f64, b).collect(),
        "<f4" => le!(f32, b).map(f64::from).collect(),
        "<i8" => le!(i64, b).map(|v| v as f64).collect(),
        "<i4" => le!(i32, b).map(f64::from).collect(),
        "|u1" | "|b1" => b.iter().map(|&v| f64::from(v)).collect(),
        d => return Err(format!("unsupported dtype {d}")),
    };
    check_len(&shape, data.len())?;
    Ok(Npy { shape, data })
}

/// Read a float `.npy` as `f32` (float64 values are rounded).
pub fn read_f32(bytes: &[u8]) -> Result<Npy<f32>, String> {
    let (descr, shape, off) = header(bytes)?;
    let b = &bytes[off..];
    let data: Vec<f32> = match descr.as_str() {
        "<f4" => le!(f32, b).collect(),
        "<f8" => le!(f64, b).map(|v| v as f32).collect(),
        d => return Err(format!("expected float dtype, got {d}")),
    };
    check_len(&shape, data.len())?;
    Ok(Npy { shape, data })
}

/// Read an integer `.npy` as `i64`.
pub fn read_i64(bytes: &[u8]) -> Result<Npy<i64>, String> {
    let (descr, shape, off) = header(bytes)?;
    let b = &bytes[off..];
    let data: Vec<i64> = match descr.as_str() {
        "<i8" => le!(i64, b).collect(),
        "<i4" => le!(i32, b).map(i64::from).collect(),
        d => return Err(format!("expected integer dtype, got {d}")),
    };
    check_len(&shape, data.len())?;
    Ok(Npy { shape, data })
}

fn check_len(shape: &[usize], n: usize) -> Result<(), String> {
    let want: usize = shape.iter().product();
    if want != n {
        return Err(format!("shape {shape:?} needs {want} values, file holds {n}"));
    }
    Ok(())
}

/// Element types `write` can store.
pub trait NpyElem: Copy {
    const DESCR: &'static str;
    fn put(self, out: &mut Vec<u8>);
}

macro_rules! elem {
    ($t:ty, $d:expr) => {
        impl NpyElem for $t {
            const DESCR: &'static str = $d;
            fn put(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    };
}
elem!(f64, "<f8");
elem!(f32, "<f4");
elem!(i64, "<i8");
elem!(i32, "<i4");
elem!(u8, "|u1");

impl NpyElem for bool {
    const DESCR: &'static str = "|b1";
    fn put(self, out: &mut Vec<u8>) {
        out.push(u8::from(self));
    }
}

/// Encode `data` with `shape` as `.npy` version 1.0 bytes.
pub fn write<T: NpyElem>(shape: &[usize], data: &[T]) -> Vec<u8> {
    let dims = match shape.len() {
        1 => format!("{},", shape[0]),
        _ => shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", "),
    };
    let mut h = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({}), }}",
        T::DESCR,
        dims
    );
    let total = 10 + h.len() + 1;
    h.push_str(&" ".repeat((64 - total % 64) % 64));
    h.push('\n');
    let mut out = Vec::with_capacity(10 + h.len() + data.len() * 8);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    for &v in data {
        v.put(&mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let b = write(&[2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let a = read_f64(&b).unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.data[5], 6.0);
        let b = write(&[4], &[1i64, -2, 3, 4]);
        assert_eq!(read_i64(&b).unwrap().data, vec![1, -2, 3, 4]);
    }
}
