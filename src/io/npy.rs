use super::array::{Array, DType};
use crate::error::{Error, Result};

const MAGIC: &[u8] = b"\x93NUMPY";

fn descriptor(text: &str) -> Result<DType> {
    let normalized = text.trim_matches(|c| c == '\'' || c == '"');
    let kind = normalized.trim_start_matches(['<', '=', '|', '>']);
    if normalized.starts_with('>') && kind != "b1" && kind != "i1" && kind != "u1" {
        return Err(Error::Format(format!("big endian npy descriptor {normalized}")));
    }
    Ok(match kind {
        "b1" => DType::Bool,
        "i1" => DType::I8,
        "i2" => DType::I16,
        "i4" => DType::I32,
        "i8" => DType::I64,
        "u1" => DType::U8,
        "u2" => DType::U16,
        "u4" => DType::U32,
        "u8" => DType::U64,
        "f2" => DType::F16,
        "f4" => DType::F32,
        "f8" => DType::F64,
        other => return Err(Error::Format(format!("unsupported npy descriptor {other}"))),
    })
}

fn field<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let start = header
        .find(key)
        .ok_or_else(|| Error::Format(format!("npy header has no {key}")))?
        + key.len();
    let rest = header[start..].trim_start().trim_start_matches(':').trim_start();
    Ok(rest)
}

pub fn parse(bytes: &[u8]) -> Result<Array> {
    if !bytes.starts_with(MAGIC) {
        return Err(Error::Format("not a npy stream".into()));
    }
    let major = bytes[6];
    let (header_length, start) = if major == 1 {
        (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[start..start + header_length])
        .map_err(|_| Error::Format("npy header is not utf8".into()))?;

    let dtype = {
        let rest = field(header, "'descr'")?;
        let end = rest[1..]
            .find(rest.as_bytes()[0] as char)
            .ok_or_else(|| Error::Format("npy descriptor is unterminated".into()))?;
        descriptor(&rest[..end + 2])?
    };
    if field(header, "'fortran_order'")?.starts_with("True") {
        return Err(Error::Format("fortran ordered npy arrays are unsupported".into()));
    }
    let shape = {
        let rest = field(header, "'shape'")?;
        let end = rest
            .find(')')
            .ok_or_else(|| Error::Format("npy shape is unterminated".into()))?;
        rest[1..end]
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(|part| {
                part.parse::<usize>()
                    .map_err(|_| Error::Format(format!("npy shape entry {part}")))
            })
            .collect::<Result<Vec<_>>>()?
    };

    let count: usize = shape.iter().product();
    let data = start + header_length;
    let wanted = count * dtype.width();
    if bytes.len() < data + wanted {
        return Err(Error::Format("npy payload is truncated".into()));
    }
    Ok(Array {
        shape,
        dtype,
        bytes: bytes[data..data + wanted].to_vec(),
    })
}
