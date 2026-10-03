use crate::error::{Error, Result};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
}

impl DType {
    pub fn width(self) -> usize {
        match self {
            DType::Bool | DType::I8 | DType::U8 => 1,
            DType::I16 | DType::U16 | DType::F16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Array {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub bytes: Vec<u8>,
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        let shift = mantissa.leading_zeros() - 21;
        let exponent = 127 - 15 - shift;
        let mantissa = (mantissa << (shift + 1)) & 0x3ff;
        return f32::from_bits(sign | (exponent << 23) | (mantissa << 13));
    }
    if exponent == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }
    f32::from_bits(sign | ((exponent + 127 - 15) << 23) | (mantissa << 13))
}

macro_rules! decode {
    ($bytes:expr, $kind:ty, $width:literal) => {
        $bytes
            .chunks_exact($width)
            .map(|chunk| <$kind>::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<$kind>>()
    };
}

impl Array {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn mismatch(&self, wanted: &str) -> Error {
        Error::Format(format!("expected {wanted} array, found {:?}", self.dtype))
    }

    pub fn to_f32(&self) -> Result<Vec<f32>> {
        match self.dtype {
            DType::F32 => Ok(decode!(self.bytes, f32, 4)),
            DType::F16 => Ok(self
                .bytes
                .chunks_exact(2)
                .map(|chunk| half_to_f32(u16::from_le_bytes(chunk.try_into().unwrap())))
                .collect()),
            _ => Err(self.mismatch("float32")),
        }
    }

    pub fn to_f64(&self) -> Result<Vec<f64>> {
        match self.dtype {
            DType::F64 => Ok(decode!(self.bytes, f64, 8)),
            DType::F32 => Ok(self.to_f32()?.into_iter().map(f64::from).collect()),
            _ => Err(self.mismatch("float64")),
        }
    }

    pub fn to_i64(&self) -> Result<Vec<i64>> {
        match self.dtype {
            DType::I64 => Ok(decode!(self.bytes, i64, 8)),
            DType::I32 => Ok(decode!(self.bytes, i32, 4)
                .into_iter()
                .map(i64::from)
                .collect()),
            DType::I16 => Ok(decode!(self.bytes, i16, 2)
                .into_iter()
                .map(i64::from)
                .collect()),
            DType::I8 => Ok(self.bytes.iter().map(|&b| i64::from(b as i8)).collect()),
            DType::U8 | DType::Bool => Ok(self.bytes.iter().map(|&b| i64::from(b)).collect()),
            _ => Err(self.mismatch("integer")),
        }
    }

    pub fn to_i16(&self) -> Result<Vec<i16>> {
        match self.dtype {
            DType::I16 => Ok(decode!(self.bytes, i16, 2)),
            _ => Err(self.mismatch("int16")),
        }
    }

    pub fn to_bool(&self) -> Result<Vec<bool>> {
        match self.dtype {
            DType::Bool | DType::U8 => Ok(self.bytes.iter().map(|&b| b != 0).collect()),
            _ => Err(self.mismatch("bool")),
        }
    }
}
