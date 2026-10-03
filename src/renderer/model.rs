use crate::error::{Error, Result};
use std::path::Path;

pub const HIDDEN: usize = 80;
pub const FEATURES: usize = 20;
pub const RADIUS: i32 = 5;
pub const SIDE: usize = 11;
pub const GRID: usize = 121;
pub const GATES: usize = 240;
pub const PREFIX_WINDOW: usize = 256;
pub const WARM_WINDOW: usize = 128;
pub const RECENT_WINDOW: usize = 60;

pub const ARTIFACT_BYTES: usize = 44_484;
pub const FIXED_BYTES: usize = 39_512;
pub const ADAPTER_BYTES: usize = 4_972;
pub const ADAPTER_INPUT: usize = 145;
pub const ADAPTER_RANK: usize = 16;
pub const HEADER_BYTES: usize = 256;
pub const ADAPTER_HEADER_BYTES: usize = 24;

pub const ARTIFACT_SHA256: &str =
    "405c34bceb55485dfd6bd3c0368bce079feea680a64c6b261904ef5b4713e240";
pub const EXPECTED_BODY_CRC32: u32 = 0x6be2_896c;
pub const EXPECTED_ADAPTER_CRC32: u32 = 0x6aa9_101c;

pub const CFG_EMIT_BIAS_Q14: usize = 0;
pub const CFG_INV_EMIT_Q16: usize = 1;
pub const CFG_INV_MAG_Q16: usize = 2;
pub const CFG_INV_DIR_Q16: usize = 3;
pub const CFG_HYSTERESIS_Q16: usize = 4;
pub const CFG_FORCE_RELEASE_Q16: usize = 5;
pub const CFG_MAX_ABS_COUNT: usize = 6;
pub const CFG_ZERO_INTENT_Q16: usize = 7;
pub const CFG_ZERO_DEBT_Q16: usize = 8;
pub const CFG_AF_Q16: usize = 9;
pub const CFG_LATERAL_FREE_Q15: usize = 10;

pub const FEATURE_Q: u32 = 8;
pub const GATE_Q: u32 = 14;
pub const SMOOTH_Q: u32 = 16;
pub const PROB_Q: u32 = 15;
pub const EXP_Q: i32 = 20;
pub const LOG_Q: u32 = 14;
pub const TANGENT_Q: u32 = 15;
pub const LUT_INTERVALS: usize = 256;
pub const LUT_POINTS: usize = 257;
pub const LUT_DOMAIN_Q14: i32 = 8 << GATE_Q;
pub const EXP_DOMAIN_Q14: i32 = 16 << GATE_Q;

const MAGIC: &[u8; 8] = b"ABCFIX1\0";
const ADAPTER_MAGIC: &[u8; 8] = b"OHV1R16\0";

const EXPECTED_SOURCE_SHA256: [u8; 32] = [
    0xe9, 0x95, 0x1a, 0x9b, 0xc2, 0x5b, 0x69, 0xbf, 0x65, 0x2c, 0xeb, 0xbc, 0xa8, 0x74, 0x9b, 0xad,
    0xd0, 0x2b, 0x8c, 0x06, 0x75, 0xa5, 0xf1, 0xad, 0x4c, 0xde, 0x7f, 0x1a, 0x86, 0x24, 0x13, 0x2a,
];
const EXPECTED_SCHEMA_TAG: [u8; 16] = [
    0x34, 0x85, 0xfc, 0x08, 0xbf, 0xb9, 0xfb, 0xc7, 0xee, 0x7f, 0x30, 0x73, 0x4e, 0x9f, 0xd0, 0xa4,
];
const SECTION_OFFSETS: [u32; 7] = [256, 34016, 36424, 36936, 37452, 37968, 38996];
const SECTION_LENGTHS: [u32; 7] = [33760, 2408, 512, 514, 514, 1028, 514];
const EXPECTED_Q: [u8; 8] = [8, 15, 14, 16, 15, 20, 14, 15];
const EXPECTED_WEIGHT_COUNTS: [u32; 4] = [4800, 19200, 80, 9680];
const EXPECTED_BIAS_COUNTS: [u32; 4] = [240, 240, 1, 121];
const EXPECTED_GROUP_COUNTS: [u16; 4] = [3, 3, 1, 121];
const EXPECTED_CONFIG: [i32; 11] = [
    24576, 50412, 87381, 436907, 32768, 2097152, 32767, 0, 32768, 98304, 32768,
];

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    crc ^ 0xffff_ffff
}

fn u16_le(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn u32_le(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn i32_le(bytes: &[u8], at: usize) -> i32 {
    u32_le(bytes, at) as i32
}

fn f32_le(bytes: &[u8], at: usize) -> f32 {
    f32::from_bits(u32_le(bytes, at))
}

fn f16_le(bytes: &[u8], at: usize) -> f32 {
    let half = u32::from(u16_le(bytes, at));
    let sign = (half & 0x8000) << 16;
    let exponent = (half >> 10) & 0x1f;
    let mut fraction = half & 0x3ff;
    let bits = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let mut unbiased = -14i32;
            while fraction & 0x400 == 0 {
                fraction <<= 1;
                unbiased -= 1;
            }
            fraction &= 0x3ff;
            sign | (((unbiased + 127) as u32) << 23) | (fraction << 13)
        }
    } else if exponent == 31 {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | ((exponent + 112) << 23) | (fraction << 13)
    };
    f32::from_bits(bits)
}

pub struct FixedModel {
    pub weights: Vec<i8>,
    pub biases: Vec<i32>,
    pub multipliers: Vec<i32>,
    pub sigmoid: Vec<u16>,
    pub tanh: Vec<i16>,
    pub exp: Vec<u32>,
    pub log: Vec<i16>,
    pub config: [i32; 11],
    pub body_crc32: u32,
}

pub struct Adapter {
    pub mean: Vec<f32>,
    pub invstd: Vec<f32>,
    pub v_scale: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub v_q: Vec<i8>,
    pub u_scale: Vec<f32>,
    pub u_bias: Vec<f32>,
    pub u_q: Vec<i8>,
}

pub struct RendererModel {
    pub fixed: FixedModel,
    pub adapter: Adapter,
}

fn model_error(message: &str) -> Error {
    Error::ModelIntegrity(message.to_string())
}

impl FixedModel {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        if blob.len() != FIXED_BYTES {
            return Err(model_error("renderer image has an unexpected length"));
        }
        let header_ok = &blob[..8] == MAGIC
            && u16_le(blob, 8) == 1
            && u16_le(blob, 10) as usize == HEADER_BYTES
            && u16_le(blob, 12) as usize == HIDDEN
            && u16_le(blob, 14) as usize == FEATURES
            && i32::from(u16_le(blob, 16)) == RADIUS
            && u16_le(blob, 18) == 2
            && u32_le(blob, 20) == 15
            && u32_le(blob, 24) as usize == FIXED_BYTES;
        if !header_ok {
            return Err(model_error("renderer header does not match the release"));
        }
        for index in 0..7 {
            if u32_le(blob, 28 + 8 * index) != SECTION_OFFSETS[index]
                || u32_le(blob, 32 + 8 * index) != SECTION_LENGTHS[index]
            {
                return Err(model_error("renderer section table does not match"));
            }
        }
        if blob[88..96] != EXPECTED_Q {
            return Err(model_error("renderer fixed point exponents differ"));
        }
        for index in 0..4 {
            if u32_le(blob, 96 + 4 * index) != EXPECTED_WEIGHT_COUNTS[index]
                || u32_le(blob, 112 + 4 * index) != EXPECTED_BIAS_COUNTS[index]
                || u16_le(blob, 128 + 2 * index) != EXPECTED_GROUP_COUNTS[index]
            {
                return Err(model_error("renderer tensor counts differ"));
            }
        }
        let mut config = [0i32; 11];
        for index in 0..11 {
            config[index] = i32_le(blob, 136 + 4 * index);
            if config[index] != EXPECTED_CONFIG[index] {
                return Err(model_error("renderer sampling configuration differs"));
            }
        }
        if blob[180..212] != EXPECTED_SOURCE_SHA256 || blob[212..228] != EXPECTED_SCHEMA_TAG {
            return Err(Error::ModelIntegrity(
                "renderer source identity differs".into(),
            ));
        }
        if blob[228..HEADER_BYTES].iter().any(|&byte| byte != 0) {
            return Err(model_error("renderer header padding is not zero"));
        }
        if u32_le(blob, 84) != EXPECTED_BODY_CRC32 {
            return Err(Error::ModelIntegrity("renderer body crc differs".into()));
        }
        let crc = crc32(&blob[HEADER_BYTES..]);
        if crc != EXPECTED_BODY_CRC32 {
            return Err(Error::ModelIntegrity("renderer body crc differs".into()));
        }

        let section = |index: usize| {
            let start = SECTION_OFFSETS[index] as usize;
            &blob[start..start + SECTION_LENGTHS[index] as usize]
        };
        Ok(Self {
            weights: section(0).iter().map(|&byte| byte as i8).collect(),
            biases: section(1)
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            multipliers: section(2)
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            sigmoid: section(3)
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            tanh: section(4)
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            exp: section(5)
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            log: section(6)
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            config,
            body_crc32: crc,
        })
    }
}

impl Adapter {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        if blob.len() != ADAPTER_BYTES {
            return Err(model_error("renderer adapter has an unexpected length"));
        }
        if crc32(blob) != EXPECTED_ADAPTER_CRC32
            || &blob[..8] != ADAPTER_MAGIC
            || u32_le(blob, 8) != 1
            || u32_le(blob, 12) as usize != ADAPTER_INPUT
            || u32_le(blob, 16) as usize != ADAPTER_RANK
            || u32_le(blob, 20) as usize != HIDDEN
        {
            return Err(model_error("renderer adapter does not match the release"));
        }
        let mut at = ADAPTER_HEADER_BYTES;
        let halves = |blob: &[u8], at: usize, count: usize| -> Vec<f32> {
            (0..count)
                .map(|index| f16_le(blob, at + 2 * index))
                .collect()
        };
        let floats = |blob: &[u8], at: usize, count: usize| -> Vec<f32> {
            (0..count)
                .map(|index| f32_le(blob, at + 4 * index))
                .collect()
        };
        let mean = halves(blob, at, ADAPTER_INPUT);
        at += ADAPTER_INPUT * 2;
        let invstd = halves(blob, at, ADAPTER_INPUT);
        at += ADAPTER_INPUT * 2;
        let v_scale = floats(blob, at, ADAPTER_RANK);
        at += ADAPTER_RANK * 4;
        let v_bias = floats(blob, at, ADAPTER_RANK);
        at += ADAPTER_RANK * 4;
        let v_q: Vec<i8> = blob[at..at + ADAPTER_RANK * ADAPTER_INPUT]
            .iter()
            .map(|&byte| byte as i8)
            .collect();
        at += ADAPTER_RANK * ADAPTER_INPUT;
        let u_scale = floats(blob, at, HIDDEN);
        at += HIDDEN * 4;
        let u_bias = floats(blob, at, HIDDEN);
        at += HIDDEN * 4;
        let u_q: Vec<i8> = blob[at..at + HIDDEN * ADAPTER_RANK]
            .iter()
            .map(|&byte| byte as i8)
            .collect();
        at += HIDDEN * ADAPTER_RANK;
        if at != ADAPTER_BYTES {
            return Err(model_error("renderer adapter layout is inconsistent"));
        }
        Ok(Self {
            mean,
            invstd,
            v_scale,
            v_bias,
            v_q,
            u_scale,
            u_bias,
            u_q,
        })
    }
}

impl RendererModel {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        if blob.len() != ARTIFACT_BYTES {
            return Err(model_error("renderer artifact has an unexpected length"));
        }
        Ok(Self {
            fixed: FixedModel::parse(&blob[..FIXED_BYTES])?,
            adapter: Adapter::parse(&blob[FIXED_BYTES..])?,
        })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(&std::fs::read(path)?)
    }

    pub fn from_release(model_dir: Option<&Path>) -> Result<Self> {
        let root = model_dir.map_or_else(crate::model_store::default_model_dir, Path::to_path_buf);
        let path = root.join("renderer_global_h80.bin");
        let digest = crate::model_store::sha256(&path)?;
        if digest != ARTIFACT_SHA256 {
            return Err(Error::ModelIntegrity(format!(
                "release model hash differs for renderer_global_h80.bin: expected {ARTIFACT_SHA256}, observed {digest}"
            )));
        }
        Self::open(path)
    }
}
