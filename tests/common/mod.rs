#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub fn golden(fixture: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(fixture)
}

fn blob(fixture: &str, name: &str, suffix: &str) -> Vec<u8> {
    let path = golden(fixture).join(format!("{name}.{suffix}"));
    std::fs::read(&path).unwrap_or_else(|error| panic!("missing {}: {error}", path.display()))
}

macro_rules! reader {
    ($name:ident, $kind:ty, $suffix:literal, $width:literal) => {
        pub fn $name(fixture: &str, field: &str) -> Vec<$kind> {
            blob(fixture, field, $suffix)
                .chunks_exact($width)
                .map(|chunk| <$kind>::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        }
    };
}

reader!(read_f32, f32, "f32", 4);
reader!(read_f64, f64, "f64", 8);
reader!(read_i16, i16, "i16", 2);
reader!(read_i32, i32, "i32", 4);
reader!(read_i64, i64, "i64", 8);
reader!(read_u64, u64, "u64", 8);
reader!(read_i8, i8, "i8", 1);
reader!(read_u8, u8, "u8", 1);

pub fn models_root() -> PathBuf {
    if let Ok(explicit) = std::env::var("ABCURVES_MODEL_DIR") {
        return PathBuf::from(explicit);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("origin")
        .join("ABCurves")
        .join("models")
}
