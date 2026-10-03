#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub fn data(fixture: &str, field: &str, suffix: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("data")
        .join(fixture)
        .join(format!("{field}.{suffix}"));
    std::fs::read(&path).unwrap_or_else(|error| panic!("missing {}: {error}", path.display()))
}

pub fn f64s(fixture: &str, field: &str) -> Vec<f64> {
    data(fixture, field, "f64")
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

pub fn i16s(fixture: &str, field: &str) -> Vec<i16> {
    data(fixture, field, "i16")
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

pub fn pairs_f64(fixture: &str, field: &str) -> Vec<[f64; 2]> {
    f64s(fixture, field)
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect()
}

pub fn pairs_i16(fixture: &str, field: &str) -> Vec<[i16; 2]> {
    i16s(fixture, field)
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect()
}

/// Release models live wherever `ABCURVES_MODEL_DIR` points, or beside a clone
/// of the Python project.
pub fn models_present() -> bool {
    let root = abcurves::model_store::default_model_dir();
    if root.join("manifest.json").is_file() {
        return true;
    }
    eprintln!(
        "skipping: no release models under {}; set ABCURVES_MODEL_DIR",
        root.display()
    );
    false
}

pub fn models_root() -> PathBuf {
    abcurves::model_store::default_model_dir()
}
