use super::array::Array;
use super::npy;
use super::zip::ZipArchive;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

pub struct Npz {
    arrays: HashMap<String, Array>,
    order: Vec<String>,
}

impl Npz {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(std::fs::read(path)?)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let archive = ZipArchive::from_bytes(bytes)?;
        let mut arrays = HashMap::new();
        let mut order = Vec::new();
        for name in archive.names() {
            let key = name.strip_suffix(".npy").unwrap_or(name).to_string();
            arrays.insert(key.clone(), npy::parse(archive.read(name)?)?);
            order.push(key);
        }
        Ok(Self { arrays, order })
    }

    pub fn names(&self) -> &[String] {
        &self.order
    }

    pub fn get(&self, name: &str) -> Option<&Array> {
        self.arrays.get(name)
    }

    pub fn array(&self, name: &str) -> Result<&Array> {
        self.get(name)
            .ok_or_else(|| Error::Format(format!("npz entry {name} is missing")))
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        self.array(name)?.to_f32()
    }

    pub fn f64(&self, name: &str) -> Result<Vec<f64>> {
        self.array(name)?.to_f64()
    }
}
