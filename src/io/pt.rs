use super::array::{Array, DType};
use super::pickle::{self, TensorRef, Value};
use super::zip::ZipArchive;
use crate::error::{Error, Result};
use std::path::Path;

pub struct Checkpoint {
    archive: ZipArchive,
    prefix: String,
    root: Value,
}

impl Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(std::fs::read(path)?)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let archive = ZipArchive::from_bytes(bytes)?;
        let pickle_name = archive
            .names()
            .iter()
            .find(|name| name.ends_with("data.pkl"))
            .cloned()
            .ok_or_else(|| Error::Format("checkpoint has no data.pkl".into()))?;
        let prefix = pickle_name.trim_end_matches("data.pkl").to_string();
        if let Some(order) = archive.get(&format!("{prefix}byteorder")) {
            if order != b"little" {
                return Err(Error::Format("checkpoint is not little endian".into()));
            }
        }
        let root = pickle::load(archive.read(&pickle_name)?)?;
        Ok(Self {
            archive,
            prefix,
            root,
        })
    }

    pub fn root(&self) -> &Value {
        &self.root
    }

    pub fn tensor(&self, reference: &TensorRef) -> Result<Array> {
        let width = reference.storage.dtype.width();
        let raw = self
            .archive
            .read(&format!("{}data/{}", self.prefix, reference.storage.key))?;
        let count: usize = reference.shape.iter().product();
        let expected: Vec<usize> = reference
            .shape
            .iter()
            .rev()
            .scan(1usize, |step, &extent| {
                let value = *step;
                *step *= extent;
                Some(value)
            })
            .collect();
        if reference.stride != expected.into_iter().rev().collect::<Vec<_>>() {
            return Err(Error::Format("checkpoint tensor is not contiguous".into()));
        }
        let start = reference.offset * width;
        let bytes = raw
            .get(start..start + count * width)
            .ok_or_else(|| Error::Format("checkpoint tensor runs past its storage".into()))?;
        Ok(Array {
            shape: reference.shape.clone(),
            dtype: reference.storage.dtype,
            bytes: bytes.to_vec(),
        })
    }

    pub fn array(&self, path: &[&str]) -> Result<Array> {
        let mut value = &self.root;
        for key in path {
            value = value.entry(key)?;
        }
        self.tensor(value.as_tensor()?)
    }

    pub fn f32(&self, path: &[&str]) -> Result<Vec<f32>> {
        let array = self.array(path)?;
        if array.dtype != DType::F32 {
            return Err(Error::Format(format!(
                "{path:?} is {:?}, not float32",
                array.dtype
            )));
        }
        array.to_f32()
    }
}
