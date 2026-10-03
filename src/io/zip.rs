use crate::error::{Error, Result};
use std::collections::HashMap;

const END_OF_CENTRAL_DIRECTORY: u32 = 0x0605_4b50;
const CENTRAL_FILE_HEADER: u32 = 0x0201_4b50;

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16> {
    bytes
        .get(offset..offset + 2)
        .map(|slice| u16::from_le_bytes(slice.try_into().unwrap()))
        .ok_or_else(|| Error::Format("truncated zip record".into()))
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    bytes
        .get(offset..offset + 4)
        .map(|slice| u32::from_le_bytes(slice.try_into().unwrap()))
        .ok_or_else(|| Error::Format("truncated zip record".into()))
}

pub struct ZipArchive {
    bytes: Vec<u8>,
    entries: HashMap<String, (usize, usize)>,
    order: Vec<String>,
}

impl ZipArchive {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_bytes(std::fs::read(path)?)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let end = (0..bytes.len().saturating_sub(21))
            .rev()
            .find(|&offset| u32_at(&bytes, offset).is_ok_and(|tag| tag == END_OF_CENTRAL_DIRECTORY))
            .ok_or_else(|| Error::Format("zip end of central directory not found".into()))?;
        let count = usize::from(u16_at(&bytes, end + 10)?);
        let mut cursor = u32_at(&bytes, end + 16)? as usize;

        let mut entries = HashMap::with_capacity(count);
        let mut order = Vec::with_capacity(count);
        for _ in 0..count {
            if u32_at(&bytes, cursor)? != CENTRAL_FILE_HEADER {
                return Err(Error::Format("zip central directory is corrupt".into()));
            }
            let method = u16_at(&bytes, cursor + 10)?;
            let size = u32_at(&bytes, cursor + 20)? as usize;
            let name_length = usize::from(u16_at(&bytes, cursor + 28)?);
            let extra_length = usize::from(u16_at(&bytes, cursor + 30)?);
            let comment_length = usize::from(u16_at(&bytes, cursor + 32)?);
            let local = u32_at(&bytes, cursor + 42)? as usize;
            let name = String::from_utf8_lossy(
                bytes
                    .get(cursor + 46..cursor + 46 + name_length)
                    .ok_or_else(|| Error::Format("truncated zip name".into()))?,
            )
            .into_owned();
            if method != 0 {
                return Err(Error::Format(format!(
                    "zip entry {name} uses compression method {method}; only stored is supported"
                )));
            }
            let local_name = usize::from(u16_at(&bytes, local + 26)?);
            let local_extra = usize::from(u16_at(&bytes, local + 28)?);
            let start = local + 30 + local_name + local_extra;
            if start + size > bytes.len() {
                return Err(Error::Format(format!(
                    "zip entry {name} runs past the file"
                )));
            }
            order.push(name.clone());
            entries.insert(name, (start, size));
            cursor += 46 + name_length + extra_length + comment_length;
        }
        Ok(Self {
            bytes,
            entries,
            order,
        })
    }

    pub fn names(&self) -> &[String] {
        &self.order
    }

    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.entries
            .get(name)
            .map(|&(start, size)| &self.bytes[start..start + size])
    }

    pub fn read(&self, name: &str) -> Result<&[u8]> {
        self.get(name)
            .ok_or_else(|| Error::Format(format!("zip entry {name} is missing")))
    }
}
