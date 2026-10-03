use super::array::DType;
use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct StorageRef {
    pub key: String,
    pub dtype: DType,
    pub elements: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorRef {
    pub storage: StorageRef,
    pub offset: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    Storage(StorageRef),
    Tensor(TensorRef),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Dict(items) => items.iter().find_map(|(name, value)| match name {
                Value::Str(text) if text == key => Some(value),
                _ => None,
            }),
            _ => None,
        }
    }

    pub fn entry(&self, key: &str) -> Result<&Value> {
        self.get(key)
            .ok_or_else(|| Error::Format(format!("checkpoint key {key} is missing")))
    }

    pub fn as_str(&self) -> Result<&str> {
        match self {
            Value::Str(text) => Ok(text),
            other => Err(Error::Format(format!("expected a string, found {other:?}"))),
        }
    }

    pub fn as_i64(&self) -> Result<i64> {
        match self {
            Value::Int(value) => Ok(*value),
            Value::Bool(value) => Ok(i64::from(*value)),
            other => Err(Error::Format(format!(
                "expected an integer, found {other:?}"
            ))),
        }
    }

    pub fn as_usize(&self) -> Result<usize> {
        let value = self.as_i64()?;
        usize::try_from(value).map_err(|_| Error::Format(format!("negative length {value}")))
    }

    pub fn as_f64(&self) -> Result<f64> {
        match self {
            Value::Float(value) => Ok(*value),
            Value::Int(value) => Ok(*value as f64),
            other => Err(Error::Format(format!("expected a float, found {other:?}"))),
        }
    }

    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Value::Bool(value) => Ok(*value),
            other => Err(Error::Format(format!("expected a bool, found {other:?}"))),
        }
    }

    pub fn as_sequence(&self) -> Result<&[Value]> {
        match self {
            Value::List(items) | Value::Tuple(items) => Ok(items),
            other => Err(Error::Format(format!(
                "expected a sequence, found {other:?}"
            ))),
        }
    }

    pub fn as_tensor(&self) -> Result<&TensorRef> {
        match self {
            Value::Tensor(tensor) => Ok(tensor),
            other => Err(Error::Format(format!("expected a tensor, found {other:?}"))),
        }
    }
}

fn storage_dtype(name: &str) -> Result<DType> {
    Ok(match name {
        "FloatStorage" => DType::F32,
        "DoubleStorage" => DType::F64,
        "HalfStorage" => DType::F16,
        "LongStorage" => DType::I64,
        "IntStorage" => DType::I32,
        "ShortStorage" => DType::I16,
        "CharStorage" => DType::I8,
        "ByteStorage" => DType::U8,
        "BoolStorage" => DType::Bool,
        other => return Err(Error::Format(format!("unsupported storage type {other}"))),
    })
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn byte(&mut self) -> Result<u8> {
        let value = *self
            .bytes
            .get(self.at)
            .ok_or_else(|| Error::Format("pickle stream ended".into()))?;
        self.at += 1;
        Ok(value)
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let slice = self
            .bytes
            .get(self.at..self.at + count)
            .ok_or_else(|| Error::Format("pickle stream ended".into()))?;
        self.at += count;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn line(&mut self) -> Result<String> {
        let end = self.bytes[self.at..]
            .iter()
            .position(|&byte| byte == b'\n')
            .ok_or_else(|| Error::Format("pickle line is unterminated".into()))?;
        let text = String::from_utf8_lossy(&self.bytes[self.at..self.at + end]).into_owned();
        self.at += end + 1;
        Ok(text)
    }
}

pub fn load(bytes: &[u8]) -> Result<Value> {
    let mut reader = Reader { bytes, at: 0 };
    let mut stack: Vec<Value> = Vec::new();
    let mut marks: Vec<usize> = Vec::new();
    let mut memo: Vec<Value> = Vec::new();

    let remember = |memo: &mut Vec<Value>, index: usize, value: Value| {
        if memo.len() <= index {
            memo.resize(index + 1, Value::None);
        }
        memo[index] = value;
    };

    loop {
        match reader.byte()? {
            0x80 => {
                reader.byte()?;
            }
            b'.' => break,
            b'N' => stack.push(Value::None),
            0x88 => stack.push(Value::Bool(true)),
            0x89 => stack.push(Value::Bool(false)),
            b'J' => {
                let raw = reader.u32()?;
                stack.push(Value::Int(i64::from(raw as i32)));
            }
            b'K' => {
                let raw = reader.byte()?;
                stack.push(Value::Int(i64::from(raw)));
            }
            b'M' => {
                let raw = u16::from_le_bytes(reader.take(2)?.try_into().unwrap());
                stack.push(Value::Int(i64::from(raw)));
            }
            0x8a => {
                let length = usize::from(reader.byte()?);
                let raw = reader.take(length)?;
                let mut value = 0i128;
                for (index, &byte) in raw.iter().enumerate() {
                    value |= i128::from(byte) << (8 * index);
                }
                if let Some(&last) = raw.last() {
                    if last & 0x80 != 0 {
                        value -= 1i128 << (8 * length);
                    }
                }
                stack.push(Value::Int(value as i64));
            }
            b'G' => {
                let raw = reader.take(8)?;
                stack.push(Value::Float(f64::from_be_bytes(raw.try_into().unwrap())));
            }
            b'X' => {
                let length = reader.u32()? as usize;
                let raw = reader.take(length)?;
                stack.push(Value::Str(String::from_utf8_lossy(raw).into_owned()));
            }
            b'U' => {
                let length = usize::from(reader.byte()?);
                let raw = reader.take(length)?;
                stack.push(Value::Bytes(raw.to_vec()));
            }
            b'T' => {
                let length = reader.u32()? as usize;
                let raw = reader.take(length)?;
                stack.push(Value::Bytes(raw.to_vec()));
            }
            b'(' => marks.push(stack.len()),
            b'}' => stack.push(Value::Dict(Vec::new())),
            b']' => stack.push(Value::List(Vec::new())),
            b')' => stack.push(Value::Tuple(Vec::new())),
            b't' => {
                let mark = marks
                    .pop()
                    .ok_or_else(|| Error::Format("pickle mark underflow".into()))?;
                let items = stack.split_off(mark);
                stack.push(Value::Tuple(items));
            }
            0x85..=0x87 => {
                let count = usize::from(reader.bytes[reader.at - 1] - 0x84);
                let items = stack.split_off(stack.len() - count);
                stack.push(Value::Tuple(items));
            }
            b'q' => {
                let index = usize::from(reader.byte()?);
                remember(
                    &mut memo,
                    index,
                    stack.last().cloned().unwrap_or(Value::None),
                );
            }
            b'r' => {
                let index = reader.u32()? as usize;
                remember(
                    &mut memo,
                    index,
                    stack.last().cloned().unwrap_or(Value::None),
                );
            }
            b'h' => {
                let index = usize::from(reader.byte()?);
                stack.push(memo[index].clone());
            }
            b'j' => {
                let index = reader.u32()? as usize;
                stack.push(memo[index].clone());
            }
            b'a' => {
                let item = stack.pop().unwrap();
                match stack.last_mut() {
                    Some(Value::List(items)) => items.push(item),
                    _ => return Err(Error::Format("pickle append onto a non list".into())),
                }
            }
            b'e' => {
                let mark = marks
                    .pop()
                    .ok_or_else(|| Error::Format("pickle mark underflow".into()))?;
                let appended = stack.split_off(mark);
                match stack.last_mut() {
                    Some(Value::List(items)) => items.extend(appended),
                    _ => return Err(Error::Format("pickle appends onto a non list".into())),
                }
            }
            b's' => {
                let item = stack.pop().unwrap();
                let key = stack.pop().unwrap();
                match stack.last_mut() {
                    Some(Value::Dict(items)) => items.push((key, item)),
                    _ => return Err(Error::Format("pickle setitem onto a non dict".into())),
                }
            }
            b'u' => {
                let mark = marks
                    .pop()
                    .ok_or_else(|| Error::Format("pickle mark underflow".into()))?;
                let pairs = stack.split_off(mark);
                match stack.last_mut() {
                    Some(Value::Dict(items)) => {
                        for pair in pairs.chunks_exact(2) {
                            items.push((pair[0].clone(), pair[1].clone()));
                        }
                    }
                    _ => return Err(Error::Format("pickle setitems onto a non dict".into())),
                }
            }
            b'c' => {
                let module = reader.line()?;
                let name = reader.line()?;
                stack.push(Value::Global(module, name));
            }
            b'Q' => {
                let identifier = stack.pop().unwrap();
                stack.push(persistent(identifier)?);
            }
            b'R' => {
                let arguments = stack.pop().unwrap();
                let callable = stack.pop().unwrap();
                stack.push(reduce(callable, arguments)?);
            }
            opcode => {
                return Err(Error::Format(format!(
                    "unsupported pickle opcode {opcode:#04x} at {}",
                    reader.at - 1
                )));
            }
        }
    }

    stack
        .pop()
        .ok_or_else(|| Error::Format("pickle stream produced no value".into()))
}

fn persistent(identifier: Value) -> Result<Value> {
    let items = identifier.as_sequence()?;
    if items.len() != 5 || items[0].as_str()? != "storage" {
        return Err(Error::Format("unsupported persistent identifier".into()));
    }
    let dtype = match &items[1] {
        Value::Global(_, name) => storage_dtype(name)?,
        other => {
            return Err(Error::Format(format!(
                "expected a storage type, found {other:?}"
            )));
        }
    };
    Ok(Value::Storage(StorageRef {
        key: items[2].as_str()?.to_string(),
        dtype,
        elements: items[4].as_usize()?,
    }))
}

fn reduce(callable: Value, arguments: Value) -> Result<Value> {
    let (module, name) = match callable {
        Value::Global(module, name) => (module, name),
        other => return Err(Error::Format(format!("cannot call {other:?}"))),
    };
    match (module.as_str(), name.as_str()) {
        ("collections", "OrderedDict") => Ok(Value::Dict(Vec::new())),
        ("torch._utils", "_rebuild_tensor_v2") => {
            let items = arguments.as_sequence()?;
            let storage = match &items[0] {
                Value::Storage(storage) => storage.clone(),
                other => {
                    return Err(Error::Format(format!(
                        "expected a storage, found {other:?}"
                    )));
                }
            };
            let dimensions = |value: &Value| -> Result<Vec<usize>> {
                value.as_sequence()?.iter().map(Value::as_usize).collect()
            };
            Ok(Value::Tensor(TensorRef {
                storage,
                offset: items[1].as_usize()?,
                shape: dimensions(&items[2])?,
                stride: dimensions(&items[3])?,
            }))
        }
        _ => Err(Error::Format(format!(
            "unsupported constructor {module}.{name}"
        ))),
    }
}
