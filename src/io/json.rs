use crate::error::{Error, Result};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(BTreeMap<String, Json>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(items) => items.get(key),
            _ => None,
        }
    }

    pub fn entry(&self, key: &str) -> Result<&Json> {
        self.get(key)
            .ok_or_else(|| Error::Format(format!("manifest key {key} is missing")))
    }

    pub fn as_str(&self) -> Result<&str> {
        match self {
            Json::String(text) => Ok(text),
            other => Err(Error::Format(format!("expected a string, found {other:?}"))),
        }
    }

    pub fn as_f64(&self) -> Result<f64> {
        match self {
            Json::Number(value) => Ok(*value),
            other => Err(Error::Format(format!("expected a number, found {other:?}"))),
        }
    }

    pub fn as_u64(&self) -> Result<u64> {
        Ok(self.as_f64()? as u64)
    }

    pub fn as_array(&self) -> Result<&[Json]> {
        match self {
            Json::Array(items) => Ok(items),
            other => Err(Error::Format(format!("expected an array, found {other:?}"))),
        }
    }

    pub fn as_object(&self) -> Result<&BTreeMap<String, Json>> {
        match self {
            Json::Object(items) => Ok(items),
            other => Err(Error::Format(format!("expected an object, found {other:?}"))),
        }
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Parser<'a> {
    fn skip(&mut self) {
        while self.at < self.bytes.len() && self.bytes[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }

    fn peek(&self) -> Result<u8> {
        self.bytes
            .get(self.at)
            .copied()
            .ok_or_else(|| Error::Format("json ended early".into()))
    }

    fn expect(&mut self, byte: u8) -> Result<()> {
        if self.peek()? != byte {
            return Err(Error::Format(format!(
                "json expected {:?} at byte {}",
                byte as char, self.at
            )));
        }
        self.at += 1;
        Ok(())
    }

    fn literal(&mut self, text: &str, value: Json) -> Result<Json> {
        if self.bytes[self.at..].starts_with(text.as_bytes()) {
            self.at += text.len();
            return Ok(value);
        }
        Err(Error::Format(format!("json literal at byte {}", self.at)))
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let byte = self.peek()?;
            self.at += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escape = self.peek()?;
                    self.at += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex = std::str::from_utf8(&self.bytes[self.at..self.at + 4])
                                .map_err(|_| Error::Format("json escape".into()))?;
                            let code = u32::from_str_radix(hex, 16)
                                .map_err(|_| Error::Format("json escape".into()))?;
                            self.at += 4;
                            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        }
                        other => {
                            return Err(Error::Format(format!("json escape {:?}", other as char)));
                        }
                    }
                }
                _ => {
                    let start = self.at - 1;
                    while self.at < self.bytes.len()
                        && self.bytes[self.at] != b'"'
                        && self.bytes[self.at] != b'\\'
                    {
                        self.at += 1;
                    }
                    out.push_str(&String::from_utf8_lossy(&self.bytes[start..self.at]));
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json> {
        let start = self.at;
        while self.at < self.bytes.len()
            && matches!(self.bytes[self.at], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        {
            self.at += 1;
        }
        std::str::from_utf8(&self.bytes[start..self.at])
            .ok()
            .and_then(|text| text.parse::<f64>().ok())
            .map(Json::Number)
            .ok_or_else(|| Error::Format(format!("json number at byte {start}")))
    }

    fn value(&mut self) -> Result<Json> {
        self.skip();
        match self.peek()? {
            b'{' => {
                self.at += 1;
                let mut items = BTreeMap::new();
                self.skip();
                if self.peek()? == b'}' {
                    self.at += 1;
                    return Ok(Json::Object(items));
                }
                loop {
                    self.skip();
                    let key = self.string()?;
                    self.skip();
                    self.expect(b':')?;
                    items.insert(key, self.value()?);
                    self.skip();
                    match self.peek()? {
                        b',' => self.at += 1,
                        b'}' => {
                            self.at += 1;
                            return Ok(Json::Object(items));
                        }
                        other => {
                            return Err(Error::Format(format!("json object {:?}", other as char)));
                        }
                    }
                }
            }
            b'[' => {
                self.at += 1;
                let mut items = Vec::new();
                self.skip();
                if self.peek()? == b']' {
                    self.at += 1;
                    return Ok(Json::Array(items));
                }
                loop {
                    items.push(self.value()?);
                    self.skip();
                    match self.peek()? {
                        b',' => self.at += 1,
                        b']' => {
                            self.at += 1;
                            return Ok(Json::Array(items));
                        }
                        other => {
                            return Err(Error::Format(format!("json array {:?}", other as char)));
                        }
                    }
                }
            }
            b'"' => Ok(Json::String(self.string()?)),
            b't' => self.literal("true", Json::Bool(true)),
            b'f' => self.literal("false", Json::Bool(false)),
            b'n' => self.literal("null", Json::Null),
            _ => self.number(),
        }
    }
}

pub fn parse(bytes: &[u8]) -> Result<Json> {
    let mut parser = Parser { bytes, at: 0 };
    let value = parser.value()?;
    parser.skip();
    Ok(value)
}

pub fn read(path: impl AsRef<std::path::Path>) -> Result<Json> {
    parse(&std::fs::read(path)?)
}
