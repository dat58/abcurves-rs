use std::fmt;

#[derive(Debug)]
pub enum Error {
    ModelIntegrity(String),
    InferenceContract(String),
    Format(String),
    Mode(String),
    Range(String),
    Numerical(String),
    Io(std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ModelIntegrity(m) => write!(f, "model integrity: {m}"),
            Error::InferenceContract(m) => write!(f, "inference contract: {m}"),
            Error::Format(m) => write!(f, "format: {m}"),
            Error::Mode(m) => write!(f, "mode: {m}"),
            Error::Range(m) => write!(f, "range: {m}"),
            Error::Numerical(m) => write!(f, "numerical: {m}"),
            Error::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::Io(value)
    }
}

pub fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::InferenceContract(message.into()))
    }
}
