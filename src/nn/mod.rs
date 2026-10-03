pub mod activation;
pub mod linear;
pub mod native;

#[cfg(feature = "candle")]
pub mod candle;

pub use native::{EventOutputs, NativeEngine, hazard_probabilities};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NeuralInference {
    #[default]
    Native,
    Candle,
}
