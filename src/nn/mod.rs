pub mod activation;
pub mod linear;
pub mod native;

#[cfg(feature = "candle")]
pub mod candle;

use crate::continuous::constants::HEADS;
use crate::error::Result;
use crate::io::Npz;

pub use native::{EventOutputs, NativeEngine, hazard_probabilities};

/// Which arithmetic evaluates the learned tensors. `Native` is the accepted
/// release path; `Candle` is a numerical and portability reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NeuralInference {
    #[default]
    Native,
    Candle,
}

pub enum Engine {
    Native(Box<NativeEngine>),
    #[cfg(feature = "candle")]
    Candle(Box<candle::CandleEngine>),
}

impl Engine {
    pub fn load(bundle: &Npz, backend: NeuralInference) -> Result<Self> {
        match backend {
            NeuralInference::Native => Ok(Engine::Native(Box::new(NativeEngine::load(bundle)?))),
            #[cfg(feature = "candle")]
            NeuralInference::Candle => {
                Ok(Engine::Candle(Box::new(candle::CandleEngine::load(bundle)?)))
            }
            #[cfg(not(feature = "candle"))]
            NeuralInference::Candle => Err(crate::error::Error::InferenceContract(
                "the candle backend requires the 'candle' feature".into(),
            )),
        }
    }

    pub fn backend(&self) -> NeuralInference {
        match self {
            Engine::Native(_) => NeuralInference::Native,
            #[cfg(feature = "candle")]
            Engine::Candle(_) => NeuralInference::Candle,
        }
    }

    pub fn motor(&mut self, coarse: &[f32], fine: &[f32], dynamics: &[f32]) -> Result<()> {
        match self {
            Engine::Native(engine) => {
                engine.motor(coarse, fine, dynamics);
                Ok(())
            }
            #[cfg(feature = "candle")]
            Engine::Candle(engine) => engine.motor(coarse, fine, dynamics),
        }
    }

    pub fn encoded(&self) -> &[f32] {
        match self {
            Engine::Native(engine) => &engine.encoded,
            #[cfg(feature = "candle")]
            Engine::Candle(engine) => &engine.encoded,
        }
    }

    pub fn set_encoded(&mut self, values: &[f32]) {
        match self {
            Engine::Native(engine) => engine.encoded.copy_from_slice(values),
            #[cfg(feature = "candle")]
            Engine::Candle(engine) => engine.encoded.copy_from_slice(values),
        }
    }

    pub fn coefficients(&self) -> &[f32] {
        match self {
            Engine::Native(engine) => &engine.coefficients,
            #[cfg(feature = "candle")]
            Engine::Candle(engine) => &engine.coefficients,
        }
    }

    pub fn choice(&mut self, geometry: &[f32], pairs: &[f32], previous_valid: bool) -> Result<()> {
        match self {
            Engine::Native(engine) => {
                engine.choice(geometry, pairs, previous_valid);
                Ok(())
            }
            #[cfg(feature = "candle")]
            Engine::Candle(engine) => engine.choice(geometry, pairs, previous_valid),
        }
    }

    pub fn logits(&self) -> [f32; HEADS] {
        match self {
            Engine::Native(engine) => engine.logits,
            #[cfg(feature = "candle")]
            Engine::Candle(engine) => engine.logits,
        }
    }

    pub fn hazard(&mut self, context: &[f32], out: &mut [f32; 2]) -> Result<()> {
        match self {
            Engine::Native(engine) => {
                engine.hazard(context, out);
                Ok(())
            }
            #[cfg(feature = "candle")]
            Engine::Candle(engine) => engine.hazard(context, out),
        }
    }

    pub fn brake(&mut self, context: &[f32], out: &mut EventOutputs) -> Result<()> {
        match self {
            Engine::Native(engine) => {
                engine.brake(context, out);
                Ok(())
            }
            #[cfg(feature = "candle")]
            Engine::Candle(engine) => engine.brake(context, out),
        }
    }

    pub fn events(&mut self, context: &[f32], out: &mut EventOutputs) -> Result<()> {
        match self {
            Engine::Native(engine) => {
                engine.events(context, out);
                Ok(())
            }
            #[cfg(feature = "candle")]
            Engine::Candle(engine) => engine.events(context, out),
        }
    }
}
