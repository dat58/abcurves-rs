pub mod continuous;
pub mod error;
pub mod io;
pub mod math;
pub mod model_store;
pub mod nn;
pub mod rng;

pub use continuous::{
    Advance, AdvanceError, ContinuousOptions, HumanStart, MovementRuntime, RuntimeFailure,
    prepare_history,
};
pub use error::{Error, Result};
pub use nn::NeuralInference;

pub fn load(options: ContinuousOptions) -> Result<MovementRuntime> {
    continuous::load(options)
}
