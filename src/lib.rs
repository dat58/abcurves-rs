pub mod continuous;
pub mod error;
pub mod io;
pub mod math;
pub mod model_store;
pub mod nn;
pub mod renderer;
pub mod rng;
pub mod static_planner;

pub use continuous::{
    Advance, AdvanceError, ContinuousOptions, ContinuousPipeline, CountTransform, HumanStart,
    MovementRuntime, PipelineAdvance, PipelineError, PipelineOptions, RuntimeFailure,
    prepare_history,
};
pub use error::{Error, Result};
pub use renderer::{RendererModel, RendererProfile, RendererStream};
pub use static_planner::{BEvent, BTrigger, FastPlanner, Intent, OnsetDetector, StaticPipeline};
pub use nn::NeuralInference;

pub fn load(options: ContinuousOptions) -> Result<MovementRuntime> {
    continuous::load(options)
}
