use super::planner::Planner;
use super::runtime::MovementRuntime;
use crate::error::Result;
use crate::io::Npz;
use crate::model_store;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ContinuousOptions {
    pub assets: Option<PathBuf>,
    pub seed: u64,
    pub initial_xy: [f64; 2],
    pub history: Option<Vec<[f64; 2]>>,
    pub prewarm: bool,
    pub skip_unused: bool,
    pub verify_assets: bool,
    pub allow_custom_assets: bool,
}

impl Default for ContinuousOptions {
    fn default() -> Self {
        Self {
            assets: None,
            seed: 7,
            initial_xy: [0.0, 0.0],
            history: None,
            prewarm: true,
            skip_unused: true,
            verify_assets: true,
            allow_custom_assets: false,
        }
    }
}

impl ContinuousOptions {
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn initial_xy(mut self, initial_xy: [f64; 2]) -> Self {
        self.initial_xy = initial_xy;
        self
    }

    pub fn history(mut self, history: Vec<[f64; 2]>) -> Self {
        self.history = Some(history);
        self
    }

    pub fn assets(mut self, assets: impl AsRef<Path>) -> Self {
        self.assets = Some(assets.as_ref().to_path_buf());
        self
    }

    pub fn prewarm(mut self, prewarm: bool) -> Self {
        self.prewarm = prewarm;
        self
    }

    pub fn skip_unused(mut self, skip_unused: bool) -> Self {
        self.skip_unused = skip_unused;
        self
    }

    pub fn verify_assets(mut self, verify: bool) -> Self {
        self.verify_assets = verify;
        self
    }

    pub fn load(self) -> Result<MovementRuntime> {
        load(self)
    }
}

pub fn load(options: ContinuousOptions) -> Result<MovementRuntime> {
    let directory = options
        .assets
        .clone()
        .unwrap_or_else(model_store::default_continuous_dir);
    if options.verify_assets {
        model_store::verify_continuous_assets(&directory, options.allow_custom_assets)?;
    }
    let bundle = Npz::open(directory.join("weights.npz"))?;
    let mut planner = Planner::load(&bundle, options.skip_unused)?;
    if options.prewarm {
        planner.prewarm();
    }
    MovementRuntime::new(
        planner,
        options.initial_xy,
        options.history.as_deref(),
        options.seed,
    )
}
