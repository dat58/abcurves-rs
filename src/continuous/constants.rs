pub const COMMON_RADIANS_PER_COUNT: f64 = 0.0003509487083280618;
pub const SAMPLE_US: i64 = 1000;
pub const COMMIT_SAMPLES: usize = 32;
pub const HISTORY_SAMPLES: usize = 640;
pub const WARM_HISTORY_SAMPLES: usize = 160;
pub const FORECAST_SAMPLES: usize = 128;

pub const VELOCITY_SCALE: f64 = 1.1982087601807259;
pub const POSITION_SCALE: f64 = 81.73350722609987;

pub const COARSE_TOKENS: usize = 70;
pub const COARSE_FEATURES: usize = 9;
pub const COARSE_LEN: usize = COARSE_TOKENS * COARSE_FEATURES;
pub const FINE_LEN: usize = 54;
pub const DYNAMICS_LEN: usize = 8;
pub const EVENT_RAW_LEN: usize = 22;
pub const EVENT_CONTEXT_LEN: usize = 32;

pub const HEADS: usize = 16;
pub const WEIGHTS: usize = 21;
pub const HIDDEN: usize = 96;
pub const GEOMETRY_STEPS: usize = 16;
pub const PAIR_LEN: usize = 20;

pub const EVENT_SCALES: [f64; EVENT_RAW_LEN] = [
    10.0, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 10.0, 128.0, 1.0, 1.0, 1.0, 128.0,
    10.0, 10.0, 0.1, 1.0, 10.0,
];

pub const MODE_MOVE: u8 = 0;
pub const MODE_BRAKE: u8 = 1;
pub const MODE_HOLD: u8 = 2;
