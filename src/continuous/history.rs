use super::constants::WARM_HISTORY_SAMPLES;
use crate::error::{Error, Result};

pub const RAW_HISTORY_SAMPLES: usize = 164;

#[derive(Clone, Debug)]
pub struct HumanStart {
    pub initial_xy: [f64; 2],
    pub history: Vec<[f64; 2]>,
    pub observed_xy: [f64; 2],
}

/// The selected training representation filters positions causally with W5
/// weights [1,2,3,2,1]/9, then differences them. Four extra genuine samples
/// avoid inventing stationary history.
pub fn prepare_history(raw_displacements: &[[f64; 2]], current_xy: [f64; 2]) -> Result<HumanStart> {
    let finite = raw_displacements
        .iter()
        .flat_map(|row| row.iter())
        .all(|value| value.is_finite());
    if raw_displacements.len() != RAW_HISTORY_SAMPLES || !finite {
        return Err(Error::InferenceContract(
            "raw_displacements must contain exactly 164 finite 1 ms pairs".into(),
        ));
    }
    if !current_xy.iter().all(|value| value.is_finite()) {
        return Err(Error::InferenceContract(
            "Position must be a finite pair in common counts".into(),
        ));
    }
    let weights = [1.0 / 9.0, 2.0 / 9.0, 3.0 / 9.0, 2.0 / 9.0, 1.0 / 9.0];
    let mut history = vec![[0.0f64; 2]; WARM_HISTORY_SAMPLES];
    for (tap, weight) in weights.iter().enumerate() {
        for step in 0..WARM_HISTORY_SAMPLES {
            history[step][0] += weight * raw_displacements[tap + step][0];
            history[step][1] += weight * raw_displacements[tap + step][1];
        }
    }
    let last = RAW_HISTORY_SAMPLES - 1;
    let lag = [
        (8.0 * raw_displacements[last][0]
            + 6.0 * raw_displacements[last - 1][0]
            + 3.0 * raw_displacements[last - 2][0]
            + raw_displacements[last - 3][0])
            / 9.0,
        (8.0 * raw_displacements[last][1]
            + 6.0 * raw_displacements[last - 1][1]
            + 3.0 * raw_displacements[last - 2][1]
            + raw_displacements[last - 3][1])
            / 9.0,
    ];
    let anchor = [current_xy[0] - lag[0], current_xy[1] - lag[1]];
    let valid = history
        .iter()
        .flat_map(|row| row.iter())
        .chain(anchor.iter())
        .all(|value| value.is_finite());
    if !valid {
        return Err(Error::Numerical(
            "History filtering exceeds finite coordinate range".into(),
        ));
    }
    Ok(HumanStart {
        initial_xy: anchor,
        history,
        observed_xy: current_xy,
    })
}
