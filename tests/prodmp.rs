#![allow(clippy::needless_range_loop)]

mod common;

use abcurves::static_planner::{ComponentCache, ProDMPConfig};
use common::{read_f64, read_i64};

const FIXTURE: &str = "prodmp";

fn weights_and_velocity(count: usize) -> (Vec<Vec<f64>>, Vec<[f64; 2]>) {
    let u = common::legacy_uniforms(4242, count * 44);
    let mut weights = Vec::new();
    let mut velocity = Vec::new();
    for index in 0..count {
        let base = index * 44;
        weights.push(
            (0..42)
                .map(|slot| (u[base + slot] - 0.5) * 2.0e4)
                .collect::<Vec<f64>>(),
        );
        velocity.push([(u[base + 42] - 0.5) * 40.0, (u[base + 43] - 0.5) * 40.0]);
    }
    (weights, velocity)
}

#[test]
fn basis_and_decoder_match_the_reference() {
    let durations = read_i64(FIXTURE, "durations");
    let horizon = *durations.iter().max().unwrap() as usize;
    let mut cache = ComponentCache::new(ProDMPConfig::default(), horizon);

    let s_grid = read_f64(FIXTURE, "s_grid");
    let phi_grid = read_f64(FIXTURE, "phi_grid");
    let dphi_grid = read_f64(FIXTURE, "dphi_grid");
    let weights = cache.prodmp().n_weights();
    let mut worst_basis = 0.0f64;
    for (sample, &expected_s) in s_grid.iter().enumerate() {
        let grid = sample * 10;
        assert_eq!(
            cache.prodmp().s_grid_value(grid),
            expected_s,
            "s grid {grid}"
        );
        for index in 0..weights {
            let phi = cache.prodmp().phi_grid()[grid * weights + index];
            let dphi = cache.prodmp().dphi_grid()[grid * weights + index];
            worst_basis = worst_basis
                .max((phi - phi_grid[sample * weights + index]).abs())
                .max((dphi - dphi_grid[sample * weights + index]).abs());
        }
    }
    eprintln!("basis gap {worst_basis:e}");
    assert!(worst_basis < 1e-14, "basis gap {worst_basis:e}");

    let xi1 = read_f64(FIXTURE, "xi1");
    let xi2 = read_f64(FIXTURE, "xi2");
    let deltas = read_f64(FIXTURE, "deltas");
    let (all_weights, all_velocity) = weights_and_velocity(durations.len());

    let mut xi_cursor = 0usize;
    let mut delta_cursor = 0usize;
    let mut worst_xi = 0.0f64;
    let mut worst_delta = 0.0f64;
    for (index, &duration) in durations.iter().enumerate() {
        let duration = duration as usize;
        let mut out = vec![[0.0f64; 2]; duration];
        {
            let components = cache.components(duration);
            for step in 0..duration {
                worst_xi = worst_xi
                    .max((components.xi1[step] - xi1[xi_cursor + step]).abs())
                    .max((components.xi2[step] - xi2[xi_cursor + step]).abs());
            }
        }
        let components = cache.components(duration).clone();
        cache.prodmp().generate_deltas(
            &components,
            &all_weights[index],
            all_velocity[index],
            &mut out,
        );
        for step in 0..duration {
            for axis in 0..2 {
                worst_delta = worst_delta
                    .max((out[step][axis] - deltas[delta_cursor + step * 2 + axis]).abs());
            }
        }
        xi_cursor += duration;
        delta_cursor += duration * 2;
    }
    eprintln!("xi gap {worst_xi:e}, delta gap {worst_delta:e}");
    assert!(worst_xi < 1e-12, "boundary components drifted {worst_xi:e}");
    assert!(worst_delta < 1e-9, "decoded deltas drifted {worst_delta:e}");
}
