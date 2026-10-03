#![allow(clippy::needless_range_loop)]

mod common;

use abcurves::rng::{Pcg64, RandomStream, head_from_seed, softmax};
use common::{read_f32, read_i64, read_u64};

const FIXTURE: &str = "rng_stream";

#[test]
fn stream_matches_reference_draws() {
    let seeds = read_u64(FIXTURE, "seeds");
    let uniform_a = read_f32(FIXTURE, "uniform_a");
    let exponential_a = read_f32(FIXTURE, "exponential_a");
    let uniform_b = read_f32(FIXTURE, "uniform_b");
    let exponential_b = read_f32(FIXTURE, "exponential_b");
    let uniform_c = read_f32(FIXTURE, "uniform_c");
    let logits = read_f32(FIXTURE, "logits");
    let categories = read_i64(FIXTURE, "categories");

    for (index, &seed) in seeds.iter().enumerate() {
        let mut stream = RandomStream::new(seed);
        assert_eq!(stream.uniform(1), uniform_a[index..index + 1]);
        assert_eq!(
            stream.exponential(16),
            exponential_a[index * 16..(index + 1) * 16]
        );
        assert_eq!(
            stream.uniform(1027),
            uniform_b[index * 1027..(index + 1) * 1027]
        );
        for step in 0..48 {
            let position = index * 48 + step;
            let weights = softmax(&logits[position * 16..(position + 1) * 16]).unwrap();
            if categories[position] < 0 {
                stream.discard_categorical(16);
            } else {
                assert_eq!(stream.categorical(&weights) as i64, categories[position]);
            }
        }
        assert_eq!(
            stream.exponential(300),
            exponential_b[index * 300..(index + 1) * 300]
        );
        assert_eq!(
            stream.uniform(129),
            uniform_c[index * 129..(index + 1) * 129]
        );
    }
}

#[test]
fn softmax_tracks_reference_within_a_few_units_in_last_place() {
    let logits = read_f32(FIXTURE, "logits");
    let probabilities = read_f32(FIXTURE, "probabilities");
    let mut worst = 0i64;
    for chunk in 0..logits.len() / 16 {
        let actual = softmax(&logits[chunk * 16..(chunk + 1) * 16]).unwrap();
        for lane in 0..16 {
            let expected = probabilities[chunk * 16 + lane];
            let distance = (actual[lane].to_bits() as i64 - expected.to_bits() as i64).abs();
            worst = worst.max(distance);
        }
    }
    assert!(
        worst <= 8,
        "softmax drifted by {worst} units in the last place; numpy uses its own SIMD expf"
    );
}

#[test]
fn pcg64_matches_numpy_default_generator() {
    let fixture = "rng_pcg64";
    let low = read_u64(fixture, "seeds");
    let high = read_u64(fixture, "seeds_high");
    let heads = read_i64(fixture, "heads");
    let words = read_u64(fixture, "words");

    for index in 0..low.len() {
        let seed = (u128::from(high[index]) << 64) | u128::from(low[index]);
        assert_eq!(head_from_seed(seed, 16) as i64, heads[index], "seed {seed}");
        let mut generator = Pcg64::from_seed(seed);
        for step in 0..16 {
            assert_eq!(
                generator.next_u64(),
                words[index * 16 + step],
                "seed {seed} step {step}"
            );
        }
    }
}
