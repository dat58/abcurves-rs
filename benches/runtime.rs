use abcurves::continuous::{
    ContinuousOptions, ContinuousPipeline, CountTransform, PipelineOptions,
};
use abcurves::renderer::{RendererModel, RendererProfile};
use abcurves::static_planner::StaticPipeline;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn models_root() -> PathBuf {
    if let Ok(explicit) = std::env::var("ABCURVES_MODEL_DIR") {
        return PathBuf::from(explicit);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("origin")
        .join("ABCurves")
        .join("models")
}

struct Samples {
    label: &'static str,
    values: Vec<f64>,
}

impl Samples {
    fn new(label: &'static str) -> Self {
        Self {
            label,
            values: Vec::new(),
        }
    }

    fn push(&mut self, nanoseconds: u128) {
        self.values.push(nanoseconds as f64 / 1000.0);
    }

    fn report(&mut self) {
        if self.values.is_empty() {
            return;
        }
        self.values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let count = self.values.len();
        let median = self.values[count / 2];
        let p95 = self.values[(count as f64 * 0.95) as usize % count];
        let max = self.values[count - 1];
        println!(
            "| {:<48} | {:>7} | {:>10.3} | {:>10.3} | {:>10.3} |",
            self.label, count, median, p95, max
        );
    }
}

fn golden(fixture: &str, name: &str, suffix: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(fixture)
        .join(format!("{name}.{suffix}"));
    std::fs::read(path).unwrap_or_default()
}

fn read_i16(fixture: &str, name: &str) -> Vec<i16> {
    golden(fixture, name, "i16")
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_f64(fixture: &str, name: &str) -> Vec<f64> {
    golden(fixture, name, "f64")
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_f32(fixture: &str, name: &str) -> Vec<f32> {
    golden(fixture, name, "f32")
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_i64(fixture: &str, name: &str) -> Vec<i64> {
    golden(fixture, name, "i64")
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn continuous(root: &Path) {
    let mut fresh_motor = Samples::new("Continuous fresh call, motor evaluated");
    let mut fresh_skipped = Samples::new("Continuous fresh call, motor skipped");
    let mut buffered = Samples::new("Continuous buffered 1 ms output");
    let mut receipt = Samples::new("Continuous target receipt");
    let mut reset = Samples::new("Continuous reset");

    for seed in 7u64..14 {
        let mut runtime = ContinuousOptions::default()
            .seed(seed)
            .assets(root.join("continuous"))
            .load()
            .unwrap();
        for tick in 1..=8_000i64 {
            if tick % 16 == 1 {
                let target = [
                    400.0 + 120.0 * (tick as f64 / 900.0).sin(),
                    -150.0 + 90.0 * (tick as f64 / 700.0).cos(),
                ];
                let start = Instant::now();
                runtime.update_target(target, (tick - 1) * 1000).unwrap();
                receipt.push(start.elapsed().as_nanos());
            }
            let before_fresh = runtime.decisions();
            let before_motor = runtime.motor_evaluations();
            let start = Instant::now();
            let block = runtime.advance(tick * 1000).unwrap();
            let elapsed = start.elapsed().as_nanos();
            std::hint::black_box(&block);
            if runtime.decisions() == before_fresh {
                buffered.push(elapsed);
            } else if runtime.motor_evaluations() == before_motor {
                fresh_skipped.push(elapsed);
            } else {
                fresh_motor.push(elapsed);
            }
        }
        let start = Instant::now();
        runtime.reset(None, None, None).unwrap();
        reset.push(start.elapsed().as_nanos());
    }
    fresh_motor.report();
    fresh_skipped.report();
    buffered.report();
    receipt.report();
    reset.report();
}

fn composed(root: &Path) {
    let profile_flat = read_i16("continuous_stream", "profile_hardware");
    if profile_flat.is_empty() {
        return;
    }
    let profile: Vec<[i16; 2]> = profile_flat.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
    let radians = read_f64("continuous_stream", "radians_per_count")[0];
    let transform = CountTransform::new(radians, true).unwrap();

    let mut fresh_motor = Samples::new("Continuous + Renderer fresh call, motor evaluated");
    let mut fresh_skipped = Samples::new("Continuous + Renderer fresh call, motor skipped");
    let mut buffered = Samples::new("Continuous + Renderer buffered 1 ms output");

    for seed in 7u64..10 {
        let options = PipelineOptions::new(transform)
            .seed(seed)
            .renderer_seed(101);
        let mut stream = ContinuousPipeline::load(&profile, options).unwrap();
        for tick in 1..=8_000i64 {
            if tick % 16 == 1 {
                stream
                    .update_target(
                        [
                            400.0 + 120.0 * (tick as f64 / 900.0).sin(),
                            -150.0 + 90.0 * (tick as f64 / 700.0).cos(),
                        ],
                        (tick - 1) * 1000,
                    )
                    .unwrap();
            }
            let before_fresh = stream.movement().decisions();
            let before_motor = stream.movement().motor_evaluations();
            let start = Instant::now();
            let block = stream.advance(tick * 1000).unwrap();
            let elapsed = start.elapsed().as_nanos();
            std::hint::black_box(&block);
            if stream.movement().decisions() == before_fresh {
                buffered.push(elapsed);
            } else if stream.movement().motor_evaluations() == before_motor {
                fresh_skipped.push(elapsed);
            } else {
                fresh_motor.push(elapsed);
            }
        }
    }
    let _ = root;
    fresh_motor.report();
    fresh_skipped.report();
    buffered.report();
}

fn static_planner(root: &Path) {
    let prefixes = read_f32("static_planner", "prefixes");
    let lengths = read_i64("static_planner", "prefix_lengths");
    let targets = read_f64("static_planner", "targets");
    let radii = read_f64("static_planner", "radii");
    let progresses = read_f64("static_planner", "progresses");
    let profiles = read_i16("static_pipeline", "profiles");
    if prefixes.is_empty() || profiles.is_empty() {
        return;
    }

    let mut plan = Samples::new("Static seed 7: fresh sampled-head plan");
    let mut prepare = Samples::new("Static renderer profile preparation");
    let mut step = Samples::new("Static renderer 1 ms report");
    let mut complete = Samples::new("Static complete handoff through last report");

    let mut pipeline = StaticPipeline::from_pretrained(7, Some(root), true).unwrap();
    let profile_window: Vec<[i16; 2]> = profiles[..512]
        .chunks_exact(2)
        .map(|p| [p[0], p[1]])
        .collect();
    for _ in 0..20 {
        let start = Instant::now();
        let profile = pipeline.prepare_renderer_profile(&profile_window).unwrap();
        prepare.push(start.elapsed().as_nanos());
        std::hint::black_box(&profile);
    }
    let profile = pipeline.prepare_renderer_profile(&profile_window).unwrap();

    let mut cursor = 0usize;
    for index in 0..lengths.len() {
        let length = lengths[index] as usize;
        let prefix: Vec<[f32; 2]> = prefixes[cursor..cursor + length * 2]
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect();
        cursor += length * 2;
        let target = [targets[index * 2], targets[index * 2 + 1]];
        for round in 0..5u64 {
            for head in 0..16 {
                let start = Instant::now();
                let intent = pipeline
                    .planner_mut()
                    .plan(
                        &prefix,
                        target,
                        radii[index],
                        progresses[index],
                        0,
                        Some(head),
                    )
                    .unwrap();
                plan.push(start.elapsed().as_nanos());

                let start = Instant::now();
                let mut stream = pipeline.begin(&profile, intent, round).unwrap();
                while !stream.complete() {
                    let inner = Instant::now();
                    let report = stream.step(pipeline.model()).unwrap();
                    step.push(inner.elapsed().as_nanos());
                    std::hint::black_box(report);
                }
                complete.push(start.elapsed().as_nanos());
            }
        }
    }
    plan.report();
    prepare.report();
    complete.report();
    step.report();
}

fn renderer(root: &Path) {
    let profile_flat = read_i16("continuous_stream", "profile_hardware");
    if profile_flat.is_empty() {
        return;
    }
    let profile_reports: Vec<[i16; 2]> =
        profile_flat.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
    let model = RendererModel::open(root.join("renderer_global_h80.bin")).unwrap();
    let profile = RendererProfile::prepare(&model, &profile_reports).unwrap();
    let mut stream = profile.begin_stream(101).unwrap();
    let mut samples = Samples::new("Renderer 1 ms report");
    for tick in 0..200_000u32 {
        let phase = f64::from(tick) / 311.0;
        let displacement = [(1.7 * phase.sin()) as f32, (1.1 * phase.cos()) as f32];
        let start = Instant::now();
        let report = stream.step(&model, displacement).unwrap();
        samples.push(start.elapsed().as_nanos());
        std::hint::black_box(report);
    }
    samples.report();
}

fn main() {
    let root = models_root();
    if !root.join("manifest.json").is_file() {
        eprintln!("skipping benchmarks: {} is absent", root.display());
        return;
    }
    println!(
        "| {:<48} | {:>7} | {:>10} | {:>10} | {:>10} |",
        "Public call", "Samples", "Median us", "p95 us", "Max us"
    );
    println!(
        "|{}|{}|{}|{}|{}|",
        "-".repeat(50),
        "-".repeat(9),
        "-".repeat(12),
        "-".repeat(12),
        "-".repeat(12)
    );
    continuous(&root);
    composed(&root);
    static_planner(&root);
    renderer(&root);
}
