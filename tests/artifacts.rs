mod common;

use abcurves::io::{Checkpoint, Npz};
use abcurves::model_store;
use common::{models_root, read_f64, read_i64, read_u8};
use sha2::{Digest, Sha256};

const FIXTURE: &str = "artifacts";

fn text_lines(field: &str) -> Vec<String> {
    String::from_utf8(read_u8(FIXTURE, field))
        .unwrap()
        .split('\n')
        .map(str::to_string)
        .collect()
}

fn check_digests(
    names: &[String],
    shapes: &[i64],
    digests: &[u8],
    mut lookup: impl FnMut(&str) -> (Vec<usize>, Vec<u8>),
) {
    let mut cursor = 0usize;
    for (index, name) in names.iter().enumerate() {
        let rank = shapes[cursor] as usize;
        let expected_shape: Vec<usize> =
            shapes[cursor + 1..cursor + 1 + rank].iter().map(|&d| d as usize).collect();
        cursor += 1 + rank;
        let (shape, bytes) = lookup(name);
        assert_eq!(shape, expected_shape, "shape of {name}");
        let observed = Sha256::digest(&bytes);
        assert_eq!(
            observed.as_slice(),
            &digests[index * 32..(index + 1) * 32],
            "contents of {name}"
        );
    }
}

#[test]
fn continuous_weights_load_exactly() {
    let path = models_root().join("continuous").join("weights.npz");
    if !path.is_file() {
        eprintln!("skipping: {} is absent", path.display());
        return;
    }
    let bundle = Npz::open(&path).unwrap();
    let names = text_lines("continuous_names");
    assert_eq!(names.len(), 43);
    check_digests(
        &names,
        &read_i64(FIXTURE, "continuous_shapes"),
        &read_u8(FIXTURE, "continuous_digests"),
        |name| {
            let array = bundle.array(name).unwrap();
            (array.shape.clone(), array.bytes.clone())
        },
    );
}

#[test]
fn planner_checkpoint_loads_exactly() {
    let path = models_root().join("planner_seed7.pt");
    if !path.is_file() {
        eprintln!("skipping: {} is absent", path.display());
        return;
    }
    let checkpoint = Checkpoint::open(&path).unwrap();
    let state = checkpoint.root().entry("model_state_dict").unwrap();
    let names = text_lines("planner_names");
    assert_eq!(names.len(), 38);
    check_digests(
        &names,
        &read_i64(FIXTURE, "planner_shapes"),
        &read_u8(FIXTURE, "planner_digests"),
        |name| {
            let value = state.get(name).unwrap_or_else(|| checkpoint.root().entry(name).unwrap());
            let array = checkpoint.tensor(value.as_tensor().unwrap()).unwrap();
            (array.shape.clone(), array.bytes.clone())
        },
    );

    let meta = read_i64(FIXTURE, "meta");
    let root = checkpoint.root();
    assert_eq!(root.entry("heads").unwrap().as_i64().unwrap(), meta[0]);
    assert_eq!(root.entry("horizon").unwrap().as_i64().unwrap(), meta[1]);
    assert_eq!(root.entry("summary_dim").unwrap().as_i64().unwrap(), meta[2]);
    assert_eq!(root.entry("target_dim").unwrap().as_i64().unwrap(), meta[3]);
    let config = root.entry("planner_config").unwrap();
    assert_eq!(config.entry("prefix_len").unwrap().as_i64().unwrap(), meta[4]);
    let prodmp = root.entry("prodmp").unwrap();
    assert_eq!(prodmp.entry("n_basis").unwrap().as_i64().unwrap(), meta[5]);

    let scalars = read_f64(FIXTURE, "scalars");
    assert_eq!(prodmp.entry("alpha").unwrap().as_f64().unwrap(), scalars[0]);
    assert_eq!(prodmp.entry("alpha_phase").unwrap().as_f64().unwrap(), scalars[1]);
    assert_eq!(prodmp.entry("ridge").unwrap().as_f64().unwrap(), scalars[2]);
    let hinge = root.entry("hinge_thresholds").unwrap().as_sequence().unwrap();
    for (index, value) in hinge.iter().enumerate() {
        assert_eq!(value.as_f64().unwrap(), scalars[3 + index]);
    }

    let text = text_lines("text");
    let seam = root.entry("seam_contract").unwrap();
    assert_eq!(root.entry("schema").unwrap().as_str().unwrap(), text[0]);
    assert_eq!(root.entry("release_schema").unwrap().as_str().unwrap(), text[1]);
    assert_eq!(root.entry("release_status").unwrap().as_str().unwrap(), text[2]);
    assert_eq!(seam.entry("schema").unwrap().as_str().unwrap(), text[3]);
    assert_eq!(
        seam.entry("trigger").unwrap().entry("reference").unwrap().as_str().unwrap(),
        text[4]
    );
    assert_eq!(
        root.entry("prefix_representation").unwrap().entry("name").unwrap().as_str().unwrap(),
        text[5]
    );
    let summary = root.entry("summary_feature_names").unwrap().as_sequence().unwrap();
    assert_eq!(summary.len(), 62);
    for (index, value) in summary.iter().enumerate() {
        assert_eq!(value.as_str().unwrap(), text[6 + index]);
    }
}

#[test]
fn release_anchors_verify() {
    let root = models_root();
    if !root.join("manifest.json").is_file() {
        eprintln!("skipping: {} is absent", root.display());
        return;
    }
    for seed in [7u32, 23] {
        let files = model_store::resolve_model_files(seed, Some(&root), true).unwrap();
        assert!(files.planner.is_file());
        assert!(files.renderer.is_file());
    }
    assert!(model_store::resolve_model_files(11, Some(&root), true).is_err());
    model_store::verify_continuous_assets(&root.join("continuous"), false).unwrap();
}
