use crate::error::{Error, Result};
use crate::io::json::{self, Json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const RELEASE_MANIFEST_SCHEMA: &str = "abcurves.release_models.v2";
pub const CONTINUOUS_MANIFEST_SCHEMA: &str = "phalm.b223.assets.v1";
pub const FROZEN_BRAKE_AND_HAZARDS: &str =
    "4c71cc4d29b4a13e613afa88438f772c478fd542c1c9095d23447ecb3c4a3afa";

// Code-level release anchors, deliberately independent of the adjacent JSON
// manifest.  A modified model plus a modified manifest must not turn
// verification into self-attestation.
pub const RELEASE_FILE_ANCHORS: [(&str, u64, &str); 4] = [
    (
        "planner_seed7.pt",
        1_500_345,
        "d82c93071224f7eb225d1f2bcf46d52669a7270db414431d7622e032439b280d",
    ),
    (
        "planner_seed23.pt",
        1_500_389,
        "d691ba155c4fa9b403c5a3e2ed9c44123fe00d3d1bee15c55ee9226f4531a23e",
    ),
    (
        "renderer_global_h80.bin",
        44_484,
        "405c34bceb55485dfd6bd3c0368bce079feea680a64c6b261904ef5b4713e240",
    ),
    (
        "renderer_global_h80_float.pt",
        144_457,
        "dd10ebe7d08011d7dbd91736e54a84faec4844f69a911a9dfd57eb83cd8b5e5c",
    ),
];

pub const FROZEN_CONTINUOUS_FILES: [(&str, &str); 7] = [
    ("brake.onnx", "ec511415341b72284498ae0333cb12da6bf26e1f9c06562bc468766bfa636c24"),
    ("choice.onnx", "805f517a27219bc20fd0418be02cbf5316a601f040533056fe61304ef5e2ac74"),
    ("choice_split.onnx", "7a8d36ab01e719423854051a6b1a4f35536e5688e2eaf3e9e154f32e5e174bf1"),
    ("events.onnx", "cd839db370bf0804550cc42e63667736ddd2c2f8fa05cd46453c7254c28dabe3"),
    ("hazard.onnx", "ebd385e53ca15bb5ec5609018bdd93f2213875a41b3c78912986b00dfda12e53"),
    ("motor.onnx", "116126ef8a6dc7aae027de596158d51e44a06fa30dcfd7058dfcde463106542c"),
    ("weights.npz", "018a096c66213c62b429caf4a8ae6f0470f6e283248a24f476f0fed3e36ea881"),
];

pub fn sha256(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut stream = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut block = vec![0u8; 1 << 20];
    loop {
        let read = stream.read(&mut block)?;
        if read == 0 {
            break;
        }
        digest.update(&block[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn default_model_dir() -> PathBuf {
    if let Ok(explicit) = std::env::var("ABCURVES_MODEL_DIR") {
        return PathBuf::from(explicit);
    }
    let candidates = [
        PathBuf::from("models"),
        PathBuf::from("origin").join("ABCurves").join("models"),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("models"),
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("origin")
            .join("ABCurves")
            .join("models"),
    ];
    for candidate in &candidates {
        if candidate.join("manifest.json").is_file() {
            return candidate.clone();
        }
    }
    candidates[0].clone()
}

#[derive(Clone, Debug)]
pub struct ModelFiles {
    pub seed: u32,
    pub planner: PathBuf,
    pub renderer: PathBuf,
    pub manifest: PathBuf,
}

fn load_manifest(directory: &Path, schema: &str) -> Result<Json> {
    let path = directory.join("manifest.json");
    if !path.is_file() {
        return Err(Error::ModelIntegrity(format!(
            "model manifest is missing: {}",
            path.display()
        )));
    }
    let manifest = json::read(&path)?;
    if manifest.get("schema").and_then(|v| v.as_str().ok()) != Some(schema) {
        return Err(Error::ModelIntegrity(format!(
            "unsupported model manifest schema in {}",
            path.display()
        )));
    }
    Ok(manifest)
}

fn verified_file(directory: &Path, name: &str, manifest: &Json) -> Result<PathBuf> {
    let record = manifest
        .get("files")
        .and_then(|files| files.get(name))
        .ok_or_else(|| {
            Error::ModelIntegrity(format!("{name:?} is not declared by the model manifest"))
        })?;
    let (_, expected_bytes, expected) = RELEASE_FILE_ANCHORS
        .iter()
        .find(|(anchor, _, _)| *anchor == name)
        .ok_or_else(|| Error::ModelIntegrity(format!("{name:?} has no immutable release anchor")))?;
    let path = directory.join(name);
    if !path.is_file() {
        return Err(Error::ModelIntegrity(format!(
            "release model is missing: {}",
            path.display()
        )));
    }
    let declared_bytes = record.get("bytes").and_then(|v| v.as_u64().ok());
    let declared_digest = record
        .get("sha256")
        .and_then(|v| v.as_str().ok())
        .map(str::to_ascii_lowercase);
    if declared_bytes != Some(*expected_bytes) || declared_digest.as_deref() != Some(*expected) {
        return Err(Error::ModelIntegrity(format!(
            "manifest declaration differs from the release anchor for {name}"
        )));
    }
    let observed_bytes = std::fs::metadata(&path)?.len();
    if observed_bytes != *expected_bytes {
        return Err(Error::ModelIntegrity(format!(
            "release model size differs for {name}: expected {expected_bytes}, observed {observed_bytes}"
        )));
    }
    let observed = sha256(&path)?;
    if observed != *expected {
        return Err(Error::ModelIntegrity(format!(
            "release model hash differs for {name}: expected {expected}, observed {observed}"
        )));
    }
    Ok(path)
}

pub fn resolve_model_files(
    seed: u32,
    model_dir: Option<&Path>,
    verify: bool,
) -> Result<ModelFiles> {
    let root = model_dir.map_or_else(default_model_dir, Path::to_path_buf);
    let manifest = load_manifest(&root, RELEASE_MANIFEST_SCHEMA)?;
    let seeds = manifest.entry("seeds")?.as_array()?;
    if !seeds.iter().any(|value| value.as_u64().is_ok_and(|found| found == u64::from(seed))) {
        return Err(Error::ModelIntegrity(format!(
            "unsupported release seed {seed}"
        )));
    }
    let planner_name = format!("planner_seed{seed}.pt");
    let (planner, renderer) = if verify {
        (
            verified_file(&root, &planner_name, &manifest)?,
            verified_file(&root, "renderer_global_h80.bin", &manifest)?,
        )
    } else {
        (
            root.join(&planner_name),
            root.join("renderer_global_h80.bin"),
        )
    };
    Ok(ModelFiles {
        seed,
        planner,
        renderer,
        manifest: root.join("manifest.json"),
    })
}

pub fn verify_continuous_assets(directory: &Path, allow_custom: bool) -> Result<Json> {
    let manifest = load_manifest(directory, CONTINUOUS_MANIFEST_SCHEMA)?;
    let files = manifest.entry("files")?.as_object()?;
    for required in ["weights.npz", "motor.onnx", "choice.onnx", "events.onnx"] {
        if !files.contains_key(required) {
            return Err(Error::ModelIntegrity(
                "Asset manifest omits a required file digest".into(),
            ));
        }
    }
    if !allow_custom {
        let brake = manifest
            .entry("reference")?
            .entry("active_dependencies")?
            .entry("brake_and_hazards")?
            .entry("sha256")?
            .as_str()?;
        if brake != FROZEN_BRAKE_AND_HAZARDS {
            return Err(Error::ModelIntegrity("Bundle is not the frozen B2-23".into()));
        }
        let frozen_matches = files.len() == FROZEN_CONTINUOUS_FILES.len()
            && FROZEN_CONTINUOUS_FILES.iter().all(|(name, digest)| {
                files.get(*name).and_then(|v| v.as_str().ok()) == Some(*digest)
            });
        if !frozen_matches {
            return Err(Error::ModelIntegrity(
                "Asset bundle differs from the selected release; custom training requires explicit opt-in".into(),
            ));
        }
    }
    for (name, declared) in files {
        let digest = declared.as_str()?;
        let path = directory.join(name);
        if sha256(&path)? != digest {
            return Err(Error::ModelIntegrity(format!("Asset digest mismatch: {name}")));
        }
    }
    Ok(manifest)
}

pub fn default_continuous_dir() -> PathBuf {
    default_model_dir().join("continuous")
}
