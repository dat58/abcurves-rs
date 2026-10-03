#![allow(clippy::needless_range_loop)]

mod common;

use abcurves::renderer::RendererModel;
use abcurves::renderer::model::{self, ADAPTER_BYTES, ARTIFACT_BYTES, FIXED_BYTES};
use common::models_root;

fn artifact() -> Option<Vec<u8>> {
    let path = models_root().join("renderer_global_h80.bin");
    if !path.is_file() {
        eprintln!("skipping: {} is absent", path.display());
        return None;
    }
    Some(std::fs::read(path).unwrap())
}

#[test]
fn release_artifact_parses_and_verifies() {
    let Some(blob) = artifact() else { return };
    assert_eq!(blob.len(), ARTIFACT_BYTES);
    let parsed = RendererModel::parse(&blob).unwrap();
    assert_eq!(parsed.fixed.body_crc32, model::EXPECTED_BODY_CRC32);
    assert_eq!(parsed.fixed.weights.len(), 33_760);
    assert_eq!(parsed.fixed.biases.len(), 602);
    assert_eq!(parsed.fixed.multipliers.len(), 128);
    assert_eq!(parsed.fixed.sigmoid.len(), model::LUT_POINTS);
    assert_eq!(parsed.fixed.tanh.len(), model::LUT_POINTS);
    assert_eq!(parsed.fixed.exp.len(), model::LUT_POINTS);
    assert_eq!(parsed.fixed.log.len(), model::LUT_POINTS);
    assert_eq!(
        parsed.fixed.config,
        [
            24576, 50412, 87381, 436907, 32768, 2097152, 32767, 0, 32768, 98304, 32768
        ]
    );
    assert_eq!(parsed.adapter.mean.len(), 145);
    assert_eq!(parsed.adapter.v_q.len(), 16 * 145);
    assert_eq!(parsed.adapter.u_q.len(), 80 * 16);
    assert_eq!(
        model::crc32(&blob[FIXED_BYTES..]),
        model::EXPECTED_ADAPTER_CRC32
    );
    assert_eq!(blob.len() - FIXED_BYTES, ADAPTER_BYTES);
    RendererModel::from_release(Some(&models_root())).unwrap();
}

#[test]
fn tampered_artifacts_are_rejected() {
    let Some(blob) = artifact() else { return };
    assert!(RendererModel::parse(&blob[..blob.len() - 1]).is_err());

    let mut body = blob.clone();
    body[300] ^= 0x01;
    assert!(RendererModel::parse(&body).is_err());

    let mut adapter = blob.clone();
    adapter[FIXED_BYTES + 100] ^= 0x01;
    assert!(RendererModel::parse(&adapter).is_err());

    let mut header = blob.clone();
    header[12] = 81;
    assert!(RendererModel::parse(&header).is_err());

    let mut padding = blob.clone();
    padding[240] = 1;
    assert!(RendererModel::parse(&padding).is_err());
}
