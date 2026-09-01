use std::path::Path;

use super::pack::{install_component, safe_archive_path, safe_relative};

#[test]
fn pack_paths_stay_relative() {
    assert!(safe_relative(Path::new("models/image.onnx")));
    assert!(safe_relative(Path::new("./models/image.onnx")));
    assert!(!safe_relative(Path::new("../image.onnx")));
    assert!(!safe_relative(Path::new("/tmp/image.onnx")));
    assert!(safe_archive_path(Path::new("model/semantic-pack.json")));
    assert!(!safe_archive_path(Path::new("model/../../escape")));
}

#[test]
fn install_names_versioned() {
    let first = install_component("facebook/PE-Core-L14-336", "2026-08-26");
    let second = install_component("facebook/PE-Core-L14-336", "2026-08-27");
    assert!(first.starts_with("facebook-pe-core-l14-336-2026-08-26-"));
    assert_ne!(first, second);
    assert!(first.chars().all(|character| character.is_ascii_lowercase()
        || character.is_ascii_digit()
        || character == '-'));
}
