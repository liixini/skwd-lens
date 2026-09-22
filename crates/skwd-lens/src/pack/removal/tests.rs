use super::*;
use serde_json::json;

fn model(root: &Path, name: &str) -> PathBuf {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("image.onnx"), [8, 9]).unwrap();
    fs::write(root.join("text.onnx"), [8, 9]).unwrap();
    fs::write(root.join("tokenizer.json"), "{}").unwrap();
    let path = root.join(name);
    fs::write(
        &path,
        serde_json::to_vec(&json!({
            "format": 1, "id": "test", "version": "v1", "dimensions": 4, "contextLength": 4,
            "image": {"model": "image.onnx", "width": 1, "height": 1,
                "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
            "text": {"model": "text.onnx", "tokenizer": "tokenizer.json"}
        }))
        .unwrap(),
    )
    .unwrap();
    path
}

fn field(number: u8, bytes: &[u8]) -> Vec<u8> {
    let mut data = vec![(number << 3) | 2, bytes.len().try_into().unwrap()];
    data.extend(bytes);
    data
}

fn external_graph(location: &str) -> Vec<u8> {
    let mut entry = field(1, b"location");
    entry.extend(field(2, location.as_bytes()));
    field(7, &field(5, &field(13, &entry)))
}

#[test]
fn default_removal_preserves_runtime_tags_imported_models_and_unrelated_files() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = model(dir.path(), "semantic-pack.json");
    for keep in [
        "runtime/libonnxruntime.so",
        "autotag-pack.json",
        "tags.db",
        "packs/other/semantic-pack.json",
        "notes.txt",
    ] {
        let path = dir.path().join(keep);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "keep").unwrap();
    }
    fs::write(dir.path().join("image.onnx"), external_graph("weights.bin")).unwrap();
    fs::write(dir.path().join("weights.bin"), "weights").unwrap();
    assert!(remove(&manifest).unwrap().removed);
    for deleted in
        ["semantic-pack.json", "image.onnx", "text.onnx", "tokenizer.json", "weights.bin"]
    {
        assert!(!dir.path().join(deleted).exists(), "{deleted}");
    }
    for keep in [
        "runtime/libonnxruntime.so",
        "autotag-pack.json",
        "tags.db",
        "packs/other/semantic-pack.json",
        "notes.txt",
    ] {
        assert_eq!(fs::read_to_string(dir.path().join(keep)).unwrap(), "keep");
    }
}

#[test]
fn manual_manifest_removal_preserves_files_shared_by_another_model() {
    let dir = tempfile::tempdir().unwrap();
    let first = model(dir.path(), "manual.json");
    let second = model(dir.path(), "semantic-pack.json");
    assert!(remove(&first).unwrap().removed);
    assert!(!first.exists());
    assert!(second.is_file());
    assert!(dir.path().join("image.onnx").is_file());
    assert!(remove(&second).unwrap().removed);
    assert!(!dir.path().join("image.onnx").exists());
}

#[test]
fn external_tensor_traversal_fails_before_removing_any_model_files() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = model(&dir.path().join("model"), "semantic-pack.json");
    fs::write(dir.path().join("outside"), "keep").unwrap();
    fs::write(manifest.parent().unwrap().join("image.onnx"), external_graph("../outside")).unwrap();
    assert!(remove(&manifest).unwrap_err().to_string().contains("escapes"));
    assert!(manifest.is_file());
    assert!(manifest.parent().unwrap().join("text.onnx").is_file());
    assert_eq!(fs::read_to_string(dir.path().join("outside")).unwrap(), "keep");
}

#[cfg(unix)]
#[test]
fn symlinked_model_file_is_never_followed_or_removed() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = model(&dir.path().join("model"), "semantic-pack.json");
    let image = manifest.parent().unwrap().join("image.onnx");
    fs::remove_file(&image).unwrap();
    fs::write(dir.path().join("outside"), [8, 9]).unwrap();
    std::os::unix::fs::symlink(dir.path().join("outside"), &image).unwrap();
    assert!(remove(&manifest).is_err());
    assert!(manifest.is_file());
    assert!(image.is_symlink());
    assert!(dir.path().join("outside").is_file());
}

#[test]
fn malformed_onnx_fails_before_removal() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = model(dir.path(), "semantic-pack.json");
    fs::write(dir.path().join("image.onnx"), [58, 100, 0]).unwrap();
    assert!(remove(&manifest).is_err());
    assert!(manifest.is_file());
    assert!(dir.path().join("text.onnx").is_file());
}

#[test]
fn remove_manifest_is_exclusive_and_needs_no_runtime() {
    let args = ["skwd-lens", "--remove-manifest", "/model/semantic-pack.json"].map(String::from);
    assert!(matches!(
        super::super::lifecycle::parse_command(&args).unwrap(),
        Some(super::super::lifecycle::LifecycleCommand::RemoveManifest { .. })
    ));
    let mut conflicting = args.to_vec();
    conflicting.extend(["--remove-pack", "test"].map(String::from));
    assert!(super::super::lifecycle::parse_command(&conflicting).is_err());
}

#[test]
fn failed_file_removal_restores_already_staged_files() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = model(dir.path(), "semantic-pack.json");
    let mut staged = 0;
    let result = remove_with(&manifest, |source, target| {
        if staged == 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "injected rename failure",
            ));
        }
        fs::rename(source, target)?;
        staged += 1;
        Ok(())
    });
    assert_eq!(staged, 2);
    assert!(result.is_err());
    assert!(manifest.is_file());
    assert!(dir.path().join("image.onnx").is_file());
    assert!(dir.path().join("text.onnx").is_file());
    assert!(dir.path().join("tokenizer.json").is_file());
}
