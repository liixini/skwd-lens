use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use super::catalog::{
    PackRole, REMOVAL_MARKER, RemovalTransaction, STATE_FILE, cleanup_transaction_files,
    load_catalog, load_state, save_state,
};
use super::lifecycle::{
    ImportMode, Operation, activate_candidate, activate_candidate_with, active_manifest,
    detected_runtime_version, doctor, failure_report, parse_command, remove, remove_with, repair,
    rollback,
};
use super::validate_pack_metadata;

fn write_pack(root: &Path, id: &str, version: &str, marker: &str) {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("image.onnx"), format!("image-{marker}")).unwrap();
    fs::write(root.join("text.onnx"), format!("text-{marker}")).unwrap();
    fs::write(root.join("tokenizer.json"), "{}").unwrap();
    fs::write(
        root.join("semantic-pack.json"),
        serde_json::to_vec_pretty(&json!({
            "format": 1,
            "id": id,
            "version": version,
            "dimensions": 4,
            "contextLength": 4,
            "image": {
                "model": "image.onnx",
                "width": 1,
                "height": 1,
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0]
            },
            "text": {
                "model": "text.onnx",
                "tokenizer": "tokenizer.json"
            }
        }))
        .unwrap(),
    )
    .unwrap();
}

fn candidate(parent: &Path, name: &str, version: &str, marker: &str) -> PathBuf {
    let path = parent.join(name);
    write_pack(&path, "example/model", version, marker);
    path
}

fn activate(models: &Path, candidate: &Path, mode: ImportMode) -> Value {
    let metadata = validate_pack_metadata(candidate).unwrap();
    let report = activate_candidate(
        models,
        candidate,
        &metadata,
        Path::new("libonnxruntime.so.1.27.0"),
        mode,
    )
    .unwrap();
    serde_json::to_value(report).unwrap()
}

#[test]
fn update_replace_and_rollback_retain_prior_components() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let sources = directory.path().join("sources");
    fs::create_dir_all(&models).unwrap();
    fs::create_dir_all(&sources).unwrap();

    let first = activate(&models, &candidate(&sources, "v1", "1", "first"), ImportMode::Install);
    assert_eq!(first["operation"], "install");
    assert!(first.get("previous").is_none());

    let update = activate(&models, &candidate(&sources, "v2", "2", "second"), ImportMode::Update);
    assert_eq!(update["operation"], "update");
    assert_eq!(update["active"]["version"], "2");
    assert_eq!(update["previous"]["version"], "1");

    let rollback = serde_json::to_value(rollback(&models, "example/model").unwrap()).unwrap();
    assert_eq!(rollback["active"]["version"], "1");
    assert_eq!(rollback["previous"]["version"], "2");

    let replacement =
        activate(&models, &candidate(&sources, "v1-rebuilt", "1", "rebuilt"), ImportMode::Replace);
    assert_eq!(replacement["operation"], "replace");
    assert_eq!(replacement["active"]["version"], "1");
    assert_ne!(replacement["active"]["component"], replacement["previous"]["component"]);

    let catalog = load_catalog(&models).unwrap();
    assert_eq!(catalog.installed.len(), 3);
    assert_eq!(catalog.installed.iter().filter(|pack| pack.role == PackRole::Active).count(), 1);
    assert_eq!(catalog.installed.iter().filter(|pack| pack.role == PackRole::Previous).count(), 1);
    assert_eq!(catalog.installed.iter().filter(|pack| pack.role == PackRole::Inactive).count(), 1);
}

#[test]
fn update_and_replace_modes_reject_wrong_version_relationships() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let sources = directory.path().join("sources");
    fs::create_dir_all(&models).unwrap();
    fs::create_dir_all(&sources).unwrap();

    let missing = candidate(&sources, "missing", "1", "missing");
    let metadata = validate_pack_metadata(&missing).unwrap();
    let error = activate_candidate(
        &models,
        &missing,
        &metadata,
        Path::new("runtime.so"),
        ImportMode::Update,
    )
    .unwrap_err();
    assert!(error.to_string().contains("no active installation"));
    assert!(missing.exists());

    activate(&models, &candidate(&sources, "v1", "1", "first"), ImportMode::Install);
    let same = candidate(&sources, "same", "1", "same");
    let metadata = validate_pack_metadata(&same).unwrap();
    assert!(
        activate_candidate(&models, &same, &metadata, Path::new("runtime.so"), ImportMode::Update,)
            .unwrap_err()
            .to_string()
            .contains("already active")
    );
    assert!(same.exists());

    let different = candidate(&sources, "different", "2", "different");
    let metadata = validate_pack_metadata(&different).unwrap();
    assert!(
        activate_candidate(
            &models,
            &different,
            &metadata,
            Path::new("runtime.so"),
            ImportMode::Replace,
        )
        .unwrap_err()
        .to_string()
        .contains("does not match active version")
    );
    assert!(different.exists());
}

#[test]
fn failed_state_publication_leaves_new_pack_inactive_and_prior_active() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let sources = directory.path().join("sources");
    fs::create_dir_all(&models).unwrap();
    fs::create_dir_all(&sources).unwrap();
    activate(&models, &candidate(&sources, "v1", "1", "first"), ImportMode::Install);
    let next = candidate(&sources, "v2", "2", "second");
    let metadata = validate_pack_metadata(&next).unwrap();

    let error = activate_candidate_with(
        &models,
        &next,
        &metadata,
        Path::new("runtime.so"),
        ImportMode::Update,
        |_, _| anyhow::bail!("simulated interruption"),
    )
    .unwrap_err();

    assert!(error.to_string().contains("retained inactive"));
    let state = load_state(&models).unwrap();
    assert_eq!(state.packs["example/model"].active.version, "1");
    let catalog = load_catalog(&models).unwrap();
    let orphan = catalog.installed.iter().find(|pack| pack.version == "2").unwrap();
    assert_eq!(orphan.role, PackRole::Inactive);
    assert!(orphan.healthy);
}

#[test]
fn post_publication_error_is_committed_even_when_written_by_is_upgraded() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let sources = directory.path().join("sources");
    fs::create_dir_all(&models).unwrap();
    fs::create_dir_all(&sources).unwrap();
    activate(&models, &candidate(&sources, "v1", "1", "first"), ImportMode::Install);
    let mut old_state = load_state(&models).unwrap();
    old_state.written_by = String::from("0.0.0-old");
    fs::write(models.join(STATE_FILE), serde_json::to_vec_pretty(&old_state).unwrap()).unwrap();
    let next = candidate(&sources, "v2", "2", "second");
    let metadata = validate_pack_metadata(&next).unwrap();

    let report = activate_candidate_with(
        &models,
        &next,
        &metadata,
        Path::new("runtime.so"),
        ImportMode::Update,
        |models, state| {
            save_state(models, state)?;
            anyhow::bail!("simulated directory sync error after publication")
        },
    )
    .unwrap();

    let report = serde_json::to_value(report).unwrap();
    assert_eq!(report["active"]["version"], "2");
    let visible = load_state(&models).unwrap();
    assert_eq!(visible.packs["example/model"].active.version, "2");
    assert_eq!(visible.written_by, env!("CARGO_PKG_VERSION"));
}

#[test]
fn safe_remove_refuses_active_then_removes_previous_and_orphan() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let sources = directory.path().join("sources");
    fs::create_dir_all(&models).unwrap();
    fs::create_dir_all(&sources).unwrap();
    activate(&models, &candidate(&sources, "v1", "1", "first"), ImportMode::Install);
    activate(&models, &candidate(&sources, "v2", "2", "second"), ImportMode::Update);
    let state = load_state(&models).unwrap();
    let active = state.packs["example/model"].active.clone();
    let previous = state.packs["example/model"].previous.clone().unwrap();

    let error = remove(&models, "example/model", None, Some(&active.component)).unwrap_err();
    assert!(error.to_string().contains("refusing to remove active"));
    assert!(models.join(&active.component).is_dir());

    let report = serde_json::to_value(
        remove(&models, "example/model", None, Some(&previous.component)).unwrap(),
    )
    .unwrap();
    assert_eq!(report["removed"], true);
    assert!(!models.join(&previous.component).exists());
    assert!(load_state(&models).unwrap().packs["example/model"].previous.is_none());

    let orphan = candidate(&sources, "orphan", "3", "orphan");
    fs::rename(&orphan, models.join("orphan-component")).unwrap();
    let report =
        serde_json::to_value(remove(&models, "example/model", Some("3"), None).unwrap()).unwrap();
    assert_eq!(report["component"], "orphan-component");
    assert!(!models.join("orphan-component").exists());
}

#[test]
fn file_digests_make_tampered_previous_pack_unhealthy_and_unrollbackable() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let sources = directory.path().join("sources");
    fs::create_dir_all(&models).unwrap();
    fs::create_dir_all(&sources).unwrap();
    activate(&models, &candidate(&sources, "v1", "1", "first"), ImportMode::Install);
    activate(&models, &candidate(&sources, "v2", "2", "second"), ImportMode::Update);
    let state = load_state(&models).unwrap();
    let previous = state.packs["example/model"].previous.as_ref().unwrap();
    assert!(!previous.files.is_empty());
    let model = models.join(&previous.component).join("image.onnx");
    let length = fs::metadata(&model).unwrap().len() as usize;
    fs::write(&model, vec![b'X'; length]).unwrap();

    let catalog = load_catalog(&models).unwrap();
    let installed =
        catalog.installed.iter().find(|pack| pack.component == previous.component).unwrap();
    assert!(!installed.healthy);
    assert!(installed.issues.iter().any(|issue| issue.code == "pack_integrity_changed"));
    assert!(rollback(&models, "example/model").unwrap_err().to_string().contains("invalid"));
}

#[test]
fn failed_remove_state_publication_restores_files_and_rollback_pointer() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let sources = directory.path().join("sources");
    fs::create_dir_all(&models).unwrap();
    fs::create_dir_all(&sources).unwrap();
    activate(&models, &candidate(&sources, "v1", "1", "first"), ImportMode::Install);
    activate(&models, &candidate(&sources, "v2", "2", "second"), ImportMode::Update);
    let before = load_state(&models).unwrap();
    let previous = before.packs["example/model"].previous.as_ref().unwrap().clone();

    let error = remove_with(&models, "example/model", None, Some(&previous.component), |_, _| {
        anyhow::bail!("simulated state failure")
    })
    .unwrap_err();

    assert!(format!("{error:#}").contains("simulated state failure"));
    assert_eq!(load_state(&models).unwrap(), before);
    assert!(models.join(previous.component).is_dir());
}

#[test]
fn interrupted_remove_restores_a_still_referenced_component() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let sources = directory.path().join("sources");
    fs::create_dir_all(&models).unwrap();
    fs::create_dir_all(&sources).unwrap();
    activate(&models, &candidate(&sources, "v1", "1", "first"), ImportMode::Install);
    let state = load_state(&models).unwrap();
    let component = state.packs["example/model"].active.component.clone();
    let transaction = models.join(".skwd-model-import-interrupted.tmp");
    fs::create_dir(&transaction).unwrap();
    fs::write(
        transaction.join(REMOVAL_MARKER),
        serde_json::to_vec(&RemovalTransaction { component: component.clone() }).unwrap(),
    )
    .unwrap();
    fs::rename(models.join(&component), transaction.join("removed")).unwrap();

    let manifest = active_manifest(&models, Some("example/model")).unwrap();

    assert_eq!(manifest, models.join(&component).join("semantic-pack.json"));
    assert!(models.join(component).is_dir());
    assert!(!transaction.exists());
}

#[test]
fn corrupt_state_repair_backs_up_and_adopts_hashed_unambiguous_packs() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let sources = directory.path().join("sources");
    fs::create_dir_all(&models).unwrap();
    fs::create_dir_all(&sources).unwrap();
    activate(&models, &candidate(&sources, "v1", "1", "first"), ImportMode::Install);
    fs::write(models.join(STATE_FILE), "{").unwrap();

    let report = serde_json::to_value(repair(&models, None).unwrap()).unwrap();

    assert_eq!(report["operation"], "repair");
    assert!(Path::new(report["backup"].as_str().unwrap()).is_file());
    assert_eq!(report["adopted"].as_array().unwrap().len(), 1);
    assert!(report["unresolved"].as_array().unwrap().is_empty());
    let state = load_state(&models).unwrap();
    assert!(!state.packs["example/model"].active.files.is_empty());
    assert_eq!(
        active_manifest(&models, None).unwrap(),
        models
            .canonicalize()
            .unwrap()
            .join(&state.packs["example/model"].active.component)
            .join("semantic-pack.json")
    );
}

#[test]
fn corrupt_state_repair_recovers_an_interrupted_retained_pack_removal() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let sources = directory.path().join("sources");
    fs::create_dir_all(&models).unwrap();
    fs::create_dir_all(&sources).unwrap();
    activate(&models, &candidate(&sources, "v1", "1", "first"), ImportMode::Install);
    let component = load_state(&models).unwrap().packs["example/model"].active.component.clone();
    let transaction = models.join(".skwd-model-import-corrupt-remove.tmp");
    fs::create_dir(&transaction).unwrap();
    fs::write(
        transaction.join(REMOVAL_MARKER),
        serde_json::to_vec(&RemovalTransaction { component: component.clone() }).unwrap(),
    )
    .unwrap();
    fs::rename(models.join(&component), transaction.join("removed")).unwrap();
    fs::write(models.join(STATE_FILE), b"{ definitely corrupt").unwrap();

    let report = serde_json::to_value(repair(&models, None).unwrap()).unwrap();

    assert_eq!(report["adopted"][0]["component"], component);
    assert!(models.join(&component).is_dir());
    assert!(!transaction.exists());
    assert_eq!(
        active_manifest(&models, Some("example/model")).unwrap(),
        models.join(component).join("semantic-pack.json")
    );
}

#[test]
fn transaction_cleanup_is_scoped_to_known_temporary_names() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path();
    fs::create_dir(models.join(".skwd-model-import-10-1.tmp")).unwrap();
    fs::write(models.join(".skwd-model-packs-state-10-2.tmp"), "partial").unwrap();
    fs::write(models.join("keep.tmp"), "user data").unwrap();

    let removed = cleanup_transaction_files(models).unwrap();

    assert_eq!(removed.len(), 2);
    assert!(models.join("keep.tmp").is_file());
}

#[test]
fn doctor_reports_state_runtime_and_index_failures_structurally() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let sources = directory.path().join("sources");
    fs::create_dir_all(&models).unwrap();
    fs::create_dir_all(&sources).unwrap();
    activate(&models, &candidate(&sources, "v1", "1", "first"), ImportMode::Install);
    let mut state = load_state(&models).unwrap();
    state.packs.get_mut("example/model").unwrap().active.runtime_version =
        Some(String::from("1.26.0"));
    save_state(&models, &state).unwrap();

    let report = serde_json::to_value(
        doctor(
            &models,
            "example/model",
            Path::new("/missing/libonnxruntime.so.1.27.0"),
            Path::new("/missing/index.sidx"),
            1,
        )
        .unwrap(),
    )
    .unwrap();

    assert_eq!(report["ok"], false);
    let codes = report["issues"]
        .as_array()
        .unwrap()
        .iter()
        .map(|issue| issue["code"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert!(codes.contains(&"runtime_incompatible"));
    assert!(codes.contains(&"runtime_version_changed"));
    assert!(codes.contains(&"index_incompatible"));
    assert_eq!(report["checks"][0]["subject"], "app");
    assert_eq!(report["checks"][1]["subject"], "model");
    assert_eq!(report["checks"][2]["subject"], "runtime");
    assert_eq!(report["checks"][3]["subject"], "index");

    fs::write(models.join(STATE_FILE), "{").unwrap();
    let report = serde_json::to_value(
        doctor(
            &models,
            "example/model",
            Path::new("/missing/runtime.so"),
            Path::new("/missing/index.sidx"),
            1,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        report["issues"].as_array().unwrap().iter().any(|issue| issue["code"] == "state_invalid")
    );
}

#[test]
fn lifecycle_parser_is_explicit_and_failures_have_codes() {
    let list = vec![
        "skwd-lens".into(),
        "--list-packs".into(),
        "--models-dir".into(),
        "/tmp/models".into(),
    ];
    assert!(parse_command(&list).unwrap().is_some());

    let conflict = vec![
        "skwd-lens".into(),
        "--list-packs".into(),
        "--rollback-pack".into(),
        "model".into(),
        "--models-dir".into(),
        "/tmp/models".into(),
    ];
    let error = parse_command(&conflict).unwrap_err();
    let report = serde_json::to_value(failure_report(Operation::List, &error)).unwrap();
    assert_eq!(report["error"]["code"], "conflicting_operations");
    assert!(!report["error"]["hint"].as_str().unwrap().is_empty());

    let remove = vec![
        "skwd-lens".into(),
        "--remove-pack".into(),
        "model".into(),
        "--models-dir".into(),
        "/tmp/models".into(),
    ];
    assert!(parse_command(&remove).unwrap_err().to_string().contains("exactly one"));

    for invalid in [
        vec![
            "skwd-lens".into(),
            "--list-packs".into(),
            "--models-dir".into(),
            "--threads".into(),
            "2".into(),
        ],
        vec![
            "skwd-lens".into(),
            "--list-packs".into(),
            "--models-dir".into(),
            "/tmp/one".into(),
            "--models-dir".into(),
            "/tmp/two".into(),
        ],
        vec![
            "skwd-lens".into(),
            "--list-packs".into(),
            "--models-dir".into(),
            "/tmp/models".into(),
            "--typo".into(),
        ],
    ] {
        assert!(parse_command(&invalid).is_err());
    }
}

#[test]
fn runtime_version_comes_only_from_versioned_library_name() {
    assert_eq!(
        detected_runtime_version(Path::new("/opt/runtime/libonnxruntime.so.1.27.0")).as_deref(),
        Some("1.27.0")
    );
    assert!(detected_runtime_version(Path::new("/opt/runtime/libonnxruntime.so")).is_none());
}

#[test]
fn shipped_pack_lock_and_manifest_are_consistent() {
    let lock: Value =
        serde_json::from_str(include_str!("../../../../packaging/default-semantic.lock.json"))
            .unwrap();
    let manifest: Value = serde_json::from_str(include_str!(
        "../../../../scripts/tagger/model-packs/maximum/semantic-pack.json"
    ))
    .unwrap();
    let package_manifest = include_str!("../../../../packaging/default-manifest.txt");
    let total = lock["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|file| file["size"].as_u64().unwrap())
        .sum::<u64>();
    let semantic = lock["files"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|file| file["source"] != "runtime")
        .map(|file| file["size"].as_u64().unwrap())
        .sum::<u64>();

    assert_eq!(total, 598_469_965);
    assert_eq!(semantic, 574_718_997);
    assert_eq!(lock["runtimeVersion"], "1.27.0");
    assert_eq!(
        lock["product"],
        "siglip2-base-p16-224@google-image-int8-attention-text-int8-stretch-v4"
    );
    assert_eq!(manifest["dimensions"], 768);
    assert_eq!(manifest["contextLength"], 64);
    assert_eq!(manifest["image"]["resizeMode"], "stretch");
    assert_eq!(manifest["text"]["tokenEmbeddings"]["rows"], 256_000);
    for license in ["Apache-2.0.txt", "CC-BY-4.0.txt", "MIT.txt"] {
        assert!(package_manifest.contains(license));
    }
}
