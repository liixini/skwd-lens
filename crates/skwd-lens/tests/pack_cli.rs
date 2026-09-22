use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};

fn write_pack(root: &Path) {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("image.onnx"), "image").unwrap();
    fs::write(root.join("text.onnx"), "text").unwrap();
    fs::write(root.join("tokenizer.json"), "{}").unwrap();
    fs::write(
        root.join("semantic-pack.json"),
        serde_json::to_vec_pretty(&json!({
            "format": 1,
            "id": "cli/model",
            "version": "1",
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

fn lens(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_skwd-lens")).args(arguments).output().unwrap()
}

fn output_json(output: &Output) -> Value {
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    serde_json::from_slice(&output.stdout).unwrap()
}

fn failure_json(output: &Output) -> Value {
    assert!(
        !output.status.success(),
        "unexpected success: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stderr).unwrap()
}

#[test]
fn lifecycle_cli_lists_status_doctors_rolls_back_and_removes() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    write_pack(&models.join("manual-component"));
    let models = models.to_str().unwrap();

    let listed = output_json(&lens(&["--list-packs", "--models-dir", models]));
    assert_eq!(listed["operation"], "list");
    assert_eq!(listed["packs"][0]["id"], "cli/model");
    assert_eq!(listed["packs"][0]["role"], "inactive");

    let status = output_json(&lens(&["--pack-status", "cli/model", "--models-dir", models]));
    assert_eq!(status["operation"], "status");
    assert!(status.get("active").is_none());
    assert_eq!(status["installed"][0]["component"], "manual-component");

    let doctor_output = lens(&[
        "--doctor-pack",
        "cli/model",
        "--models-dir",
        models,
        "--runtime",
        "/definitely/missing/libonnxruntime.so",
        "--index",
        "/definitely/missing/index.sidx",
    ]);
    assert!(!doctor_output.status.success());
    assert!(doctor_output.stderr.is_empty());
    let doctor: Value = serde_json::from_slice(&doctor_output.stdout).unwrap();
    assert_eq!(doctor["operation"], "doctor");
    assert_eq!(doctor["ok"], false);
    assert!(doctor["issues"].as_array().unwrap().iter().any(|issue| {
        issue["code"] == "active_pack_missing" || issue["code"] == "index_incompatible"
    }));

    let rollback = lens(&["--rollback-pack", "cli/model", "--models-dir", models]);
    assert!(!rollback.status.success());
    let rollback: Value = serde_json::from_slice(&rollback.stderr).unwrap();
    assert_eq!(rollback["error"]["code"], "pack_not_installed");

    let removed = output_json(&lens(&[
        "--remove-pack",
        "cli/model",
        "--pack-component",
        "manual-component",
        "--models-dir",
        models,
    ]));
    assert_eq!(removed["operation"], "remove");
    assert_eq!(removed["removed"], true);
    assert!(!Path::new(models).join("manual-component").exists());

    let missing = lens(&["--pack-status", "cli/model", "--models-dir", models]);
    assert!(!missing.status.success());
    let missing: Value = serde_json::from_slice(&missing.stderr).unwrap();
    assert_eq!(missing["error"]["code"], "pack_not_found");
    assert!(!missing["error"]["hint"].as_str().unwrap().is_empty());
}

#[test]
fn lifecycle_cli_rejects_missing_duplicate_unknown_and_flag_valued_options() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().to_str().unwrap();

    let missing = failure_json(&lens(&["--pack-status", "--models-dir", models]));
    assert_eq!(missing["error"]["code"], "missing_argument");

    let duplicate = failure_json(&lens(&[
        "--pack-status",
        "cli/model",
        "--models-dir",
        models,
        "--models-dir",
        models,
    ]));
    assert_eq!(duplicate["error"]["code"], "duplicate_argument");

    let unknown =
        failure_json(&lens(&["--pack-status", "cli/model", "--models-dir", models, "--surprise"]));
    assert_eq!(unknown["error"]["code"], "unknown_argument");

    let flag_valued = failure_json(&lens(&[
        "--remove-pack",
        "cli/model",
        "--models-dir",
        models,
        "--pack-component",
        "--pack-version",
        "1",
    ]));
    assert_eq!(flag_valued["error"]["code"], "missing_argument");
}

#[test]
fn lifecycle_cli_repairs_a_corrupt_ledger_without_manual_edits() {
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    write_pack(&models.join("manual-component"));
    fs::write(models.join(".skwd-model-packs.json"), b"{ broken").unwrap();
    let models_arg = models.to_str().unwrap();

    let repaired = output_json(&lens(&["--repair-pack-state", "--models-dir", models_arg]));

    assert_eq!(repaired["operation"], "repair");
    assert_eq!(repaired["adopted"][0]["component"], "manual-component");
    assert!(Path::new(repaired["backup"].as_str().unwrap()).is_file());
    let status = output_json(&lens(&["--pack-status", "cli/model", "--models-dir", models_arg]));
    assert_eq!(status["active"]["component"], "manual-component");
    assert!(!status["active"]["files"].as_array().unwrap().is_empty());
}

// This is the ONNX backend conformance suite's real Identity model, kept tiny so the public CLI
// lifecycle can prove model loading without shipping a large pack in Git.
const IDENTITY_ONNX: &[u8] = &[
    0x08, 0x0c, 0x12, 0x0c, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2d, 0x74, 0x65, 0x73, 0x74,
    0x3a, 0x5b, 0x0a, 0x10, 0x0a, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e,
    0x74, 0x69, 0x74, 0x79, 0x12, 0x0d, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x69, 0x64, 0x65, 0x6e, 0x74,
    0x69, 0x74, 0x79, 0x5a, 0x1b, 0x0a, 0x01, 0x78, 0x12, 0x16, 0x0a, 0x14, 0x08, 0x01, 0x12, 0x10,
    0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x02, 0x0a, 0x02, 0x08, 0x02,
    0x62, 0x1b, 0x0a, 0x01, 0x79, 0x12, 0x16, 0x0a, 0x14, 0x08, 0x01, 0x12, 0x10, 0x0a, 0x02, 0x08,
    0x01, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x02, 0x0a, 0x02, 0x08, 0x02, 0x42, 0x04, 0x0a,
    0x00, 0x10, 0x18,
];

fn write_real_pack(root: &Path, version: &str, marker: &str) {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("image.onnx"), IDENTITY_ONNX).unwrap();
    fs::write(root.join("text.onnx"), IDENTITY_ONNX).unwrap();
    fs::write(root.join("marker.txt"), marker).unwrap();
    fs::write(
        root.join("tokenizer.json"),
        serde_json::to_vec_pretty(&json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "test": 1 },
                "unk_token": "[UNK]"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("semantic-pack.json"),
        serde_json::to_vec_pretty(&json!({
            "format": 1,
            "id": "cli/real",
            "version": version,
            "dimensions": 4,
            "contextLength": 4,
            "image": {
                "model": "image.onnx",
                "input": "x",
                "output": "y",
                "width": 1,
                "height": 1,
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0]
            },
            "text": {
                "model": "text.onnx",
                "tokenizer": "tokenizer.json",
                "input": "x",
                "output": "y",
                "eosToken": 0,
                "padToken": 0
            }
        }))
        .unwrap(),
    )
    .unwrap();
}

fn write_index(path: &Path, model: &str) {
    let mut bytes = Vec::from(b"SKWDSEM3".as_slice());
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&(model.len() as u32).to_le_bytes());
    bytes.extend_from_slice(model.as_bytes());
    bytes.extend_from_slice(&31_u64.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(b"item");
    bytes.extend_from_slice(&37_u64.to_le_bytes());
    for value in [1.0_f32, 0.0, 0.0, 0.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

#[test]
#[ignore = "requires SKWD_LENS_TEST_RUNTIME pointing to a compatible real ONNX Runtime"]
fn public_cli_import_update_rollback_doctor_resolve_and_remove_real_models() {
    let runtime = PathBuf::from(
        std::env::var_os("SKWD_LENS_TEST_RUNTIME")
            .expect("set SKWD_LENS_TEST_RUNTIME to libonnxruntime.so"),
    );
    let directory = tempfile::tempdir().unwrap();
    let models = directory.path().join("models");
    let first = directory.path().join("first");
    let second = directory.path().join("second");
    let index = directory.path().join("index.sidx");
    write_real_pack(&first, "1", "first");
    write_real_pack(&second, "2", "second");
    write_index(&index, "cli/real@1");
    let models_arg = models.to_str().unwrap();
    let runtime_arg = runtime.to_str().unwrap();
    let index_arg = index.to_str().unwrap();

    let installed = output_json(&lens(&[
        "--install-pack",
        first.join("semantic-pack.json").to_str().unwrap(),
        "--models-dir",
        models_arg,
        "--runtime",
        runtime_arg,
    ]));
    assert_eq!(installed["active"]["version"], "1");
    assert!(!installed["active"]["files"].as_array().unwrap().is_empty());

    let healthy = output_json(&lens(&[
        "--doctor-pack",
        "cli/real",
        "--models-dir",
        models_arg,
        "--runtime",
        runtime_arg,
        "--index",
        index_arg,
    ]));
    assert_eq!(healthy["ok"], true);

    let updated = output_json(&lens(&[
        "--update-pack",
        second.join("semantic-pack.json").to_str().unwrap(),
        "--models-dir",
        models_arg,
        "--runtime",
        runtime_arg,
    ]));
    assert_eq!(updated["active"]["version"], "2");
    assert_eq!(updated["previous"]["version"], "1");

    let rolled_back =
        output_json(&lens(&["--rollback-pack", "cli/real", "--models-dir", models_arg]));
    assert_eq!(rolled_back["active"]["version"], "1");
    let retired_component = rolled_back["previous"]["component"].as_str().unwrap();

    let missing_index = directory.path().join("missing.sidx");
    let resolved = lens(&[
        "--models-dir",
        models_arg,
        "--pack-id",
        "cli/real",
        "--runtime",
        runtime_arg,
        "--index",
        missing_index.to_str().unwrap(),
        "--query",
        "test",
    ]);
    assert!(!resolved.status.success());
    assert!(String::from_utf8_lossy(&resolved.stderr).contains("open index"));

    let removed = output_json(&lens(&[
        "--remove-pack",
        "cli/real",
        "--pack-component",
        retired_component,
        "--models-dir",
        models_arg,
    ]));
    assert_eq!(removed["removed"], true);
    assert!(!models.join(retired_component).exists());
    let active_component = rolled_back["active"]["component"].as_str().unwrap();
    let refused = lens(&[
        "--remove-pack",
        "cli/real",
        "--pack-component",
        active_component,
        "--models-dir",
        models_arg,
    ]);
    assert_eq!(failure_json(&refused)["error"]["code"], "active_pack");
    let removed = output_json(&lens(&[
        "--remove-pack",
        "cli/real",
        "--pack-component",
        active_component,
        "--models-dir",
        models_arg,
        "--allow-active",
    ]));
    assert_eq!(removed["removed"], true);
    assert!(!models.join(active_component).exists());
    let status = lens(&["--pack-status", "cli/real", "--models-dir", models_arg]);
    assert!(!status.status.success());
}
