use std::io::Write;

use super::*;

#[test]
fn parses_semantic_index() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("test.sidx");
    let mut file = File::create(&path).unwrap();
    file.write_all(INDEX_MAGIC).unwrap();
    file.write_all(&2_u32.to_le_bytes()).unwrap();
    file.write_all(&4_u32.to_le_bytes()).unwrap();
    file.write_all(b"test").unwrap();
    file.write_all(&[0; 8]).unwrap();
    file.write_all(&1_u64.to_le_bytes()).unwrap();
    file.write_all(&3_u32.to_le_bytes()).unwrap();
    file.write_all(b"one").unwrap();
    file.write_all(&[0; 8]).unwrap();
    file.write_all(&0.25_f32.to_le_bytes()).unwrap();
    file.write_all(&0.75_f32.to_le_bytes()).unwrap();
    drop(file);

    let index = SemanticIndex::load(&path).unwrap();
    assert_eq!(index.model, "test");
    assert_eq!(index.keys, ["one"]);
    assert_eq!(index.fingerprints, [0]);
    assert_eq!(index.embeddings, [0.25, 0.75]);
    assert_eq!(index.dimensions, 2);
}

#[test]
fn max_pools_frames() {
    let keys = vec![String::from("video"), String::from("video"), String::from("still")];
    let index = SemanticIndex {
        model: String::from("test@1"),
        groups: group_keys(&keys),
        keys,
        fingerprints: vec![0; 3],
        embeddings: vec![0.1, 0.0, 0.9, 0.0, 0.8, 0.0],
        dimensions: 2,
    };

    let ranked = index.rank(&[1.0, 0.0], 3);
    assert_eq!(ranked, [("video", 0.9), ("still", 0.8)]);
}

#[test]
fn two_item_library_ranks() {
    let keys = vec![String::from("forest"), String::from("abstract")];
    let index = SemanticIndex {
        model: String::from("test@1"),
        groups: group_keys(&keys),
        keys,
        fingerprints: vec![0; 2],
        embeddings: vec![0.9, 0.1, 0.2, 0.8],
        dimensions: 2,
    };

    assert!(index.exclusion_threshold(&[0.0, 1.0]).is_none());
    assert_eq!(index.rank(&[1.0, 0.0], 10), [("forest", 0.9), ("abstract", 0.2)]);
}

#[test]
fn negation_removes_cluster() {
    let keys = (0..6).map(|index| format!("item-{index}")).collect::<Vec<_>>();
    let index = SemanticIndex {
        model: String::from("test@1"),
        groups: group_keys(&keys),
        keys,
        fingerprints: vec![0; 6],
        embeddings: vec![0.9, 0.02, 0.8, 0.04, 0.7, 0.06, 0.6, 0.82, 0.5, 0.88, 0.4, 0.94],
        dimensions: 2,
    };
    let negative = [0.0, 1.0];
    let threshold = index.exclusion_threshold(&negative).unwrap();
    let ranked = index.rank_excluding(&[1.0, 0.0], 6, Some((&negative, threshold)));

    assert_eq!(
        ranked.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
        ["item-0", "item-1", "item-2"]
    );
}

#[test]
fn relevance_window_drops_tail() {
    let ranked = vec![("one", 0.150), ("two", 0.140), ("three", 0.130), ("tail", 0.080)];
    assert_eq!(relevant_results(ranked.clone(), Some(0.025), None, 0, Some(256)), ranked[..3]);
    assert_eq!(relevant_results(ranked.clone(), Some(0.005), None, 2, Some(256)), ranked[..2]);
    assert_eq!(relevant_results(ranked.clone(), Some(1.0), None, 0, Some(2)), ranked[..2]);
    assert_eq!(relevant_results(ranked.clone(), None, None, 0, None), ranked);
}

#[test]
fn flat_scores_no_fillers() {
    let ranked = vec![("one", 0.063), ("two", 0.061), ("three", 0.060), ("four", 0.059)];

    assert!(relevant_results(ranked.clone(), Some(0.022), Some(0.015), 0, None).is_empty());
    assert_eq!(relevant_results(ranked.clone(), Some(0.022), Some(0.0), 0, None), ranked);
    assert_eq!(relevant_results(ranked.clone(), Some(0.022), Some(0.015), 1, None), ranked[..1]);
}

#[test]
fn tiny_library_skips_prominence() {
    let ranked = vec![("one", 0.063), ("two", 0.061)];

    assert_eq!(relevant_results(ranked.clone(), Some(0.022), Some(0.015), 0, None), ranked);
}

#[test]
fn reuses_matching_entries() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("test.sidx");
    let entries = vec![
        (
            BuildEntry {
                key: String::from("same"),
                path: PathBuf::new(),
                fingerprint: 11,
                view: ImageView::Full,
            },
            vec![0.25, 0.75],
        ),
        (
            BuildEntry {
                key: String::from("changed"),
                path: PathBuf::new(),
                fingerprint: 12,
                view: ImageView::Full,
            },
            vec![0.5, 0.5],
        ),
    ];
    write_index(&path, "test@1", 99, &entries, 2).unwrap();

    let reusable = reusable_embeddings(&path, "test@1", 2);

    assert_eq!(reusable.get(&(String::from("same"), 11)), Some(&vec![0.25, 0.75]));
    assert!(!reusable.contains_key(&(String::from("changed"), 13)));
    assert!(reusable_embeddings(&path, "test@2", 2).is_empty());
}

#[test]
fn rejects_wrong_embedding_size() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("invalid.sidx");
    let entries = vec![(
        BuildEntry {
            key: String::from("invalid"),
            path: PathBuf::new(),
            fingerprint: 1,
            view: ImageView::Full,
        },
        vec![0.5],
    )];

    assert!(write_index(&path, "test@1", 1, &entries, 2).is_err());
    assert!(!path.exists());
}

#[test]
fn rejects_oversized_strings() {
    let directory = tempfile::tempdir().unwrap();
    let model_path = directory.path().join("model.sidx");
    let oversized = "x".repeat(MAX_INDEX_STRING_BYTES + 1);
    assert!(write_index(&model_path, &oversized, 1, &[], 2).is_err());
    assert!(!model_path.exists());

    let key_path = directory.path().join("key.sidx");
    let entries = vec![(
        BuildEntry { key: oversized, path: PathBuf::new(), fingerprint: 1, view: ImageView::Full },
        vec![0.5, 0.5],
    )];
    assert!(write_index(&key_path, "test@1", 1, &entries, 2).is_err());
    assert!(!key_path.exists());
}

#[test]
fn manifest_defaults() {
    let clip: Manifest = serde_json::from_value(serde_json::json!({
        "format": 1,
        "id": "clip",
        "version": "1",
        "dimensions": 2,
        "contextLength": 77,
        "image": {
            "model": "image.onnx", "width": 1, "height": 1,
            "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]
        },
        "text": { "model": "text.onnx", "tokenizer": "tokenizer.json" }
    }))
    .unwrap();
    assert_eq!(clip.image.input, "images");
    assert_eq!(clip.image.output, "image_embedding");
    assert!(matches!(clip.image.resize_mode, ResizeMode::Cover));
    assert!(matches!(clip.image.resize_filter, ResizeFilter::CatmullRom));
    assert_eq!(clip.text.input, "tokens");
    assert_eq!(clip.text.output, "text_embedding");
    assert_eq!(clip.text.eos_token, 49_407);
    assert!(clip.text.attention_mask.is_none());
    assert!(clip.text.mask_padding);
    assert!(!clip.text.lowercase);

    let siglip: Manifest = serde_json::from_value(serde_json::json!({
        "format": 1,
        "id": "siglip",
        "version": "1",
        "dimensions": 2,
        "contextLength": 64,
        "image": {
            "model": "image.onnx", "input": "pixel_values", "output": "embedding",
            "resizeMode": "stretch",
            "resizeFilter": "bilinear",
            "width": 1, "height": 1, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]
        },
        "text": {
            "model": "text.onnx", "tokenizer": "tokenizer.json",
            "input": "input_ids", "attentionMask": "attention_mask",
            "maskPadding": false, "lowercase": true,
            "output": "embedding", "eosToken": 1, "padToken": 0
        }
    }))
    .unwrap();
    assert_eq!(siglip.image.input, "pixel_values");
    assert!(matches!(siglip.image.resize_mode, ResizeMode::Stretch));
    assert!(matches!(siglip.image.resize_filter, ResizeFilter::Bilinear));
    assert_eq!(siglip.text.attention_mask.as_deref(), Some("attention_mask"));
    assert!(!siglip.text.mask_padding);
    assert!(siglip.text.lowercase);
    assert_eq!(siglip.text.eos_token, 1);
}

#[test]
fn crops_image_views() {
    let source = image::RgbImage::from_fn(6, 2, |x, _| image::Rgb([x as u8, 0, 0]));

    let full = image_view(source.clone(), ImageView::Full);
    let center = image_view(source.clone(), ImageView::Center);
    let left = image_view(source.clone(), ImageView::LeftThird);
    let right = image_view(source, ImageView::RightThird);

    assert_eq!(full.dimensions(), (6, 2));
    assert_eq!(center.dimensions(), (2, 2));
    assert_eq!(center.get_pixel(0, 0)[0], 2);
    assert_eq!(left.dimensions(), (2, 2));
    assert_eq!(left.get_pixel(0, 0)[0], 0);
    assert_eq!(right.dimensions(), (2, 2));
    assert_eq!(right.get_pixel(0, 0)[0], 4);
}

#[test]
fn naflex_patch_grids() {
    assert_eq!(patch_grid(360, 640, 16, 256).unwrap(), [12, 21]);
    assert_eq!(patch_grid(480, 2560, 16, 256).unwrap(), [7, 36]);
    assert_eq!(patch_grid(1440, 2560, 16, 256).unwrap(), [12, 21]);
    assert!(patch_grid(0, 1, 16, 256).is_err());
}

#[test]
fn normalizes_vectors() {
    let mut values = vec![3.0, 4.0];
    normalize(&mut values).unwrap();
    assert!((values[0] - 0.6).abs() < 0.0001);
    assert!((values[1] - 0.8).abs() < 0.0001);
    assert!(normalize(&mut [0.0, 0.0]).is_err());
}

#[test]
fn token_table_dequantizes_rows() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tokens.bin");
    let mut file = File::create(&path).unwrap();
    file.write_all(b"SKWDTOK1").unwrap();
    file.write_all(&3_u32.to_le_bytes()).unwrap();
    file.write_all(&2_u32.to_le_bytes()).unwrap();
    file.write_all(&0.5_f32.to_le_bytes()).unwrap();
    file.write_all(&10_u8.to_le_bytes()).unwrap();
    file.write_all(&[0; 3]).unwrap();
    file.write_all(&[10, 12, 20, 8, 9, 14]).unwrap();
    drop(file);
    let model = TokenEmbeddingModel {
        table: path.clone(),
        input: String::from("token_embeddings"),
        rows: 3,
        dimensions: 2,
        scale: 0.5,
        zero_point: 10,
    };

    let table = TokenEmbeddingTable::load(&path, &model).unwrap();

    assert_eq!(table.lookup(&[2, 0]).unwrap(), [-0.5, 2.0, 0.0, 1.0]);
    assert!(table.lookup(&[-1]).is_err());
    assert!(table.lookup(&[3]).is_err());
}

#[test]
fn projection_row_major_weights() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("projection.bin");
    let mut file = File::create(&path).unwrap();
    file.write_all(b"SKWDPRJ1").unwrap();
    file.write_all(&2_u32.to_le_bytes()).unwrap();
    file.write_all(&3_u32.to_le_bytes()).unwrap();
    for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    drop(file);
    let model =
        TextProjectionModel { path: path.clone(), input_dimensions: 2, output_dimensions: 3 };

    let projection = TextProjection::load(&path, &model).unwrap();

    assert_eq!(projection.apply(&[2.0, 3.0]).unwrap(), [14.0, 19.0, 24.0]);
    assert!(projection.apply(&[1.0]).is_err());
}
