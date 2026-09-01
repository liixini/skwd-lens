use super::*;

#[test]
fn search_request_defaults() {
    let request: SearchRequest = serde_json::from_value(serde_json::json!({
        "generation": 7,
        "query": "rainy city",
        "topK": 12
    }))
    .unwrap();
    assert_eq!(request.negative_weight, 0.5);
    assert_eq!(request.min_score_prominence, None);
    assert!(!request.embedding_only);
    assert_eq!(serde_json::to_value(&request).unwrap()["topK"], 12);
}

#[test]
fn response_variants_exclusive() {
    let matched = SearchResponse::from_matches(
        3,
        vec![Match { rank: 1, key: String::from("one"), score: 0.8 }],
        1.5,
        2.5,
    );
    let embedded = SearchResponse::from_embedding(4, vec![0.25, 0.75], 3.5);
    let failed = SearchResponse::failed(5, &"bad query");

    assert_eq!(matched.matches.len(), 1);
    assert!(matched.embedding.is_none() && matched.error.is_none());
    assert_eq!(embedded.embedding, Some(vec![0.25, 0.75]));
    assert!(embedded.matches.is_empty() && embedded.error.is_none());
    assert_eq!(failed.error.as_deref(), Some("bad query"));
    assert!(failed.matches.is_empty() && failed.embedding.is_none());
}

#[test]
fn build_request_roundtrip() {
    let request = BuildRequest {
        fingerprint: 9,
        entries: vec![BuildEntry {
            key: String::from("static:a.png"),
            path: "/tmp/a.webp".into(),
            fingerprint: 11,
            view: ImageView::Full,
        }],
    };
    let decoded: BuildRequest =
        serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
    assert_eq!(decoded, request);
}

#[test]
fn full_view_implicit() {
    let full: BuildEntry = serde_json::from_value(serde_json::json!({
        "key": "one", "path": "/tmp/one", "fingerprint": 1
    }))
    .unwrap();
    let left: BuildEntry = serde_json::from_value(serde_json::json!({
        "key": "one", "path": "/tmp/one", "fingerprint": 2, "view": "leftThird"
    }))
    .unwrap();

    assert_eq!(full.view, ImageView::Full);
    assert_eq!(left.view, ImageView::LeftThird);
    assert!(serde_json::to_value(full).unwrap().get("view").is_none());
    assert_eq!(serde_json::to_value(left).unwrap()["view"], "leftThird");
}
