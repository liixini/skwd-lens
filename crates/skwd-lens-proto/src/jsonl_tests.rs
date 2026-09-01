use super::*;

#[test]
fn compact_line_per_record() {
    let mut output = Vec::new();

    write_json_line(&mut output, &serde_json::json!({ "key": "opaque", "value": 7 })).unwrap();

    assert_eq!(
        output,
        br#"{"key":"opaque","value":7}
"#
    );
}
