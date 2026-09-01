use std::io;

use super::*;

type DecodedIndex = (IndexHeader, Vec<(IndexEntry, Vec<f32>)>);

fn header(dimensions: u32, model: &[u8], count: u64) -> Vec<u8> {
    let mut bytes = Vec::from(INDEX_MAGIC.as_slice());
    bytes.extend_from_slice(&dimensions.to_le_bytes());
    bytes.extend_from_slice(&(model.len() as u32).to_le_bytes());
    bytes.extend_from_slice(model);
    bytes.extend_from_slice(&19_u64.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes
}

fn index(entries: &[(&str, u64, &[f32])]) -> Vec<u8> {
    let mut bytes = header(2, b"model@1", entries.len() as u64);
    for (key, fingerprint, embedding) in entries {
        bytes.extend_from_slice(&(key.len() as u32).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(&fingerprint.to_le_bytes());
        for value in *embedding {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn decode(bytes: &[u8]) -> io::Result<DecodedIndex> {
    let mut reader = IndexReader::new(bytes)?;
    let count = reader.header().count;
    let dimensions = reader.header().dimensions as usize;
    let mut entries = Vec::new();
    let mut embeddings = Vec::new();
    for _ in 0..count {
        let start = embeddings.len();
        let entry = reader.read_entry_into(&mut embeddings)?;
        entries.push((entry, embeddings[start..start + dimensions].to_vec()));
    }
    Ok((reader.finish()?, entries))
}

#[test]
fn roundtrip_float_bits() {
    let bytes = index(&[("same", 23, &[0.25, 0.75]), ("same", 29, &[f32::NAN, -0.0])]);

    let (header, entries) = decode(&bytes).unwrap();

    assert_eq!(
        header,
        IndexHeader { dimensions: 2, model: "model@1".into(), fingerprint: 19, count: 2 }
    );
    assert_eq!(entries[0].0, IndexEntry { key: "same".into(), fingerprint: 23 });
    assert_eq!(entries[0].1, [0.25, 0.75]);
    assert_eq!(entries[1].0, IndexEntry { key: "same".into(), fingerprint: 29 });
    assert_eq!(entries[1].1[0].to_bits(), f32::NAN.to_bits());
    assert_eq!(entries[1].1[1].to_bits(), (-0.0_f32).to_bits());
}

#[test]
fn fingerprints_skip_embeddings() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("index.sidx");
    std::fs::write(&path, index(&[("", 23, &[0.25, 0.75]), ("two", 29, &[0.5, 1.0])])).unwrap();

    let header = read_index_header(&path).unwrap();
    let fingerprints = read_index_fingerprints(&path).unwrap();

    assert_eq!(header.model, "model@1");
    assert_eq!(header.fingerprint, 19);
    assert_eq!(fingerprints[""], 23);
    assert_eq!(fingerprints["two"], 29);
}

#[test]
fn skip_entry_truncated() {
    let mut bytes = index(&[("one", 23, &[0.25, 0.75])]);
    bytes.pop();
    let mut reader = IndexReader::new(bytes.as_slice()).unwrap();

    assert_eq!(reader.skip_entry().unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
}

#[test]
fn complete_validation_rejects_truncated_and_trailing_bodies() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("index.sidx");
    let complete = index(&[("one", 23, &[0.25, 0.75])]);
    std::fs::write(&path, &complete).unwrap();
    assert_eq!(validate_index(&path).unwrap().count, 1);

    std::fs::write(&path, &complete[..complete.len() - 1]).unwrap();
    assert_eq!(validate_index(&path).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);

    let mut trailing = complete;
    trailing.push(0);
    std::fs::write(&path, trailing).unwrap();
    assert_eq!(validate_index(&path).unwrap_err().kind(), io::ErrorKind::InvalidData);
}

#[test]
fn zero_entry_index() {
    let bytes = header(2, b"model@1", 0);
    let reader = IndexReader::new(bytes.as_slice()).unwrap();
    assert_eq!(reader.finish().unwrap().count, 0);

    let bytes = header(u32::MAX, b"model@1", 0);
    let reader = IndexReader::new(bytes.as_slice()).unwrap();
    assert_eq!(reader.finish().unwrap().dimensions, u32::MAX);
}

#[test]
fn hostile_entry_count() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("index.sidx");
    std::fs::write(&path, header(2, b"model@1", u64::MAX)).unwrap();

    assert_eq!(read_index_fingerprints(&path).unwrap_err().kind(), io::ErrorKind::InvalidData);
}

#[test]
fn truncated_prefixes_fail() {
    let bytes = index(&[("one", 23, &[0.25, 0.75]), ("two", 29, &[0.5, 1.0])]);
    for length in 0..bytes.len() {
        assert!(decode(&bytes[..length]).is_err(), "prefix {length}");
    }
}

#[test]
fn malformed_headers_rejected() {
    let mut bad_magic = header(2, b"model@1", 0);
    bad_magic[0] = b'X';
    let cases = [bad_magic, header(0, b"model@1", 0), header(2, b"", 0), header(2, &[0xff], 0), {
        let mut bytes = Vec::from(INDEX_MAGIC.as_slice());
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&((1_u32 << 20) + 1).to_le_bytes());
        bytes
    }];

    for bytes in cases {
        assert_eq!(
            IndexReader::new(bytes.as_slice()).err().unwrap().kind(),
            io::ErrorKind::InvalidData
        );
    }
}

#[test]
fn malformed_entries_rejected() {
    let mut invalid_key = header(2, b"model@1", 1);
    invalid_key.extend_from_slice(&1_u32.to_le_bytes());
    invalid_key.push(0xff);
    invalid_key.extend_from_slice(&23_u64.to_le_bytes());
    invalid_key.extend_from_slice(&[0_u8; 8]);
    assert_eq!(decode(&invalid_key).unwrap_err().kind(), io::ErrorKind::InvalidData);

    let mut oversized_key = header(2, b"model@1", 1);
    oversized_key.extend_from_slice(&((1_u32 << 20) + 1).to_le_bytes());
    assert_eq!(decode(&oversized_key).unwrap_err().kind(), io::ErrorKind::InvalidData);

    let mut trailing = index(&[("one", 23, &[0.25, 0.75])]);
    trailing.push(0);
    assert_eq!(decode(&trailing).unwrap_err().kind(), io::ErrorKind::InvalidData);
}

#[test]
fn entry_failure_preserves_embeddings() {
    let mut bytes = header(2, b"model@1", 1);
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(b"one");
    bytes.extend_from_slice(&23_u64.to_le_bytes());
    bytes.extend_from_slice(&0.25_f32.to_le_bytes());
    let mut reader = IndexReader::new(bytes.as_slice()).unwrap();
    let mut embeddings = vec![7.0];

    let error = reader.read_entry_into(&mut embeddings).unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    assert_eq!(embeddings, [7.0]);
}

#[test]
fn finish_requires_all_entries() {
    let bytes = index(&[("one", 23, &[0.25, 0.75])]);
    let reader = IndexReader::new(bytes.as_slice()).unwrap();
    assert_eq!(reader.finish().unwrap_err().kind(), io::ErrorKind::InvalidData);
}

#[test]
fn manifest_identity_read() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("semantic-pack.json");
    std::fs::write(&path, br#"{"format":1,"id":"siglip","version":"v2"}"#).unwrap();
    assert_eq!(manifest_identity(&path).unwrap(), "siglip@v2");
}

#[test]
fn cache_names_profile_specific() {
    let full = cache_index_name("siglip2@example", "full");
    assert_eq!(full, cache_index_name("siglip2@example", "full"));
    assert_ne!(full, cache_index_name("siglip2@example", "multiview"));
    assert_ne!(full, cache_index_name("pe-core@example", "full"));
}
