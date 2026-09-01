mod index;
mod jsonl;
mod search;

pub use index::{
    INDEX_MAGIC, IndexEntry, IndexHeader, IndexReader, MAX_INDEX_STRING_BYTES, cache_index_name,
    manifest_identity, read_index_fingerprints, read_index_header, validate_index,
};
pub use jsonl::write_json_line;
pub use search::{
    BuildEntry, BuildProgress, BuildRequest, ImageView, Match, SearchRequest, SearchResponse,
};

pub const PROTOCOL_VERSION: u32 = 1;
pub const TOKEN_EMBEDDING_MAGIC: &[u8; 8] = b"SKWDTOK1";
pub const TEXT_PROJECTION_MAGIC: &[u8; 8] = b"SKWDPRJ1";
