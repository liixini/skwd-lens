use std::collections::HashMap;
use std::io::{self, Read};
use std::path::Path;

pub const INDEX_MAGIC: &[u8; 8] = b"SKWDSEM3";
pub const MAX_INDEX_STRING_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexHeader {
    pub dimensions: u32,
    pub model: String,
    pub fingerprint: u64,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    pub key: String,
    pub fingerprint: u64,
}

#[derive(serde::Deserialize)]
struct ManifestIdentity {
    id: String,
    version: String,
}

pub struct IndexReader<R> {
    reader: R,
    header: IndexHeader,
    remaining: u64,
}

impl<R: Read> IndexReader<R> {
    pub fn new(mut reader: R) -> io::Result<Self> {
        let header = read_header(&mut reader)?;
        let remaining = header.count;
        Ok(Self { reader, header, remaining })
    }

    pub const fn header(&self) -> &IndexHeader {
        &self.header
    }

    pub fn read_entry_into(&mut self, embeddings: &mut Vec<f32>) -> io::Result<IndexEntry> {
        let entry = self.read_entry_header()?;
        let start = embeddings.len();
        embeddings.try_reserve_exact(self.header.dimensions as usize).map_err(|error| {
            invalid(format!("semantic index dimensions are too large: {error}"))
        })?;
        let result = (0..self.header.dimensions).try_for_each(|_| {
            embeddings.push(f32::from_le_bytes(read_array(&mut self.reader)?));
            Ok::<_, io::Error>(())
        });
        if result.is_err() {
            embeddings.truncate(start);
        }
        result?;
        self.remaining -= 1;
        Ok(entry)
    }

    pub fn skip_entry(&mut self) -> io::Result<IndexEntry> {
        let entry = self.read_entry_header()?;
        let bytes = u64::from(self.header.dimensions) * 4;
        let copied = io::copy(&mut self.reader.by_ref().take(bytes), &mut io::sink())?;
        if copied != bytes {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "semantic index embedding is truncated",
            ));
        }
        self.remaining -= 1;
        Ok(entry)
    }

    pub fn finish(mut self) -> io::Result<IndexHeader> {
        if self.remaining != 0 {
            return Err(invalid(format!("semantic index has {} unread entries", self.remaining)));
        }
        let mut trailing = [0_u8; 1];
        if self.reader.read(&mut trailing)? != 0 {
            return Err(invalid("semantic index has trailing data"));
        }
        Ok(self.header)
    }

    fn read_entry_header(&mut self) -> io::Result<IndexEntry> {
        if self.remaining == 0 {
            return Err(invalid("semantic index has no remaining entries"));
        }
        let key_length = read_u32(&mut self.reader)? as usize;
        let key = read_string(&mut self.reader, key_length)?;
        let fingerprint = read_u64(&mut self.reader)?;
        Ok(IndexEntry { key, fingerprint })
    }
}

pub fn read_index_header(path: &Path) -> io::Result<IndexHeader> {
    let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
    read_header(&mut reader)
}

pub fn validate_index(path: &Path) -> io::Result<IndexHeader> {
    let mut reader = IndexReader::new(std::io::BufReader::new(std::fs::File::open(path)?))?;
    if reader.header().count == 0 {
        return Err(invalid("semantic index is empty"));
    }
    for _ in 0..reader.header().count {
        reader.skip_entry()?;
    }
    reader.finish()
}

pub fn read_index_fingerprints(path: &Path) -> io::Result<HashMap<String, u64>> {
    let mut reader = IndexReader::new(std::io::BufReader::new(std::fs::File::open(path)?))?;
    let mut entries = HashMap::new();
    let count = usize::try_from(reader.header().count)
        .map_err(|_| invalid("semantic index entry count is too large"))?;
    entries
        .try_reserve(count)
        .map_err(|error| invalid(format!("semantic index entry count is too large: {error}")))?;
    for _ in 0..reader.header().count {
        let entry = reader.skip_entry()?;
        entries.insert(entry.key, entry.fingerprint);
    }
    reader.finish()?;
    Ok(entries)
}

pub fn manifest_identity(path: &Path) -> io::Result<String> {
    let identity: ManifestIdentity = serde_json::from_slice(&std::fs::read(path)?)
        .map_err(|error| invalid(error.to_string()))?;
    Ok(format!("{}@{}", identity.id, identity.version))
}

#[must_use]
pub fn cache_index_name(model: &str, profile: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in model.bytes().chain([0xff]).chain(profile.bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("index-{hash:016x}.sidx")
}

fn read_header(reader: &mut impl Read) -> io::Result<IndexHeader> {
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != INDEX_MAGIC {
        return Err(invalid("unsupported semantic index"));
    }
    let dimensions = read_u32(reader)?;
    if dimensions == 0 {
        return Err(invalid("semantic index dimensions are empty"));
    }
    let model_length = read_u32(reader)? as usize;
    if model_length == 0 {
        return Err(invalid("semantic index model is empty"));
    }
    let model = read_string(reader, model_length)?;
    let fingerprint = read_u64(reader)?;
    let count = read_u64(reader)?;
    Ok(IndexHeader { dimensions, model, fingerprint, count })
}

fn read_string(reader: &mut impl Read, length: usize) -> io::Result<String> {
    if length > MAX_INDEX_STRING_BYTES {
        return Err(invalid("semantic index string size is invalid"));
    }
    let mut value = vec![0_u8; length];
    reader.read_exact(&mut value)?;
    String::from_utf8(value).map_err(|error| invalid(error.to_string()))
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut value = [0_u8; 4];
    reader.read_exact(&mut value)?;
    Ok(u32::from_le_bytes(value))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_array(reader)?))
}

fn read_array<const N: usize>(reader: &mut impl Read) -> io::Result<[u8; N]> {
    let mut value = [0_u8; N];
    reader.read_exact(&mut value)?;
    Ok(value)
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
#[path = "index_tests.rs"]
mod tests;
