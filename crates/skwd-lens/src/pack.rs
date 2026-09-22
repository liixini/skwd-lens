use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::DirBuilderExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    Manifest, TextProjection, TextTokenizer, TokenEmbeddingTable, load_cpu_session, load_manifest,
};

mod catalog;
mod lifecycle;
mod removal;

use lifecycle::{ImportMode, InstallReport};
pub(super) use lifecycle::{
    active_manifest, execute, failure_report, parse_command, requested_operation,
};

const MAX_PACK_ENTRIES: usize = 100_000;
const MAX_PACK_BYTES: u64 = 16 * 1_024 * 1_024 * 1_024;
static IMPORT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackMetadata {
    format: u32,
    id: String,
    version: String,
    dimensions: usize,
    bytes: u64,
    files: Vec<PackFileDigest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PackFileDigest {
    path: String,
    bytes: u64,
    sha256: String,
}

fn import(
    source: &Path,
    models_dir: &Path,
    runtime: &Path,
    threads: usize,
    mode: ImportMode,
) -> anyhow::Result<InstallReport> {
    let source = source
        .canonicalize()
        .with_context(|| format!("resolve model pack {}", source.display()))?;
    ensure!(source.is_file(), "model pack source is not a file");
    let models_dir = catalog::canonical_models_dir(models_dir)?;
    let _lock = catalog::MutationLock::acquire(&models_dir)?;
    catalog::cleanup_transaction_files(&models_dir)?;
    let work = unique_work_directory(&models_dir)?;
    let guard = WorkGuard(work.clone());
    let pack_root = if source.file_name().is_some_and(|name| name == "semantic-pack.json") {
        let root = source.parent().context("semantic manifest has no parent directory")?;
        ensure!(!work.starts_with(root), "model pack cannot contain the install directory");
        root.to_path_buf()
    } else {
        validate_archive(&source)?;
        let extracted = work.join("extracted");
        fs::create_dir(&extracted)?;
        extract_archive(&source, &extracted)?;
        validate_tree(&extracted)?;
        find_manifest_root(&extracted)?
    };
    let source_metadata = validate_pack_metadata(&pack_root)?;
    let candidate = work.join("candidate");
    copy_tree(&pack_root, &candidate)?;
    let metadata = validate_pack_root(&candidate, runtime, threads)?;
    ensure!(
        metadata == source_metadata,
        "model pack changed while it was copied; retry from an immutable source"
    );
    let report = lifecycle::activate_candidate(&models_dir, &candidate, &metadata, runtime, mode)?;
    drop(guard);
    Ok(report)
}

fn validate_pack_root(root: &Path, runtime: &Path, threads: usize) -> anyhow::Result<PackMetadata> {
    let runtime = runtime
        .canonicalize()
        .with_context(|| format!("resolve ONNX Runtime {}", runtime.display()))?;
    ensure!(fs::symlink_metadata(&runtime)?.is_file(), "ONNX Runtime is not a regular file");
    let (manifest, metadata) = load_pack_metadata(root)?;
    let root = root.canonicalize()?;
    let image_model = pack_file(&root, &manifest.image.model, "image model")?;
    let text_model = pack_file(&root, &manifest.text.model, "text model")?;
    let tokenizer = pack_file(&root, &manifest.text.tokenizer, "tokenizer")?;
    let token_embeddings = manifest
        .text
        .token_embeddings
        .as_ref()
        .map(|model| pack_file(&root, &model.table, "token table"))
        .transpose()?;
    let projection = manifest
        .text
        .projection
        .as_ref()
        .map(|model| pack_file(&root, &model.path, "text projection"))
        .transpose()?;
    ort::init_from(&runtime)
        .with_context(|| format!("load ONNX Runtime {}", runtime.display()))?
        .commit();
    {
        let session = load_cpu_session(&image_model, threads, "image")?;
        require_input(&session, &manifest.image.input, "image")?;
        if let Some(name) = &manifest.image.pixel_attention_mask {
            require_input(&session, name, "image")?;
        }
        if let Some(name) = &manifest.image.spatial_shapes {
            require_input(&session, name, "image")?;
        }
        require_output(&session, &manifest.image.output, "image")?;
    }
    {
        let session = load_cpu_session(&text_model, threads, "text")?;
        require_input(&session, &manifest.text.input, "text")?;
        if let Some(name) = &manifest.text.attention_mask {
            require_input(&session, name, "text")?;
        }
        if let Some(model) = &manifest.text.token_embeddings {
            require_input(&session, &model.input, "text")?;
        }
        require_output(&session, &manifest.text.output, "text")?;
    }
    TextTokenizer::load(&tokenizer, manifest.text.eos_token)?;
    if let (Some(path), Some(model)) = (token_embeddings, &manifest.text.token_embeddings) {
        TokenEmbeddingTable::load(&path, model)?;
    }
    if let (Some(path), Some(model)) = (projection, &manifest.text.projection) {
        TextProjection::load(&path, model)?;
    }
    Ok(metadata)
}

fn validate_pack_metadata(root: &Path) -> anyhow::Result<PackMetadata> {
    load_pack_metadata(root).map(|(_, metadata)| metadata)
}

fn load_pack_metadata(root: &Path) -> anyhow::Result<(Manifest, PackMetadata)> {
    let root = root
        .canonicalize()
        .with_context(|| format!("resolve model-pack root {}", root.display()))?;
    let (bytes, files) = validate_tree(&root)?;
    let manifest_path = root.join("semantic-pack.json");
    ensure!(manifest_path.is_file(), "model pack has no root semantic-pack.json");
    let manifest = load_manifest(&manifest_path)?;
    pack_file(&root, &manifest.image.model, "image model")?;
    pack_file(&root, &manifest.text.model, "text model")?;
    pack_file(&root, &manifest.text.tokenizer, "tokenizer")?;
    manifest
        .text
        .token_embeddings
        .as_ref()
        .map(|model| pack_file(&root, &model.table, "token table"))
        .transpose()?;
    manifest
        .text
        .projection
        .as_ref()
        .map(|model| pack_file(&root, &model.path, "text projection"))
        .transpose()?;
    let metadata = PackMetadata {
        format: manifest.format,
        id: manifest.id.clone(),
        version: manifest.version.clone(),
        dimensions: manifest.dimensions,
        bytes,
        files,
    };
    Ok((manifest, metadata))
}

fn require_input(session: &ort::session::Session, name: &str, model: &str) -> anyhow::Result<()> {
    ensure!(
        session.inputs().iter().any(|input| input.name() == name),
        "semantic {model} model has no input named {name}"
    );
    Ok(())
}

fn require_output(session: &ort::session::Session, name: &str, model: &str) -> anyhow::Result<()> {
    ensure!(
        session.outputs().iter().any(|output| output.name() == name),
        "semantic {model} model has no output named {name}"
    );
    Ok(())
}

fn pack_file(root: &Path, relative: &Path, label: &str) -> anyhow::Result<PathBuf> {
    ensure!(safe_relative(relative), "{label} path must stay within the model pack");
    let path = root.join(relative);
    let metadata =
        fs::symlink_metadata(&path).with_context(|| format!("read {label} {}", path.display()))?;
    ensure!(metadata.is_file() && !metadata.file_type().is_symlink(), "{label} is not a file");
    let canonical =
        path.canonicalize().with_context(|| format!("resolve {label} {}", path.display()))?;
    ensure!(canonical.starts_with(root), "{label} escapes the model pack");
    Ok(canonical)
}

pub(super) fn safe_relative(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
        && path.components().any(|component| matches!(component, Component::Normal(_)))
}

fn validate_archive(source: &Path) -> anyhow::Result<()> {
    let output = Command::new("bsdtar")
        .arg("-tf")
        .arg(source)
        .output()
        .with_context(|| "bsdtar is required to import model-pack archives")?;
    let listing = String::from_utf8(checked_output(output, "read model-pack archive")?.stdout)
        .context("model-pack archive contains a non-UTF-8 path")?;
    let mut entries = 0;
    for entry in listing.lines().filter(|entry| !entry.is_empty()) {
        entries += 1;
        ensure!(entries <= MAX_PACK_ENTRIES, "model pack has too many entries");
        ensure!(safe_archive_path(Path::new(entry)), "unsafe archive path: {entry}");
    }
    Ok(())
}

fn checked_output(output: Output, operation: &str) -> anyhow::Result<Output> {
    if output.status.success() {
        return Ok(output);
    }
    let detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if detail.is_empty() {
        anyhow::bail!("cannot {operation}")
    }
    anyhow::bail!("cannot {operation}: {detail}")
}

pub(super) fn safe_archive_path(path: &Path) -> bool {
    !path.is_absolute()
        && path.components().all(|component| {
            !matches!(component, Component::ParentDir | Component::RootDir | Component::Prefix(_))
        })
}

fn extract_archive(source: &Path, destination: &Path) -> anyhow::Result<()> {
    let output = Command::new("bsdtar")
        .arg("-xf")
        .arg(source)
        .arg("-C")
        .arg(destination)
        .args(["--no-same-owner", "--no-same-permissions"])
        .output()
        .context("run bsdtar")?;
    checked_output(output, "extract model-pack archive")?;
    Ok(())
}

fn find_manifest_root(root: &Path) -> anyhow::Result<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut manifests = Vec::new();
    while let Some(directory) = stack.pop() {
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.is_dir() {
                stack.push(path);
            } else if metadata.is_file()
                && path.file_name().is_some_and(|name| name == "semantic-pack.json")
            {
                manifests.push(path);
            }
        }
    }
    ensure!(manifests.len() == 1, "model-pack archive must contain exactly one semantic-pack.json");
    Ok(manifests.pop().unwrap().parent().unwrap().to_path_buf())
}

fn validate_tree(root: &Path) -> anyhow::Result<(u64, Vec<PackFileDigest>)> {
    let mut stack = vec![root.to_path_buf()];
    let mut entries = 0;
    let mut bytes = 0u64;
    let mut files = Vec::new();
    while let Some(path) = stack.pop() {
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("inspect model-pack entry {}", path.display()))?;
        entries += 1;
        ensure!(entries <= MAX_PACK_ENTRIES, "model pack has too many entries");
        ensure!(!metadata.file_type().is_symlink(), "model pack contains a symbolic link");
        if metadata.is_dir() {
            for entry in fs::read_dir(&path)? {
                stack.push(entry?.path());
            }
        } else if metadata.is_file() {
            bytes = bytes.checked_add(metadata.len()).context("model-pack size overflow")?;
            ensure!(bytes <= MAX_PACK_BYTES, "model pack expands beyond 16 GiB");
            let relative = path
                .strip_prefix(root)
                .context("model-pack entry escaped its root")?
                .to_str()
                .context("model-pack paths must be valid UTF-8")?
                .replace(std::path::MAIN_SEPARATOR, "/");
            let mut file = fs::File::open(&path)?;
            let mut hasher = Sha256::new();
            let mut buffer = vec![0_u8; 1024 * 1024].into_boxed_slice();
            loop {
                let read = file.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
            files.push(PackFileDigest {
                path: relative,
                bytes: metadata.len(),
                sha256: format!("{:x}", hasher.finalize()),
            });
        } else {
            anyhow::bail!("model pack contains an unsupported file: {}", path.display());
        }
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok((bytes, files))
}

fn copy_tree(source: &Path, destination: &Path) -> anyhow::Result<()> {
    fs::create_dir(destination)?;
    let mut stack = vec![(source.to_path_buf(), destination.to_path_buf())];
    while let Some((from, to)) = stack.pop() {
        for entry in fs::read_dir(&from)? {
            let entry = entry?;
            let from = entry.path();
            let to = to.join(entry.file_name());
            let metadata = fs::symlink_metadata(&from)?;
            if metadata.is_dir() {
                fs::create_dir(&to)?;
                stack.push((from, to));
            } else if metadata.is_file() {
                fs::copy(&from, &to)
                    .with_context(|| format!("copy model-pack file {}", from.display()))?;
            } else {
                anyhow::bail!("model pack contains an unsupported file: {}", from.display());
            }
        }
    }
    Ok(())
}

fn unique_work_directory(models_dir: &Path) -> anyhow::Result<PathBuf> {
    for _ in 0..32 {
        let sequence = IMPORT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path =
            models_dir.join(format!(".skwd-model-import-{}-{sequence}.tmp", std::process::id()));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        builder.mode(0o700);
        match builder.create(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    anyhow::bail!("cannot allocate a model-pack import directory")
}

pub(super) fn install_component(id: &str, version: &str) -> String {
    let mut slug =
        id.chars()
            .chain(std::iter::once('-'))
            .chain(version.chars())
            .map(|character| {
                if character.is_ascii_alphanumeric() { character.to_ascii_lowercase() } else { '-' }
            })
            .collect::<String>();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    let slug = slug.trim_matches('-');
    let slug = if slug.is_empty() { "model" } else { slug };
    let mut hasher = StableHasher::default();
    id.hash(&mut hasher);
    version.hash(&mut hasher);
    format!("{}-{:016x}", slug.chars().take(72).collect::<String>(), hasher.finish())
}

#[derive(Default)]
struct StableHasher(u64);

impl Hasher for StableHasher {
    fn write(&mut self, bytes: &[u8]) {
        if self.0 == 0 {
            self.0 = 0xcbf2_9ce4_8422_2325;
        }
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

struct WorkGuard(PathBuf);

impl Drop for WorkGuard {
    fn drop(&mut self) {
        if self.0.is_dir() {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

#[cfg(test)]
#[path = "pack/lifecycle_tests.rs"]
mod lifecycle_tests;
