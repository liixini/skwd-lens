use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};

use super::{PackFileDigest, PackMetadata, validate_pack_metadata};

pub(super) const STATE_FORMAT: u32 = 1;
pub(super) const STATE_FILE: &str = ".skwd-model-packs.json";
const LOCK_FILE: &str = ".skwd-model-packs.lock";
const STATE_TEMP_PREFIX: &str = ".skwd-model-packs-state-";
pub(super) const REMOVAL_MARKER: &str = ".skwd-removal.json";
static STATE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RemovalTransaction {
    pub component: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PackReference {
    pub component: String,
    pub id: String,
    pub version: String,
    pub format: u32,
    pub dimensions: usize,
    pub bytes: u64,
    #[serde(default)]
    pub files: Vec<PackFileDigest>,
    pub installed_by: String,
    pub runtime_api_minor: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_version: Option<String>,
}

impl PackReference {
    pub(super) fn from_metadata(
        component: String,
        metadata: &PackMetadata,
        runtime_version: Option<String>,
    ) -> Self {
        Self {
            component,
            id: metadata.id.clone(),
            version: metadata.version.clone(),
            format: metadata.format,
            dimensions: metadata.dimensions,
            bytes: metadata.bytes,
            files: metadata.files.clone(),
            installed_by: env!("CARGO_PKG_VERSION").to_string(),
            runtime_api_minor: ort::MINOR_VERSION,
            runtime_version,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PackTrack {
    pub active: PackReference,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous: Option<PackReference>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PackState {
    pub format: u32,
    pub written_by: String,
    #[serde(default)]
    pub packs: BTreeMap<String, PackTrack>,
}

impl Default for PackState {
    fn default() -> Self {
        Self {
            format: STATE_FORMAT,
            written_by: env!("CARGO_PKG_VERSION").to_string(),
            packs: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum PackRole {
    Active,
    Previous,
    Inactive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Severity {
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PackIssue {
    pub code: String,
    pub severity: Severity,
    pub message: String,
    pub hint: String,
}

impl PackIssue {
    pub(super) fn error(
        code: impl Into<String>,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Error,
            message: message.into(),
            hint: hint.into(),
        }
    }

    pub(super) fn warning(
        code: impl Into<String>,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Warning,
            message: message.into(),
            hint: hint.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct InstalledPack {
    pub component: String,
    pub id: String,
    pub version: String,
    pub format: u32,
    pub dimensions: usize,
    pub bytes: u64,
    #[serde(skip)]
    pub files: Vec<PackFileDigest>,
    pub manifest: PathBuf,
    pub role: PackRole,
    pub healthy: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<PackIssue>,
}

impl InstalledPack {
    pub(super) fn reference(&self) -> PackReference {
        PackReference {
            component: self.component.clone(),
            id: self.id.clone(),
            version: self.version.clone(),
            format: self.format,
            dimensions: self.dimensions,
            bytes: self.bytes,
            files: self.files.clone(),
            installed_by: String::from("pre-ledger"),
            runtime_api_minor: ort::MINOR_VERSION,
            runtime_version: None,
        }
    }
}

pub(super) struct Catalog {
    pub state: PackState,
    pub installed: Vec<InstalledPack>,
}

pub(super) struct MutationLock(File);

impl MutationLock {
    pub(super) fn acquire(models_dir: &Path) -> anyhow::Result<Self> {
        let path = models_dir.join(LOCK_FILE);
        if path.symlink_metadata().is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            anyhow::bail!("model-pack lock must not be a symbolic link: {}", path.display());
        }
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("open model-pack lock {}", path.display()))?;
        file.lock()
            .with_context(|| format!("lock model-pack directory {}", models_dir.display()))?;
        Ok(Self(file))
    }
}

impl Drop for MutationLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

pub(super) fn canonical_models_dir(models_dir: &Path) -> anyhow::Result<PathBuf> {
    fs::create_dir_all(models_dir)
        .with_context(|| format!("create model directory {}", models_dir.display()))?;
    models_dir
        .canonicalize()
        .with_context(|| format!("resolve model directory {}", models_dir.display()))
}

pub(super) fn load_catalog(models_dir: &Path) -> anyhow::Result<Catalog> {
    let state = load_state(models_dir)?;
    catalog_from_state(models_dir, state)
}

pub(super) fn catalog_from_state(models_dir: &Path, state: PackState) -> anyhow::Result<Catalog> {
    let installed = scan_installed(models_dir, &state)?;
    Ok(Catalog { state, installed })
}

pub(super) fn load_state(models_dir: &Path) -> anyhow::Result<PackState> {
    let path = models_dir.join(STATE_FILE);
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(PackState::default());
        }
        Err(error) => return Err(error).with_context(|| format!("inspect {}", path.display())),
    };
    ensure!(metadata.is_file() && !metadata.file_type().is_symlink(), "invalid model-pack state");
    let state: PackState = serde_json::from_slice(&fs::read(&path)?)
        .with_context(|| format!("parse model-pack state {}", path.display()))?;
    validate_state(&state)?;
    Ok(state)
}

fn validate_state(state: &PackState) -> anyhow::Result<()> {
    ensure!(state.format == STATE_FORMAT, "unsupported model-pack state format");
    let mut components = HashSet::new();
    for (id, track) in &state.packs {
        ensure!(!id.trim().is_empty() && id == &track.active.id, "invalid active pack identity");
        validate_reference(&track.active)?;
        ensure!(components.insert(track.active.component.as_str()), "duplicate pack component");
        if let Some(previous) = &track.previous {
            ensure!(previous.id == *id, "invalid previous pack identity");
            validate_reference(previous)?;
            ensure!(components.insert(previous.component.as_str()), "duplicate pack component");
        }
    }
    Ok(())
}

fn validate_reference(reference: &PackReference) -> anyhow::Result<()> {
    ensure!(safe_component(&reference.component), "invalid pack component");
    ensure!(!reference.id.trim().is_empty(), "empty pack id in state");
    ensure!(!reference.version.trim().is_empty(), "empty pack version in state");
    ensure!(reference.format > 0, "invalid pack format in state");
    ensure!(reference.dimensions > 0, "invalid pack dimensions in state");
    ensure!(reference.runtime_api_minor > 0, "invalid runtime API in state");
    let mut paths = HashSet::new();
    let mut recorded_bytes = 0_u64;
    for file in &reference.files {
        ensure!(safe_file_digest(file), "invalid model-pack file digest in state");
        ensure!(paths.insert(file.path.as_str()), "duplicate model-pack file digest in state");
        recorded_bytes = recorded_bytes
            .checked_add(file.bytes)
            .context("model-pack file sizes overflow in state")?;
    }
    if !reference.files.is_empty() {
        ensure!(
            recorded_bytes == reference.bytes,
            "model-pack byte total differs from file digests"
        );
    }
    Ok(())
}

fn safe_file_digest(file: &PackFileDigest) -> bool {
    super::safe_relative(Path::new(&file.path))
        && file.sha256.len() == 64
        && file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

pub(super) fn save_state(models_dir: &Path, state: &PackState) -> anyhow::Result<()> {
    validate_state(state)?;
    let mut state = state.clone();
    state.written_by = env!("CARGO_PKG_VERSION").to_string();
    let bytes = serde_json::to_vec_pretty(&state)?;
    let sequence = STATE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp = models_dir.join(format!("{STATE_TEMP_PREFIX}{}-{sequence}.tmp", std::process::id()));
    let mut file = OpenOptions::new().create_new(true).write(true).open(&temp)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    if let Err(error) = fs::rename(&temp, models_dir.join(STATE_FILE)) {
        let _ = fs::remove_file(&temp);
        return Err(error).context("publish model-pack state");
    }
    sync_directory(models_dir)?;
    Ok(())
}

pub(super) fn cleanup_transaction_files(models_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut removed = Vec::new();
    for entry in fs::read_dir(models_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !(name.starts_with(".skwd-model-import-") || name.starts_with(STATE_TEMP_PREFIX))
            || !name.ends_with(".tmp")
        {
            continue;
        }
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.is_dir() && !metadata.file_type().is_symlink() {
            recover_removal_transaction(models_dir, &path)?;
            fs::remove_dir_all(&path)?;
        } else if metadata.is_file() && !metadata.file_type().is_symlink() {
            fs::remove_file(&path)?;
        } else {
            continue;
        }
        removed.push(path);
    }
    if !removed.is_empty() {
        sync_directory(models_dir)?;
    }
    Ok(removed)
}

pub(super) fn recover_removal_transactions_for_repair(models_dir: &Path) -> anyhow::Result<()> {
    for entry in fs::read_dir(models_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(".skwd-model-import-") || !name.ends_with(".tmp") {
            continue;
        }
        let transaction = entry.path();
        let metadata = fs::symlink_metadata(&transaction)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            continue;
        }
        let marker = transaction.join(REMOVAL_MARKER);
        if !marker.is_file() {
            continue;
        }
        let removal: RemovalTransaction = serde_json::from_slice(&fs::read(&marker)?)
            .with_context(|| format!("parse removal transaction {}", marker.display()))?;
        ensure!(safe_component(&removal.component), "invalid removal transaction component");
        let removed = transaction.join("removed");
        let original = models_dir.join(&removal.component);
        match (original.exists(), removed.exists()) {
            (false, true) => {
                let removed_metadata = fs::symlink_metadata(&removed)?;
                ensure!(
                    removed_metadata.is_dir() && !removed_metadata.file_type().is_symlink(),
                    "interrupted removal contains an invalid staged component"
                );
                fs::rename(&removed, &original).with_context(|| {
                    format!(
                        "restore interrupted removal of model-pack component {} before state repair",
                        removal.component
                    )
                })?;
                sync_directory(models_dir)?;
            }
            (true, false) => {}
            (true, true) => anyhow::bail!(
                "interrupted removal has both staged and installed copies of {}",
                removal.component
            ),
            (false, false) => {
                anyhow::bail!("interrupted removal lost model-pack component {}", removal.component)
            }
        }
    }
    Ok(())
}

fn recover_removal_transaction(models_dir: &Path, transaction: &Path) -> anyhow::Result<()> {
    let marker = transaction.join(REMOVAL_MARKER);
    if !marker.is_file() {
        return Ok(());
    }
    let removal: RemovalTransaction = serde_json::from_slice(&fs::read(&marker)?)
        .with_context(|| format!("parse removal transaction {}", marker.display()))?;
    ensure!(safe_component(&removal.component), "invalid removal transaction component");
    let state = load_state(models_dir).with_context(|| {
        format!(
            "preserve interrupted removal of {} until model-pack state is repaired",
            removal.component
        )
    })?;
    let referenced = state.packs.values().any(|track| {
        track.active.component == removal.component
            || track.previous.as_ref().is_some_and(|value| value.component == removal.component)
    });
    if !referenced {
        return Ok(());
    }
    let removed = transaction.join("removed");
    let original = models_dir.join(&removal.component);
    if original.is_dir() && !removed.exists() {
        return Ok(());
    }
    ensure!(!original.exists(), "interrupted removal has both staged and installed copies");
    ensure!(removed.is_dir(), "interrupted removal lost referenced component");
    fs::rename(&removed, &original).with_context(|| {
        format!("restore interrupted removal of model-pack component {}", removal.component)
    })?;
    sync_directory(models_dir)?;
    Ok(())
}

pub(super) fn sync_directory(path: &Path) -> anyhow::Result<()> {
    File::open(path)?.sync_all().with_context(|| format!("sync directory {}", path.display()))
}

pub(super) fn sync_tree(root: &Path) -> anyhow::Result<()> {
    let mut directories = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.is_dir() {
            directories.push(path.clone());
            for entry in fs::read_dir(&path)? {
                stack.push(entry?.path());
            }
        } else if metadata.is_file() {
            File::open(&path)?.sync_all()?;
        }
    }
    for directory in directories.into_iter().rev() {
        sync_directory(&directory)?;
    }
    Ok(())
}

fn scan_installed(models_dir: &Path, state: &PackState) -> anyhow::Result<Vec<InstalledPack>> {
    let references = references_by_component(state);
    let mut installed = Vec::new();
    let mut seen = HashSet::new();
    let mut entries = fs::read_dir(models_dir)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let component = entry.file_name().to_string_lossy().into_owned();
        if component.starts_with('.') {
            continue;
        }
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            continue;
        }
        let manifest = path.join("semantic-pack.json");
        if !manifest.exists() {
            continue;
        }
        seen.insert(component.clone());
        let (role, expected) = references
            .get(component.as_str())
            .map_or((PackRole::Inactive, None), |(role, reference)| (*role, Some(*reference)));
        installed.push(scanned_pack(component, manifest, role, expected));
    }
    for (component, (role, reference)) in references {
        if seen.contains(component) {
            continue;
        }
        installed.push(InstalledPack {
            component: component.to_string(),
            id: reference.id.clone(),
            version: reference.version.clone(),
            format: reference.format,
            dimensions: reference.dimensions,
            bytes: reference.bytes,
            files: reference.files.clone(),
            manifest: models_dir.join(component).join("semantic-pack.json"),
            role,
            healthy: false,
            issues: vec![PackIssue::error(
                "pack_missing",
                format!("{} pack component {component} is missing", reference.id),
                "restore the component or install a compatible replacement before rollback",
            )],
        });
    }
    installed.sort_by(|left, right| {
        (&left.id, &left.version, &left.component).cmp(&(
            &right.id,
            &right.version,
            &right.component,
        ))
    });
    Ok(installed)
}

fn references_by_component(state: &PackState) -> HashMap<&str, (PackRole, &PackReference)> {
    let mut references = HashMap::new();
    for track in state.packs.values() {
        references.insert(track.active.component.as_str(), (PackRole::Active, &track.active));
        if let Some(previous) = &track.previous {
            references.insert(previous.component.as_str(), (PackRole::Previous, previous));
        }
    }
    references
}

fn scanned_pack(
    component: String,
    manifest: PathBuf,
    role: PackRole,
    expected: Option<&PackReference>,
) -> InstalledPack {
    match validate_pack_metadata(manifest.parent().expect("manifest has parent")) {
        Ok(metadata) => {
            let mut issues = Vec::new();
            if let Some(expected) = expected {
                if expected.id != metadata.id
                    || expected.version != metadata.version
                    || expected.format != metadata.format
                    || expected.dimensions != metadata.dimensions
                {
                    issues.push(PackIssue::error(
                        "state_manifest_mismatch",
                        format!("component {component} does not match its recorded identity"),
                        "install a verified replacement; do not edit installed manifests in place",
                    ));
                }
                if expected.files.is_empty() {
                    issues.push(PackIssue::error(
                        "pack_integrity_unrecorded",
                        format!("component {component} has no recorded file digests"),
                        "run the state repair command to adopt and hash the installed component",
                    ));
                } else if expected.files != metadata.files {
                    issues.push(PackIssue::error(
                        "pack_integrity_changed",
                        format!(
                            "component {component} no longer matches its recorded SHA-256 file digests ({} bytes now; {} recorded)",
                            metadata.bytes, expected.bytes
                        ),
                        "reinstall a verified copy; installed model-pack files must never be edited in place",
                    ));
                }
            }
            InstalledPack {
                component,
                id: metadata.id,
                version: metadata.version,
                format: metadata.format,
                dimensions: metadata.dimensions,
                bytes: metadata.bytes,
                files: metadata.files,
                manifest,
                role,
                healthy: !issues.iter().any(|issue| issue.severity == Severity::Error),
                issues,
            }
        }
        Err(error) => {
            let expected = expected.cloned();
            InstalledPack {
                component: component.clone(),
                id: expected.as_ref().map_or_else(String::new, |value| value.id.clone()),
                version: expected.as_ref().map_or_else(String::new, |value| value.version.clone()),
                format: expected.as_ref().map_or(0, |value| value.format),
                dimensions: expected.as_ref().map_or(0, |value| value.dimensions),
                bytes: expected.as_ref().map_or(0, |value| value.bytes),
                files: expected.as_ref().map_or_else(Vec::new, |value| value.files.clone()),
                manifest,
                role,
                healthy: false,
                issues: vec![PackIssue::error(
                    "pack_invalid",
                    format!("component {component} is invalid: {error:#}"),
                    "remove the inactive component or install a verified replacement",
                )],
            }
        }
    }
}

pub(super) fn safe_component(component: &str) -> bool {
    !component.is_empty()
        && Path::new(component).components().all(|part| matches!(part, Component::Normal(_)))
        && Path::new(component).components().count() == 1
}
