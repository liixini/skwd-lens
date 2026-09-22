use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::Serialize;

use super::catalog::{
    Catalog, InstalledPack, MutationLock, PackIssue, PackReference, PackRole, PackState, PackTrack,
    REMOVAL_MARKER, RemovalTransaction, STATE_FILE, STATE_FORMAT, Severity, canonical_models_dir,
    cleanup_transaction_files, load_catalog, load_state, recover_removal_transactions_for_repair,
    save_state, sync_directory, sync_tree,
};
use super::{PackMetadata, WorkGuard, install_component, unique_work_directory};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImportMode {
    Install,
    Update,
    Replace,
}

impl ImportMode {
    const fn operation(self) -> Operation {
        match self {
            Self::Install => Operation::Install,
            Self::Update => Operation::Update,
            Self::Replace => Operation::Replace,
        }
    }
}

#[derive(Debug)]
pub(crate) enum LifecycleCommand {
    Import {
        mode: ImportMode,
        source: PathBuf,
        models_dir: PathBuf,
        runtime: PathBuf,
        threads: usize,
    },
    List {
        models_dir: PathBuf,
    },
    Status {
        models_dir: PathBuf,
        id: String,
    },
    Doctor {
        models_dir: PathBuf,
        id: String,
        runtime: PathBuf,
        index: PathBuf,
        threads: usize,
    },
    Rollback {
        models_dir: PathBuf,
        id: String,
    },
    Remove {
        models_dir: PathBuf,
        id: String,
        version: Option<String>,
        component: Option<String>,
        allow_active: bool,
    },
    RemoveManifest {
        manifest: PathBuf,
    },
    Repair {
        models_dir: PathBuf,
        component: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Operation {
    Install,
    Update,
    Replace,
    List,
    Status,
    Doctor,
    Rollback,
    Remove,
    Repair,
}

impl fmt::Display for Operation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Install => "install",
            Self::Update => "update",
            Self::Replace => "replace",
            Self::List => "list",
            Self::Status => "status",
            Self::Doctor => "doctor",
            Self::Rollback => "rollback",
            Self::Remove => "remove",
            Self::Repair => "repair",
        };
        formatter.write_str(name)
    }
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum LifecycleReport {
    Import(InstallReport),
    List(ListReport),
    Status(StatusReport),
    Doctor(DoctorReport),
    Rollback(RollbackReport),
    Remove(RemoveReport),
    RemoveManifest(super::removal::RemovalReport),
    Repair(RepairReport),
}

impl LifecycleReport {
    pub(crate) const fn successful(&self) -> bool {
        match self {
            Self::Doctor(report) => report.ok,
            _ => true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InstallReport {
    operation: Operation,
    format: u32,
    id: String,
    version: String,
    dimensions: usize,
    manifest: PathBuf,
    bytes: u64,
    active: PackReference,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous: Option<PackReference>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ListReport {
    operation: Operation,
    format: u32,
    app_version: String,
    models_dir: PathBuf,
    state_written_by: String,
    packs: Vec<InstalledPack>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StatusReport {
    operation: Operation,
    format: u32,
    app_version: String,
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    active: Option<PackReference>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous: Option<PackReference>,
    installed: Vec<InstalledPack>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct CompatibilityCheck {
    subject: String,
    required: String,
    actual: String,
    compatible: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DoctorReport {
    operation: Operation,
    format: u32,
    ok: bool,
    id: String,
    models_dir: PathBuf,
    checks: Vec<CompatibilityCheck>,
    issues: Vec<PackIssue>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RollbackReport {
    operation: Operation,
    format: u32,
    id: String,
    active: PackReference,
    previous: PackReference,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoveReport {
    operation: Operation,
    format: u32,
    id: String,
    version: String,
    component: String,
    removed: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RepairReport {
    operation: Operation,
    format: u32,
    models_dir: PathBuf,
    backup: PathBuf,
    adopted: Vec<PackReference>,
    unresolved: Vec<InstalledPack>,
}

#[derive(Debug)]
pub(super) struct LifecycleError {
    code: &'static str,
    message: String,
    hint: String,
}

impl LifecycleError {
    fn new(code: &'static str, message: impl Into<String>, hint: impl Into<String>) -> Self {
        Self { code, message: message.into(), hint: hint.into() }
    }
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for LifecycleError {}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FailureReport {
    operation: Operation,
    format: u32,
    ok: bool,
    error: FailureDetail,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FailureDetail {
    code: String,
    message: String,
    hint: String,
}

pub(crate) fn failure_report(operation: Operation, error: &anyhow::Error) -> FailureReport {
    let lifecycle = error.downcast_ref::<LifecycleError>();
    FailureReport {
        operation,
        format: STATE_FORMAT,
        ok: false,
        error: FailureDetail {
            code: lifecycle
                .map_or_else(|| String::from("operation_failed"), |value| value.code.into()),
            message: format!("{error:#}"),
            hint: lifecycle.map_or_else(
                || String::from("run the matching status or doctor command for recovery details"),
                |value| value.hint.clone(),
            ),
        },
    }
}

pub(crate) fn requested_operation(arguments: &[String]) -> Option<Operation> {
    const FLAGS: [(&str, Operation); 10] = [
        ("--install-pack", Operation::Install),
        ("--update-pack", Operation::Update),
        ("--replace-pack", Operation::Replace),
        ("--list-packs", Operation::List),
        ("--pack-status", Operation::Status),
        ("--doctor-pack", Operation::Doctor),
        ("--rollback-pack", Operation::Rollback),
        ("--remove-pack", Operation::Remove),
        ("--remove-manifest", Operation::Remove),
        ("--repair-pack-state", Operation::Repair),
    ];
    FLAGS.iter().find_map(|(flag, operation)| {
        arguments.iter().any(|value| value == flag).then_some(*operation)
    })
}

pub(crate) fn parse_command(arguments: &[String]) -> anyhow::Result<Option<LifecycleCommand>> {
    let requested = [
        ("--install-pack", Operation::Install),
        ("--update-pack", Operation::Update),
        ("--replace-pack", Operation::Replace),
        ("--list-packs", Operation::List),
        ("--pack-status", Operation::Status),
        ("--doctor-pack", Operation::Doctor),
        ("--rollback-pack", Operation::Rollback),
        ("--remove-pack", Operation::Remove),
        ("--remove-manifest", Operation::Remove),
        ("--repair-pack-state", Operation::Repair),
    ]
    .into_iter()
    .filter(|(flag, _)| arguments.iter().any(|value| value == flag))
    .collect::<Vec<_>>();
    if requested.is_empty() {
        return Ok(None);
    }
    if requested.len() != 1 {
        return Err(LifecycleError::new(
            "conflicting_operations",
            "model-pack lifecycle operations are mutually exclusive",
            "run one of install, update, replace, list, status, doctor, rollback, remove, or repair",
        )
        .into());
    }
    let (_, operation) = requested[0];
    validate_lifecycle_syntax(arguments, operation)?;
    if let Some(manifest) = optional_value(arguments, "--remove-manifest") {
        ensure!(arguments.len() == 3, "--remove-manifest accepts only the model manifest path");
        return Ok(Some(LifecycleCommand::RemoveManifest { manifest: PathBuf::from(manifest) }));
    }
    let models_dir = required_value(arguments, "--models-dir").map(PathBuf::from)?;
    let threads = optional_number(arguments, "--threads")?.unwrap_or(4);
    ensure!(threads > 0, "--threads must be positive");
    let command = match operation {
        Operation::Install | Operation::Update | Operation::Replace => {
            let flag = match operation {
                Operation::Install => "--install-pack",
                Operation::Update => "--update-pack",
                Operation::Replace => "--replace-pack",
                _ => unreachable!(),
            };
            let mode = match operation {
                Operation::Install => ImportMode::Install,
                Operation::Update => ImportMode::Update,
                Operation::Replace => ImportMode::Replace,
                _ => unreachable!(),
            };
            LifecycleCommand::Import {
                mode,
                source: PathBuf::from(required_value(arguments, flag)?),
                models_dir,
                runtime: runtime_path(arguments)?,
                threads,
            }
        }
        Operation::List => LifecycleCommand::List { models_dir },
        Operation::Status => LifecycleCommand::Status {
            models_dir,
            id: required_value(arguments, "--pack-status")?.to_string(),
        },
        Operation::Doctor => LifecycleCommand::Doctor {
            models_dir,
            id: required_value(arguments, "--doctor-pack")?.to_string(),
            runtime: runtime_path(arguments)?,
            index: PathBuf::from(required_value(arguments, "--index")?),
            threads,
        },
        Operation::Rollback => LifecycleCommand::Rollback {
            models_dir,
            id: required_value(arguments, "--rollback-pack")?.to_string(),
        },
        Operation::Remove => {
            let version = optional_value(arguments, "--pack-version").map(String::from);
            let component = optional_value(arguments, "--pack-component").map(String::from);
            if version.is_some() == component.is_some() {
                return Err(LifecycleError::new(
                    "ambiguous_remove",
                    "remove requires exactly one of --pack-version or --pack-component",
                    "use status to copy the exact inactive component when a version has revisions",
                )
                .into());
            }
            LifecycleCommand::Remove {
                models_dir,
                id: required_value(arguments, "--remove-pack")?.to_string(),
                version,
                component,
                allow_active: arguments.iter().any(|value| value == "--allow-active"),
            }
        }
        Operation::Repair => LifecycleCommand::Repair {
            models_dir,
            component: optional_value(arguments, "--pack-component").map(String::from),
        },
    };
    Ok(Some(command))
}

fn validate_lifecycle_syntax(arguments: &[String], operation: Operation) -> anyhow::Result<()> {
    const KNOWN: [(&str, bool); 17] = [
        ("--install-pack", true),
        ("--update-pack", true),
        ("--replace-pack", true),
        ("--list-packs", false),
        ("--pack-status", true),
        ("--doctor-pack", true),
        ("--rollback-pack", true),
        ("--remove-pack", true),
        ("--remove-manifest", true),
        ("--allow-active", false),
        ("--repair-pack-state", false),
        ("--models-dir", true),
        ("--runtime", true),
        ("--threads", true),
        ("--index", true),
        ("--pack-version", true),
        ("--pack-component", true),
    ];
    let allowed: &[&str] = match operation {
        Operation::Install => &["--install-pack", "--models-dir", "--runtime", "--threads"],
        Operation::Update => &["--update-pack", "--models-dir", "--runtime", "--threads"],
        Operation::Replace => &["--replace-pack", "--models-dir", "--runtime", "--threads"],
        Operation::List => &["--list-packs", "--models-dir"],
        Operation::Status => &["--pack-status", "--models-dir"],
        Operation::Doctor => {
            &["--doctor-pack", "--models-dir", "--runtime", "--threads", "--index"]
        }
        Operation::Rollback => &["--rollback-pack", "--models-dir"],
        Operation::Remove => &[
            "--remove-pack",
            "--remove-manifest",
            "--models-dir",
            "--pack-version",
            "--pack-component",
            "--allow-active",
        ],
        Operation::Repair => &["--repair-pack-state", "--models-dir", "--pack-component"],
    };
    let mut seen = std::collections::HashSet::new();
    let mut position = 1;
    while position < arguments.len() {
        let flag = arguments[position].as_str();
        let Some((_, takes_value)) = KNOWN.iter().find(|(known, _)| *known == flag) else {
            return Err(LifecycleError::new(
                "unknown_argument",
                format!("unknown model-pack argument: {flag}"),
                "use only the options documented for the selected lifecycle operation",
            )
            .into());
        };
        if !seen.insert(flag) {
            return Err(LifecycleError::new(
                "duplicate_argument",
                format!("model-pack argument was supplied more than once: {flag}"),
                "pass each operation and option exactly once",
            )
            .into());
        }
        if !allowed.contains(&flag) {
            return Err(LifecycleError::new(
                "unexpected_argument",
                format!("{flag} is not valid for {operation}"),
                "use only the options documented for the selected lifecycle operation",
            )
            .into());
        }
        if *takes_value {
            let value = arguments.get(position + 1).filter(|value| !value.starts_with('-'));
            if value.is_none() {
                return Err(LifecycleError::new(
                    "missing_argument",
                    format!("use {flag} <value>"),
                    "pass every path and identifier as a separate argument; prefix dash-leading paths with ./",
                )
                .into());
            }
            position += 2;
        } else {
            position += 1;
        }
    }
    Ok(())
}

fn runtime_path(arguments: &[String]) -> anyhow::Result<PathBuf> {
    optional_value(arguments, "--runtime")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("SKWD_LENS_ORT_DYLIB").map(PathBuf::from))
        .or_else(|| std::env::var_os("SKWD_SEMANTIC_ORT_DYLIB").map(PathBuf::from))
        .ok_or_else(|| {
            LifecycleError::new(
                "runtime_required",
                "use --runtime <libonnxruntime.so> or SKWD_LENS_ORT_DYLIB",
                "doctor and imports load both model sessions against the selected CPU runtime",
            )
            .into()
        })
}

fn required_value<'a>(arguments: &'a [String], flag: &str) -> anyhow::Result<&'a str> {
    optional_value(arguments, flag).ok_or_else(|| {
        LifecycleError::new(
            "missing_argument",
            format!("use {flag} <value>"),
            "pass every path and identifier as a separate argument",
        )
        .into()
    })
}

fn optional_value<'a>(arguments: &'a [String], flag: &str) -> Option<&'a str> {
    arguments
        .iter()
        .position(|argument| argument == flag)
        .and_then(|position| arguments.get(position + 1))
        .map(String::as_str)
}

fn optional_number(arguments: &[String], flag: &str) -> anyhow::Result<Option<usize>> {
    optional_value(arguments, flag)
        .map(|value| value.parse().with_context(|| format!("invalid {flag} value: {value}")))
        .transpose()
}

pub(super) fn list(models_dir: &Path) -> anyhow::Result<ListReport> {
    let models_dir = canonical_models_dir(models_dir)?;
    let catalog = load_catalog(&models_dir)?;
    Ok(ListReport {
        operation: Operation::List,
        format: STATE_FORMAT,
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        models_dir,
        state_written_by: catalog.state.written_by,
        packs: catalog.installed,
    })
}

pub(super) fn status(models_dir: &Path, id: &str) -> anyhow::Result<StatusReport> {
    let models_dir = canonical_models_dir(models_dir)?;
    let catalog = load_catalog(&models_dir)?;
    let installed = catalog.installed.into_iter().filter(|pack| pack.id == id).collect::<Vec<_>>();
    let track = catalog.state.packs.get(id);
    if installed.is_empty() && track.is_none() {
        return Err(LifecycleError::new(
            "pack_not_found",
            format!("no installed model pack has id {id}"),
            "run --list-packs to see installed identities",
        )
        .into());
    }
    Ok(StatusReport {
        operation: Operation::Status,
        format: STATE_FORMAT,
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        id: id.to_string(),
        active: track.map(|value| value.active.clone()),
        previous: track.and_then(|value| value.previous.clone()),
        installed,
    })
}

pub(super) fn activate_candidate(
    models_dir: &Path,
    candidate: &Path,
    metadata: &PackMetadata,
    runtime: &Path,
    mode: ImportMode,
) -> anyhow::Result<InstallReport> {
    activate_candidate_with(models_dir, candidate, metadata, runtime, mode, save_state)
}

pub(super) fn activate_candidate_with<F>(
    models_dir: &Path,
    candidate: &Path,
    metadata: &PackMetadata,
    runtime: &Path,
    mode: ImportMode,
    save: F,
) -> anyhow::Result<InstallReport>
where
    F: FnOnce(&Path, &PackState) -> anyhow::Result<()>,
{
    let catalog = load_catalog(models_dir)?;
    let prior = prior_track(&catalog, &metadata.id, mode)?;
    if mode == ImportMode::Update
        && prior.as_ref().is_some_and(|track| track.active.version == metadata.version)
    {
        return Err(LifecycleError::new(
            "same_version",
            format!("{} {} is already active", metadata.id, metadata.version),
            "use --replace-pack for a rebuilt pack with the same model version",
        )
        .into());
    }
    if mode == ImportMode::Replace
        && prior.as_ref().is_some_and(|track| track.active.version != metadata.version)
    {
        return Err(LifecycleError::new(
            "different_version",
            format!(
                "replacement version {} does not match active version {}",
                metadata.version,
                prior.as_ref().unwrap().active.version
            ),
            "use --update-pack when moving to a different model version",
        )
        .into());
    }
    let component = available_component(models_dir, &metadata.id, &metadata.version)?;
    let destination = models_dir.join(&component);
    sync_tree(candidate)?;
    fs::rename(candidate, &destination)
        .with_context(|| format!("publish model pack {}", destination.display()))?;
    sync_directory(models_dir)?;
    let active =
        PackReference::from_metadata(component, metadata, detected_runtime_version(runtime));
    let previous = prior.map(|track| track.active);
    let mut state = catalog.state;
    state.packs.insert(
        metadata.id.clone(),
        PackTrack { active: active.clone(), previous: previous.clone() },
    );
    if let Err(error) = save(models_dir, &state)
        && !state_is_visible(models_dir, &state)
    {
        return Err(LifecycleError::new(
            "state_publish_failed",
            format!(
                "pack files were retained inactive at {} but activation failed: {error:#}",
                destination.display()
            ),
            "run --list-packs, then retry the operation or safely remove the inactive component",
        )
        .into());
    }
    Ok(InstallReport {
        operation: mode.operation(),
        format: metadata.format,
        id: metadata.id.clone(),
        version: metadata.version.clone(),
        dimensions: metadata.dimensions,
        manifest: destination.join("semantic-pack.json"),
        bytes: metadata.bytes,
        active,
        previous,
    })
}

fn prior_track(catalog: &Catalog, id: &str, mode: ImportMode) -> anyhow::Result<Option<PackTrack>> {
    if let Some(track) = catalog.state.packs.get(id) {
        return Ok(Some(track.clone()));
    }
    let adoptable =
        catalog.installed.iter().filter(|pack| pack.id == id && pack.healthy).collect::<Vec<_>>();
    if adoptable.len() == 1 {
        return Ok(Some(PackTrack { active: adoptable[0].reference(), previous: None }));
    }
    if mode != ImportMode::Install {
        let (code, message) = if adoptable.is_empty() {
            ("pack_not_installed", format!("model pack {id} has no active installation"))
        } else {
            (
                "pack_state_ambiguous",
                format!("model pack {id} has multiple untracked installations"),
            )
        };
        return Err(LifecycleError::new(
            code,
            message,
            "run --pack-status and use --install-pack to establish an explicit active version",
        )
        .into());
    }
    Ok(None)
}

fn available_component(models_dir: &Path, id: &str, version: &str) -> anyhow::Result<String> {
    let base = install_component(id, version);
    if !models_dir.join(&base).exists() {
        return Ok(base);
    }
    for revision in 1..=10_000 {
        let component = format!("{base}-r{revision}");
        if !models_dir.join(&component).exists() {
            return Ok(component);
        }
    }
    anyhow::bail!("cannot allocate a model-pack revision component")
}

pub(super) fn rollback(models_dir: &Path, id: &str) -> anyhow::Result<RollbackReport> {
    let models_dir = canonical_models_dir(models_dir)?;
    let _lock = MutationLock::acquire(&models_dir)?;
    cleanup_transaction_files(&models_dir)?;
    let catalog = load_catalog(&models_dir)?;
    let mut state = catalog.state;
    let track = state.packs.get_mut(id).ok_or_else(|| {
        LifecycleError::new(
            "pack_not_installed",
            format!("model pack {id} has no active installation"),
            "run --list-packs to see installed identities",
        )
    })?;
    let previous = track.previous.clone().ok_or_else(|| {
        LifecycleError::new(
            "rollback_unavailable",
            format!("model pack {id} has no retained previous version"),
            "install an update or replacement before requesting rollback",
        )
    })?;
    ensure_reference_healthy(&catalog.installed, &track.active)?;
    ensure_reference_healthy(&catalog.installed, &previous)?;
    let active = std::mem::replace(&mut track.active, previous);
    track.previous = Some(active);
    let report = RollbackReport {
        operation: Operation::Rollback,
        format: STATE_FORMAT,
        id: id.to_string(),
        active: track.active.clone(),
        previous: track.previous.clone().expect("rollback retains prior active"),
    };
    save_visible_state(&models_dir, &state)?;
    Ok(report)
}

fn ensure_reference_healthy(
    installed: &[InstalledPack],
    reference: &PackReference,
) -> anyhow::Result<()> {
    if installed.iter().any(|pack| pack.component == reference.component && pack.healthy) {
        return Ok(());
    }
    Err(LifecycleError::new(
        "pack_unhealthy",
        format!("pack component {} is missing or invalid", reference.component),
        "run doctor and install a verified replacement before changing active versions",
    )
    .into())
}

pub(super) fn remove(
    models_dir: &Path,
    id: &str,
    version: Option<&str>,
    component: Option<&str>,
    allow_active: bool,
) -> anyhow::Result<RemoveReport> {
    remove_with(models_dir, id, version, component, allow_active, save_state)
}

pub(super) fn remove_with<F>(
    models_dir: &Path,
    id: &str,
    version: Option<&str>,
    component: Option<&str>,
    allow_active: bool,
    save: F,
) -> anyhow::Result<RemoveReport>
where
    F: FnOnce(&Path, &PackState) -> anyhow::Result<()>,
{
    let models_dir = canonical_models_dir(models_dir)?;
    let _lock = MutationLock::acquire(&models_dir)?;
    cleanup_transaction_files(&models_dir)?;
    let catalog = load_catalog(&models_dir)?;
    let matches = catalog
        .installed
        .iter()
        .filter(|pack| {
            pack.id == id
                && version.is_none_or(|value| pack.version == value)
                && component.is_none_or(|value| pack.component == value)
        })
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return Err(LifecycleError::new(
            "pack_not_found",
            format!("no installed component matches model pack {id}"),
            "run --pack-status to copy an installed version or component exactly",
        )
        .into());
    }
    if matches.len() != 1 {
        return Err(LifecycleError::new(
            "ambiguous_remove",
            format!("multiple components match model pack {id}"),
            "retry with --pack-component using the exact inactive component from status",
        )
        .into());
    }
    let target = matches[0];
    if target.role == PackRole::Active && !allow_active {
        return Err(LifecycleError::new(
            "active_pack",
            format!("refusing to remove active component {}", target.component),
            "rollback or install an update first, then remove the inactive component",
        )
        .into());
    }
    let mut state = catalog.state;
    let mut changed = false;
    if target.role == PackRole::Active {
        state.packs.remove(id);
        changed = true;
    }
    if let Some(track) = state.packs.get_mut(id)
        && track.previous.as_ref().is_some_and(|value| value.component == target.component)
    {
        track.previous = None;
        changed = true;
    }
    let work = unique_work_directory(&models_dir)?;
    let guard = WorkGuard(work.clone());
    let marker = work.join(REMOVAL_MARKER);
    fs::write(
        &marker,
        serde_json::to_vec_pretty(&RemovalTransaction { component: target.component.clone() })?,
    )?;
    fs::File::open(&marker)?.sync_all()?;
    sync_directory(&work)?;
    let removed = work.join("removed");
    let path = models_dir.join(&target.component);
    fs::rename(&path, &removed)
        .with_context(|| format!("retire model-pack component {}", target.component))?;
    sync_directory(&work)?;
    sync_directory(&models_dir)?;
    if changed
        && let Err(error) = save(&models_dir, &state)
        && !state_is_visible(&models_dir, &state)
    {
        if let Err(restore_error) = fs::rename(&removed, &path) {
            std::mem::forget(guard);
            return Err(restore_error).with_context(|| {
                format!(
                    "preserved staged model-pack component {} after state publication failed: {error:#}",
                    target.component
                )
            });
        }
        sync_directory(&models_dir)?;
        return Err(error).context("publish model-pack removal state");
    }
    let report = RemoveReport {
        operation: Operation::Remove,
        format: STATE_FORMAT,
        id: target.id.clone(),
        version: target.version.clone(),
        component: target.component.clone(),
        removed: true,
    };
    drop(guard);
    Ok(report)
}

fn save_visible_state(models_dir: &Path, state: &PackState) -> anyhow::Result<()> {
    if let Err(error) = save_state(models_dir, state)
        && !state_is_visible(models_dir, state)
    {
        return Err(error);
    }
    Ok(())
}

fn state_is_visible(models_dir: &Path, desired: &PackState) -> bool {
    load_state(models_dir)
        .is_ok_and(|saved| saved.format == desired.format && saved.packs == desired.packs)
}

pub(super) fn repair(
    models_dir: &Path,
    selected_component: Option<&str>,
) -> anyhow::Result<RepairReport> {
    let models_dir = canonical_models_dir(models_dir)?;
    let _lock = MutationLock::acquire(&models_dir)?;
    let legacy_state = match load_state(&models_dir) {
        Ok(state) if state_needs_digest_migration(&state) => {
            cleanup_transaction_files(&models_dir)?;
            Some(state)
        }
        Ok(_) => {
            return Err(LifecycleError::new(
                "state_not_corrupt",
                "model-pack state is already valid and contains integrity digests",
                "use list or doctor instead; repair is reserved for an unreadable or pre-digest ledger",
            )
            .into());
        }
        Err(_) => {
            recover_removal_transactions_for_repair(&models_dir)?;
            None
        }
    };
    let catalog = super::catalog::catalog_from_state(&models_dir, PackState::default())?;
    if let Some(component) = selected_component {
        let selected = catalog.installed.iter().find(|pack| pack.component == component);
        if !selected.is_some_and(|pack| pack.healthy) {
            return Err(LifecycleError::new(
                "pack_unhealthy",
                format!("selected repair component {component} is missing or invalid"),
                "run doctor without repair and select a healthy component reported by list",
            )
            .into());
        }
    }
    let (state, adopted) = if let Some(legacy) = legacy_state {
        migrate_digest_state(legacy, &catalog.installed)?
    } else {
        reconstruct_state(&catalog.installed, selected_component)
    };
    let backup = available_corrupt_state_backup(&models_dir)?;
    fs::rename(models_dir.join(STATE_FILE), &backup)
        .with_context(|| format!("back up corrupt model-pack state to {}", backup.display()))?;
    sync_directory(&models_dir)?;
    if let Err(error) = save_visible_state(&models_dir, &state) {
        fs::rename(&backup, models_dir.join(STATE_FILE))
            .context("restore corrupt model-pack state after repair failed")?;
        sync_directory(&models_dir)?;
        return Err(error).context("publish repaired model-pack state");
    }
    cleanup_transaction_files(&models_dir)?;
    let adopted_components = adopted
        .iter()
        .map(|reference| reference.component.as_str())
        .collect::<std::collections::HashSet<_>>();
    let unresolved = catalog
        .installed
        .into_iter()
        .filter(|pack| !adopted_components.contains(pack.component.as_str()))
        .collect();
    Ok(RepairReport {
        operation: Operation::Repair,
        format: STATE_FORMAT,
        models_dir,
        backup,
        adopted,
        unresolved,
    })
}

fn state_needs_digest_migration(state: &PackState) -> bool {
    state.packs.values().any(|track| {
        track.active.files.is_empty()
            || track.previous.as_ref().is_some_and(|reference| reference.files.is_empty())
    })
}

fn migrate_digest_state(
    mut state: PackState,
    installed: &[InstalledPack],
) -> anyhow::Result<(PackState, Vec<PackReference>)> {
    let mut adopted = Vec::new();
    for track in state.packs.values_mut() {
        track.active = rehash_reference(&track.active, installed)?;
        adopted.push(track.active.clone());
        if let Some(previous) = track.previous.as_mut() {
            *previous = rehash_reference(previous, installed)?;
            adopted.push(previous.clone());
        }
    }
    Ok((state, adopted))
}

fn rehash_reference(
    reference: &PackReference,
    installed: &[InstalledPack],
) -> anyhow::Result<PackReference> {
    let pack = installed.iter().find(|pack| pack.component == reference.component);
    let valid = pack.is_some_and(|pack| {
        pack.healthy
            && pack.id == reference.id
            && pack.version == reference.version
            && pack.format == reference.format
            && pack.dimensions == reference.dimensions
    });
    if !valid {
        return Err(LifecycleError::new(
            "pack_unhealthy",
            format!(
                "pre-digest ledger component {} is missing, invalid, or has a different identity",
                reference.component
            ),
            "restore the recorded component before repair, or reinstall a verified pack and rebuild state",
        )
        .into());
    }
    let pack = pack.expect("validated installed pack");
    let mut migrated = reference.clone();
    migrated.bytes = pack.bytes;
    migrated.files.clone_from(&pack.files);
    Ok(migrated)
}

fn reconstruct_state(
    installed: &[InstalledPack],
    selected_component: Option<&str>,
) -> (PackState, Vec<PackReference>) {
    let mut groups = std::collections::BTreeMap::<&str, Vec<&InstalledPack>>::new();
    for pack in installed.iter().filter(|pack| pack.healthy) {
        groups.entry(pack.id.as_str()).or_default().push(pack);
    }
    let mut adopted = Vec::new();
    let mut state = PackState::default();
    for (id, packs) in groups {
        let selected = selected_component
            .and_then(|component| packs.iter().copied().find(|pack| pack.component == component))
            .or_else(|| (packs.len() == 1).then_some(packs[0]));
        if let Some(pack) = selected {
            let reference = pack.reference();
            state
                .packs
                .insert(id.to_string(), PackTrack { active: reference.clone(), previous: None });
            adopted.push(reference);
        }
    }
    (state, adopted)
}

fn available_corrupt_state_backup(models_dir: &Path) -> anyhow::Result<PathBuf> {
    for revision in 0..10_000 {
        let candidate = models_dir
            .join(format!(".skwd-model-packs-corrupt-{}-{revision}.json", std::process::id()));
        if candidate
            .symlink_metadata()
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
        {
            return Ok(candidate);
        }
    }
    anyhow::bail!("cannot allocate a corrupt model-pack state backup")
}

pub(crate) fn active_manifest(models_dir: &Path, id: Option<&str>) -> anyhow::Result<PathBuf> {
    let models_dir = canonical_models_dir(models_dir)?;
    let _lock = MutationLock::acquire(&models_dir)?;
    cleanup_transaction_files(&models_dir)?;
    let state = load_state(&models_dir)?;
    let track = if let Some(id) = id {
        state.packs.get(id).ok_or_else(|| {
            LifecycleError::new(
                "pack_not_installed",
                format!("model pack {id} has no active installation"),
                "run --list-packs to see active model identities",
            )
        })?
    } else if state.packs.len() == 1 {
        state.packs.values().next().expect("one active model-pack track")
    } else {
        return Err(LifecycleError::new(
            "pack_selection_required",
            "active model-pack resolution requires --pack-id when the ledger does not contain exactly one model",
            "pass --pack-id <model-id> or use an explicit --manifest path",
        )
        .into());
    };
    let manifest = models_dir.join(&track.active.component).join("semantic-pack.json");
    let metadata = fs::symlink_metadata(&manifest)
        .with_context(|| format!("inspect active model manifest {}", manifest.display()))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "active model manifest is invalid"
    );
    Ok(manifest)
}

pub(super) fn doctor(
    models_dir: &Path,
    id: &str,
    runtime: &Path,
    index: &Path,
    threads: usize,
) -> anyhow::Result<DoctorReport> {
    let models_dir = canonical_models_dir(models_dir)?;
    let mut issues = Vec::new();
    let state = match load_state(&models_dir) {
        Ok(state) => state,
        Err(error) => {
            issues.push(PackIssue::error(
                "state_invalid",
                format!("model-pack state cannot be read: {error:#}"),
                "run --repair-pack-state to back up the corrupt ledger and safely adopt unambiguous healthy packs",
            ));
            PackState::default()
        }
    };
    let catalog = load_catalog_with_state(&models_dir, state)?;
    let track = catalog.state.packs.get(id);
    let active = track.map(|value| &value.active);
    if active.is_none() {
        issues.push(PackIssue::error(
            "active_pack_missing",
            format!("model pack {id} has no explicit active version"),
            "install the pack or use --install-pack to adopt a single pre-ledger installation",
        ));
    }
    let installed = active.and_then(|reference| {
        catalog.installed.iter().find(|pack| pack.component == reference.component)
    });
    if let Some(pack) = installed {
        issues.extend(pack.issues.clone());
    }
    for path in stale_transaction_files(&models_dir)? {
        issues.push(PackIssue::warning(
            "stale_transaction",
            format!("incomplete transaction remains at {}", path.display()),
            "a subsequent lifecycle mutation will clean transaction files while holding the lock",
        ));
    }
    let checks = vec![
        app_compatibility(active),
        model_compatibility(id, active, installed),
        runtime_compatibility(active, installed, runtime, threads, &mut issues),
        index_compatibility(id, active, index, &mut issues),
    ];
    let ok = checks.iter().all(|check| check.compatible)
        && !issues.iter().any(|issue| issue.severity == Severity::Error);
    Ok(DoctorReport {
        operation: Operation::Doctor,
        format: STATE_FORMAT,
        ok,
        id: id.to_string(),
        models_dir,
        checks,
        issues,
    })
}

fn app_compatibility(active: Option<&PackReference>) -> CompatibilityCheck {
    CompatibilityCheck {
        subject: String::from("app"),
        required: String::from("Lens model-pack state 1 and semantic manifest format 1"),
        actual: active.map_or_else(
            || format!("skwd-lens {} (no active pack)", env!("CARGO_PKG_VERSION")),
            |reference| {
                format!(
                    "skwd-lens {}; installed by {}; pack format {}",
                    env!("CARGO_PKG_VERSION"),
                    reference.installed_by,
                    reference.format
                )
            },
        ),
        compatible: active.is_some_and(|reference| reference.format == 1),
        detail: String::from("app compatibility is governed by state and manifest formats"),
    }
}

fn model_compatibility(
    id: &str,
    active: Option<&PackReference>,
    installed: Option<&InstalledPack>,
) -> CompatibilityCheck {
    CompatibilityCheck {
        subject: String::from("model"),
        required: active.map_or_else(
            || id.to_string(),
            |value| format!("{}@{} ({} dimensions)", value.id, value.version, value.dimensions),
        ),
        actual: installed.map_or_else(
            || String::from("missing"),
            |pack| format!("{}@{} ({} dimensions)", pack.id, pack.version, pack.dimensions),
        ),
        compatible: installed.is_some_and(|pack| pack.healthy),
        detail: String::from("the active manifest identity and embedding width must match state"),
    }
}

fn runtime_compatibility(
    active: Option<&PackReference>,
    installed: Option<&InstalledPack>,
    runtime: &Path,
    threads: usize,
    issues: &mut Vec<PackIssue>,
) -> CompatibilityCheck {
    let runtime_version = detected_runtime_version(runtime);
    let recorded_version = active.and_then(|reference| reference.runtime_version.as_deref());
    if let (Some(recorded), Some(selected)) = (recorded_version, runtime_version.as_deref())
        && recorded != selected
    {
        issues.push(PackIssue::warning(
            "runtime_version_changed",
            format!(
                "pack was installed with ONNX Runtime {recorded}, but doctor selected {selected}"
            ),
            "use the packaged runtime for reproducibility; doctor still proves compatibility by loading the models",
        ));
    }
    let runtime_result = installed
        .filter(|pack| pack.healthy)
        .map(|pack| super::validate_pack_root(pack.manifest.parent().unwrap(), runtime, threads));
    let runtime_error = runtime_result.as_ref().and_then(|result| result.as_ref().err());
    if let Some(error) = runtime_error {
        issues.push(PackIssue::error(
            "runtime_incompatible",
            format!("active pack cannot load with the selected runtime: {error:#}"),
            "select the packaged CPU runtime or install a pack built for this runtime API",
        ));
    }
    let runtime_ok = runtime_result.is_some_and(|result| result.is_ok());
    CompatibilityCheck {
        subject: String::from("runtime"),
        required: recorded_version.map_or_else(
            || format!("ONNX Runtime API >= 1.{}.x", ort::MINOR_VERSION),
            |version| {
                format!("ONNX Runtime API >= 1.{}.x; installed with {version}", ort::MINOR_VERSION)
            },
        ),
        actual: runtime_version.unwrap_or_else(|| runtime.display().to_string()),
        compatible: runtime_ok,
        detail: String::from("doctor loads both CPU model sessions and validates their I/O names"),
    }
}

fn index_compatibility(
    id: &str,
    active: Option<&PackReference>,
    index: &Path,
    issues: &mut Vec<PackIssue>,
) -> CompatibilityCheck {
    let expected_model = active.map(|reference| format!("{}@{}", reference.id, reference.version));
    let expected_dimensions = active.map(|reference| reference.dimensions);
    let index_result = skwd_lens_proto::validate_index(index);
    let index_ok = index_result.as_ref().is_ok_and(|header| {
        expected_model.as_ref() == Some(&header.model)
            && expected_dimensions == Some(header.dimensions as usize)
    });
    if !index_ok {
        let actual = index_result.as_ref().map_or_else(
            |error| format!("unreadable: {error}"),
            |header| format!("{} ({} dimensions)", header.model, header.dimensions),
        );
        issues.push(PackIssue::error(
            "index_incompatible",
            format!("semantic index is incompatible: {actual}"),
            "rebuild the SKWDSEM3 index with the active model pack before searching",
        ));
    }
    CompatibilityCheck {
        subject: String::from("index"),
        required: format!(
            "SKWDSEM3 {} ({} dimensions)",
            expected_model.as_deref().unwrap_or(id),
            expected_dimensions.unwrap_or_default()
        ),
        actual: index_result.map_or_else(
            |error| format!("unreadable: {error}"),
            |header| format!("SKWDSEM3 {} ({} dimensions)", header.model, header.dimensions),
        ),
        compatible: index_ok,
        detail: String::from(
            "index format, model identity, and embedding width are checked together",
        ),
    }
}

fn load_catalog_with_state(models_dir: &Path, state: PackState) -> anyhow::Result<Catalog> {
    super::catalog::catalog_from_state(models_dir, state)
}

fn stale_transaction_files(models_dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(models_dir)?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            ((name.starts_with(".skwd-model-import-")
                || name.starts_with(".skwd-model-packs-state-"))
                && name.ends_with(".tmp"))
            .then(|| entry.path())
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

pub(super) fn detected_runtime_version(runtime: &Path) -> Option<String> {
    let canonical = runtime.canonicalize().unwrap_or_else(|_| runtime.to_path_buf());
    let name = canonical.file_name()?.to_str()?;
    name.strip_prefix("libonnxruntime.so.").map(String::from)
}

pub(crate) fn execute(command: LifecycleCommand) -> anyhow::Result<LifecycleReport> {
    match command {
        LifecycleCommand::Import { mode, source, models_dir, runtime, threads } => {
            super::import(&source, &models_dir, &runtime, threads, mode)
                .map(LifecycleReport::Import)
        }
        LifecycleCommand::List { models_dir } => list(&models_dir).map(LifecycleReport::List),
        LifecycleCommand::Status { models_dir, id } => {
            status(&models_dir, &id).map(LifecycleReport::Status)
        }
        LifecycleCommand::Doctor { models_dir, id, runtime, index, threads } => {
            doctor(&models_dir, &id, &runtime, &index, threads).map(LifecycleReport::Doctor)
        }
        LifecycleCommand::Rollback { models_dir, id } => {
            rollback(&models_dir, &id).map(LifecycleReport::Rollback)
        }
        LifecycleCommand::Remove { models_dir, id, version, component, allow_active } => {
            remove(&models_dir, &id, version.as_deref(), component.as_deref(), allow_active)
                .map(LifecycleReport::Remove)
        }
        LifecycleCommand::RemoveManifest { manifest } => {
            super::removal::remove(&manifest).map(LifecycleReport::RemoveManifest)
        }
        LifecycleCommand::Repair { models_dir, component } => {
            repair(&models_dir, component.as_deref()).map(LifecycleReport::Repair)
        }
    }
}
