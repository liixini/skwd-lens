use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::Serialize;

use super::{load_manifest, pack_file, unique_work_directory};

#[cfg(test)]
mod tests;

#[derive(Debug, Serialize)]
pub(crate) struct RemovalReport {
    operation: &'static str,
    removed: bool,
    manifest: PathBuf,
    files: usize,
}

pub(super) fn remove(manifest: &Path) -> anyhow::Result<RemovalReport> {
    remove_with(manifest, |source, target| fs::rename(source, target))
}

fn remove_with(
    manifest: &Path,
    mut rename: impl FnMut(&Path, &Path) -> std::io::Result<()>,
) -> anyhow::Result<RemovalReport> {
    let manifest = manifest.canonicalize().context("resolve model manifest")?;
    let root = manifest.parent().context("model manifest has no parent")?;
    if let Some(models) = root.parent()
        && models.join(super::catalog::STATE_FILE).is_file()
        && manifest.file_name().is_some_and(|name| name == "semantic-pack.json")
    {
        let component =
            root.file_name().and_then(|name| name.to_str()).context("invalid model directory")?;
        let catalog = super::catalog::load_catalog(models)?;
        if let Some(pack) = catalog.installed.iter().find(|pack| pack.component == component) {
            super::lifecycle::remove(models, &pack.id, None, Some(component), true)?;
            return Ok(RemovalReport {
                operation: "remove",
                removed: true,
                manifest,
                files: pack.files.len(),
            });
        }
    }
    let _lock = super::catalog::MutationLock::acquire(root).context(
        "cannot delete this model here; system-installed models must be removed with the system package manager",
    )?;
    let mut files = model_files(&manifest)?;
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path != manifest
            && path.extension().is_some_and(|ext| ext == "json")
            && load_manifest(&path).is_ok()
        {
            for shared in model_files(&path)? {
                files.remove(&shared);
            }
        }
    }
    files.insert(manifest.clone());
    let work = unique_work_directory(root).context(
        "cannot delete this model here; system-installed models must be removed with the system package manager",
    )?;
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    for (index, source) in files.iter().enumerate() {
        let target = work.join(index.to_string());
        if let Err(error) = rename(source, &target) {
            for (original, staged) in moved.iter().rev() {
                fs::rename(staged, original).with_context(|| {
                    format!(
                        "restore {} after deletion failed; remaining files are in {}",
                        original.display(),
                        work.display()
                    )
                })?;
            }
            fs::remove_dir(&work)?;
            return Err(error).with_context(|| format!("delete model file {}", source.display()));
        }
        moved.push((source.clone(), target));
    }
    fs::remove_dir_all(&work).context("remove retired model files")?;
    Ok(RemovalReport { operation: "remove", removed: true, manifest, files: files.len() })
}

fn model_files(manifest: &Path) -> anyhow::Result<BTreeSet<PathBuf>> {
    let model = load_manifest(manifest)?;
    let root = manifest.parent().context("model manifest has no parent")?;
    let root = root.canonicalize()?;
    let mut paths = vec![model.image.model.clone(), model.text.model.clone(), model.text.tokenizer];
    paths.extend(model.text.token_embeddings.map(|table| table.table));
    paths.extend(model.text.projection.map(|projection| projection.path));
    for graph in [&model.image.model, &model.text.model] {
        let path = pack_file(&root, graph, "model graph")?;
        let file = fs::File::open(&path)?;
        let data = unsafe { memmap2::Mmap::map(&file)? };
        let mut external = Vec::new();
        external_files(&data, "model", 0, &mut external)?;
        for relative in external {
            ensure!(
                super::safe_relative(Path::new(&relative)),
                "external tensor path escapes the model"
            );
            paths.push(graph.parent().unwrap_or(Path::new("")).join(relative));
        }
    }
    paths.iter().map(|path| pack_file(&root, path, "model file")).collect()
}

fn varint(data: &mut &[u8]) -> anyhow::Result<u64> {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        let (&byte, rest) = data.split_first().context("truncated ONNX field")?;
        *data = rest;
        ensure!(shift < 63 || byte <= 1, "invalid ONNX integer");
        value |= u64::from(byte & 127) << shift;
        if byte & 128 == 0 {
            return Ok(value);
        }
    }
    anyhow::bail!("invalid ONNX integer")
}

fn external_files(
    mut data: &[u8],
    kind: &str,
    depth: usize,
    out: &mut Vec<String>,
) -> anyhow::Result<()> {
    ensure!(depth < 64, "ONNX graph nesting exceeds limit");
    let mut key = None;
    let mut value = None;
    while !data.is_empty() {
        let tag = varint(&mut data)?;
        let number = tag >> 3;
        ensure!(number > 0, "invalid ONNX field");
        let size = match tag & 7 {
            0 => {
                varint(&mut data)?;
                continue;
            }
            1 => 8,
            2 => usize::try_from(varint(&mut data)?)?,
            5 => 4,
            _ => anyhow::bail!("unsupported ONNX wire type"),
        };
        ensure!(size <= data.len(), "truncated ONNX field");
        let (field, rest) = data.split_at(size);
        data = rest;
        if tag & 7 != 2 {
            continue;
        }
        if kind == "entry" {
            if number == 1 {
                key = Some(std::str::from_utf8(field)?);
            }
            if number == 2 {
                value = Some(std::str::from_utf8(field)?);
            }
        }
        let child = match (kind, number) {
            ("model", 7) | ("training", 1 | 2) | ("attribute", 6 | 11) => "graph",
            ("model", 20) => "training",
            ("model", 25) => "function",
            ("graph", 1) | ("function", 7) => "node",
            ("graph", 5) | ("attribute", 5 | 10) | ("sparse", 1 | 2) => "tensor",
            ("graph", 15) | ("attribute", 22 | 23) => "sparse",
            ("node", 5) | ("function", 11) => "attribute",
            ("tensor", 13) => "entry",
            _ => continue,
        };
        external_files(field, child, depth + 1, out)?;
    }
    if key == Some("location") {
        out.push(value.context("ONNX external tensor has no location")?.to_string());
    }
    Ok(())
}
