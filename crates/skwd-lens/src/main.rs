use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, ensure};
use memmap2::Mmap;
use ort::{session::Session, value::Tensor};
use sentencepiece_rs::SentencePieceProcessor;
use serde::{Deserialize, Serialize};
use skwd_lens_proto::{
    BuildEntry, BuildProgress, BuildRequest, INDEX_MAGIC, ImageView, IndexReader,
    MAX_INDEX_STRING_BYTES, Match, SearchRequest, SearchResponse, TEXT_PROJECTION_MAGIC,
    TOKEN_EMBEDDING_MAGIC, write_json_line,
};
use tokenizers::Tokenizer;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

mod pack;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Manifest {
    format: u32,
    id: String,
    version: String,
    dimensions: usize,
    context_length: usize,
    image: ImageModel,
    text: TextModel,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ImageModel {
    model: PathBuf,
    width: usize,
    height: usize,
    mean: [f32; 3],
    std: [f32; 3],
    #[serde(default)]
    resize_mode: ResizeMode,
    #[serde(default)]
    resize_filter: ResizeFilter,
    #[serde(default = "default_image_input")]
    input: String,
    #[serde(default = "default_image_output")]
    output: String,
    #[serde(default)]
    patch_size: Option<usize>,
    #[serde(default)]
    max_num_patches: Option<usize>,
    #[serde(default)]
    pixel_attention_mask: Option<String>,
    #[serde(default)]
    spatial_shapes: Option<String>,
    #[serde(default)]
    supported_shapes: Vec<[usize; 2]>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ResizeMode {
    #[default]
    Cover,
    Stretch,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ResizeFilter {
    Bilinear,
    #[default]
    CatmullRom,
}

impl ResizeFilter {
    fn image_filter(&self) -> image::imageops::FilterType {
        match self {
            Self::Bilinear => image::imageops::FilterType::Triangle,
            Self::CatmullRom => image::imageops::FilterType::CatmullRom,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TextModel {
    model: PathBuf,
    tokenizer: PathBuf,
    #[serde(default = "default_text_input")]
    input: String,
    #[serde(default)]
    attention_mask: Option<String>,
    #[serde(default = "default_mask_padding")]
    mask_padding: bool,
    #[serde(default)]
    lowercase: bool,
    #[serde(default = "default_text_output")]
    output: String,
    #[serde(default = "default_eos_token")]
    eos_token: u32,
    #[serde(default)]
    pad_token: u32,
    #[serde(default)]
    token_embeddings: Option<TokenEmbeddingModel>,
    #[serde(default)]
    projection: Option<TextProjectionModel>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TokenEmbeddingModel {
    table: PathBuf,
    input: String,
    rows: usize,
    dimensions: usize,
    scale: f32,
    zero_point: u8,
}

struct TokenEmbeddingTable {
    mmap: Mmap,
    input: String,
    rows: usize,
    dimensions: usize,
    scale: f32,
    zero_point: u8,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TextProjectionModel {
    path: PathBuf,
    input_dimensions: usize,
    output_dimensions: usize,
}

struct TextProjection {
    mmap: Mmap,
    input_dimensions: usize,
    output_dimensions: usize,
}

struct SemanticIndex {
    model: String,
    keys: Vec<String>,
    fingerprints: Vec<u64>,
    groups: Vec<(String, Vec<usize>)>,
    embeddings: Vec<f32>,
    dimensions: usize,
}

struct Engine {
    manifest: Manifest,
    index: SemanticIndex,
    session: Session,
    tokenizer: TextTokenizer,
    token_embeddings: Option<TokenEmbeddingTable>,
    projection: Option<TextProjection>,
    model_load_ms: f64,
    index_load_ms: f64,
}

enum TextTokenizer {
    HuggingFace(Box<Tokenizer>),
    SentencePiece(Box<SentencePieceProcessor>),
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Report<'a> {
    format: u32,
    model: &'a str,
    index_model: &'a str,
    items: usize,
    embeddings: usize,
    dimensions: usize,
    threads: usize,
    model_load_ms: f64,
    index_load_ms: f64,
    query_ms: f64,
    search_ms: f64,
    resident_mib: Option<f64>,
    proportional_mib: Option<f64>,
    matches: Vec<Match>,
}

struct Arguments {
    manifest: PathBuf,
    index: PathBuf,
    runtime: PathBuf,
    query: Option<String>,
    serve: bool,
    build_index: bool,
    progress_json: bool,
    top_k: usize,
    threads: usize,
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().collect();
    let operation = pack::requested_operation(&arguments);
    match run(&arguments) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            if let Some(operation) = operation {
                let report = pack::failure_report(operation, &error);
                let stderr = std::io::stderr();
                let mut writer = BufWriter::new(stderr.lock());
                let _ = serde_json::to_writer_pretty(&mut writer, &report);
                let _ = writeln!(writer);
            } else {
                eprintln!("{error:#}");
            }
            ExitCode::FAILURE
        }
    }
}

fn run(arguments: &[String]) -> anyhow::Result<bool> {
    if let Some(command) = pack::parse_command(arguments)? {
        let report = pack::execute(command)?;
        let successful = report.successful();
        serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
        println!();
        return Ok(successful);
    }
    if arguments.iter().skip(1).any(|argument| argument == "--version" || argument == "-V") {
        ensure!(arguments.len() == 2, "--version cannot be combined with other arguments");
        println!("skwd-lens {}", env!("CARGO_PKG_VERSION"));
        return Ok(true);
    }
    let arguments = Arguments::parse(arguments)?;
    if arguments.build_index {
        build_index(&arguments)?;
        return Ok(true);
    }
    let mut engine = Engine::load(&arguments)?;
    if arguments.serve {
        engine.serve()?;
        return Ok(true);
    }
    let request = SearchRequest {
        generation: 0,
        query: arguments.query.context("use --query <natural language>")?,
        negative_query: None,
        negative_weight: 0.5,
        top_k: arguments.top_k,
        score_window: None,
        min_score_prominence: None,
        max_results: None,
        min_results: 0,
        embedding_only: false,
    };
    let (matches, query_ms, search_ms) = engine.search(&request)?;
    let (resident_mib, proportional_mib) = memory_mib();
    let report = Report {
        format: 1,
        model: &engine.manifest.id,
        index_model: &engine.index.model,
        items: engine.index.unique_items(),
        embeddings: engine.index.keys.len(),
        dimensions: engine.index.dimensions,
        threads: arguments.threads,
        model_load_ms: engine.model_load_ms,
        index_load_ms: engine.index_load_ms,
        query_ms,
        search_ms,
        resident_mib,
        proportional_mib,
        matches,
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(true)
}

impl Engine {
    fn load(arguments: &Arguments) -> anyhow::Result<Self> {
        let manifest_path = arguments.manifest.canonicalize().with_context(|| {
            format!("resolve semantic manifest {}", arguments.manifest.display())
        })?;
        let manifest = load_manifest(&manifest_path)?;
        ort::init_from(&arguments.runtime)
            .with_context(|| format!("load ONNX Runtime {}", arguments.runtime.display()))?
            .commit();

        let index_started = Instant::now();
        let index = SemanticIndex::load(&arguments.index)?;
        let index_load_ms = index_started.elapsed().as_secs_f64() * 1_000.0;
        let expected_index_model = format!("{}@{}", manifest.id, manifest.version);
        ensure!(
            index.model == expected_index_model,
            "index uses {} but model pack requires {}",
            index.model,
            expected_index_model
        );
        ensure!(
            index.dimensions == manifest.dimensions,
            "index has {} dimensions but model expects {}",
            index.dimensions,
            manifest.dimensions
        );

        let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let model_path = resolve(root, &manifest.text.model);
        let tokenizer_path = resolve(root, &manifest.text.tokenizer);
        let token_embeddings = manifest
            .text
            .token_embeddings
            .as_ref()
            .map(|model| TokenEmbeddingTable::load(&resolve(root, &model.table), model))
            .transpose()?;
        let projection = manifest
            .text
            .projection
            .as_ref()
            .map(|model| TextProjection::load(&resolve(root, &model.path), model))
            .transpose()?;
        let model_started = Instant::now();
        let session = load_cpu_session(&model_path, arguments.threads, "text")?;
        let tokenizer = TextTokenizer::load(&tokenizer_path, manifest.text.eos_token)?;
        let model_load_ms = model_started.elapsed().as_secs_f64() * 1_000.0;
        Ok(Self {
            manifest,
            index,
            session,
            tokenizer,
            token_embeddings,
            projection,
            model_load_ms,
            index_load_ms,
        })
    }

    fn search(&mut self, request: &SearchRequest) -> anyhow::Result<(Vec<Match>, f64, f64)> {
        ensure!(!request.query.trim().is_empty(), "query must not be empty");
        ensure!(request.top_k > 0, "top-k must be positive");
        ensure!(
            request.negative_weight.is_finite() && request.negative_weight >= 0.0,
            "invalid negative weight"
        );
        ensure!(
            request.score_window.is_none_or(|window| window.is_finite() && window >= 0.0),
            "invalid score window"
        );
        ensure!(
            request
                .min_score_prominence
                .is_none_or(|prominence| prominence.is_finite() && prominence >= 0.0),
            "invalid minimum score prominence"
        );
        ensure!(request.max_results.is_none_or(|limit| limit > 0), "invalid maximum result count");
        let query_started = Instant::now();
        let mut query_embedding = self.encode_query(&request.query)?;
        let negative_embedding = if let Some(negative) =
            request.negative_query.as_deref().filter(|value| !value.trim().is_empty())
        {
            let negative_embedding = self.encode_query(negative)?;
            for (positive, negative) in query_embedding.iter_mut().zip(&negative_embedding) {
                *positive -= request.negative_weight * *negative;
            }
            Some(negative_embedding)
        } else {
            None
        };
        normalize(&mut query_embedding)?;
        let query_ms = query_started.elapsed().as_secs_f64() * 1_000.0;

        let search_started = Instant::now();
        let limit = request.top_k.min(self.index.unique_items());
        let exclusion = negative_embedding
            .as_deref()
            .and_then(|embedding| self.index.exclusion_threshold(embedding))
            .map(|threshold| (negative_embedding.as_deref().unwrap(), threshold));
        let ranked = self.index.rank_excluding(&query_embedding, limit, exclusion);
        let ranked = relevant_results(
            ranked,
            request.score_window,
            request.min_score_prominence,
            request.min_results,
            request.max_results,
        );
        let search_ms = search_started.elapsed().as_secs_f64() * 1_000.0;
        let matches = ranked
            .iter()
            .enumerate()
            .map(|(rank, (key, score))| Match {
                rank: rank + 1,
                key: key.to_string(),
                score: *score,
            })
            .collect();
        Ok((matches, query_ms, search_ms))
    }

    fn encode_query(&mut self, query: &str) -> anyhow::Result<Vec<f32>> {
        let normalized_query = self.manifest.text.lowercase.then(|| query.to_lowercase());
        let query = normalized_query.as_deref().unwrap_or(query);
        let (tokens, mut attention_mask) = tokenize(
            &self.tokenizer,
            query,
            self.manifest.context_length,
            self.manifest.text.eos_token,
            self.manifest.text.pad_token,
        )?;
        if !self.manifest.text.mask_padding {
            attention_mask.fill(1);
        }
        let external_embeddings =
            self.token_embeddings.as_ref().map(|table| table.lookup(&tokens)).transpose()?;
        let token_tensor = Tensor::from_array(([1, self.manifest.context_length], tokens))?;
        let outputs = match (
            self.manifest.text.attention_mask.as_deref(),
            self.token_embeddings.as_ref().zip(external_embeddings),
        ) {
            (Some(mask_name), Some((table, embeddings))) => {
                let mask_tensor =
                    Tensor::from_array(([1, self.manifest.context_length], attention_mask))?;
                let embedding_tensor = Tensor::from_array((
                    [1, self.manifest.context_length, table.dimensions],
                    embeddings,
                ))?;
                self.session.run(ort::inputs![
                    self.manifest.text.input.as_str() => token_tensor,
                    mask_name => mask_tensor,
                    table.input.as_str() => embedding_tensor,
                ])?
            }
            (None, Some((table, embeddings))) => {
                let embedding_tensor = Tensor::from_array((
                    [1, self.manifest.context_length, table.dimensions],
                    embeddings,
                ))?;
                self.session.run(ort::inputs![
                    self.manifest.text.input.as_str() => token_tensor,
                    table.input.as_str() => embedding_tensor,
                ])?
            }
            (Some(mask_name), None) => {
                let mask_tensor =
                    Tensor::from_array(([1, self.manifest.context_length], attention_mask))?;
                self.session.run(ort::inputs![
                    self.manifest.text.input.as_str() => token_tensor,
                    mask_name => mask_tensor,
                ])?
            }
            (None, None) => self.session.run(ort::inputs![
                self.manifest.text.input.as_str() => token_tensor,
            ])?,
        };
        let embedding = outputs[self.manifest.text.output.as_str()].try_extract_array::<f32>()?;
        let embedding: Vec<f32> = embedding.iter().copied().collect();
        let embedding = if let Some(projection) = &self.projection {
            projection.apply(&embedding)?
        } else {
            embedding
        };
        ensure!(embedding.len() == self.manifest.dimensions, "unexpected text embedding shape");
        Ok(embedding)
    }

    fn normalized_query(&mut self, query: &str) -> anyhow::Result<Vec<f32>> {
        ensure!(!query.trim().is_empty(), "query must not be empty");
        let mut embedding = self.encode_query(query)?;
        normalize(&mut embedding)?;
        Ok(embedding)
    }

    fn serve(&mut self) -> anyhow::Result<()> {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .context("build semantic serve runtime")?
            .block_on(self.serve_requests())
    }

    async fn serve_requests(&mut self) -> anyhow::Result<()> {
        let mut lines = tokio::io::BufReader::new(tokio::io::stdin()).lines();
        let mut stdout = tokio::io::stdout();
        let mut buffer = Vec::new();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            let response = match serde_json::from_str::<SearchRequest>(&line) {
                Ok(request) if request.embedding_only => {
                    let started = Instant::now();
                    match self.normalized_query(&request.query) {
                        Ok(embedding) => SearchResponse::from_embedding(
                            request.generation,
                            embedding,
                            started.elapsed().as_secs_f64() * 1_000.0,
                        ),
                        Err(error) => SearchResponse::failed(request.generation, &error),
                    }
                }
                Ok(request) => match self.search(&request) {
                    Ok((matches, query_ms, search_ms)) => SearchResponse::from_matches(
                        request.generation,
                        matches,
                        query_ms,
                        search_ms,
                    ),
                    Err(error) => SearchResponse::failed(request.generation, &error),
                },
                Err(error) => SearchResponse::failed(0, &error),
            };
            buffer.clear();
            write_json_line(&mut buffer, &response)?;
            stdout.write_all(&buffer).await?;
            stdout.flush().await?;
        }
        Ok(())
    }
}

impl TextTokenizer {
    fn load(path: &Path, expected_eos: u32) -> anyhow::Result<Self> {
        let tokenizer = if path.extension().is_some_and(|extension| extension == "model") {
            let tokenizer = SentencePieceProcessor::open(path)
                .with_context(|| format!("load semantic tokenizer {}", path.display()))?;
            ensure!(
                tokenizer.eos_id() == Some(expected_eos as usize),
                "semantic tokenizer EOS does not match the model pack"
            );
            Self::SentencePiece(Box::new(tokenizer))
        } else {
            Self::HuggingFace(Box::new(
                Tokenizer::from_file(path)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))
                    .with_context(|| format!("load semantic tokenizer {}", path.display()))?,
            ))
        };
        Ok(tokenizer)
    }

    fn encode(&self, query: &str) -> anyhow::Result<(Vec<i64>, Vec<i64>)> {
        match self {
            Self::HuggingFace(tokenizer) => {
                let encoding = tokenizer
                    .encode(query, true)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                Ok((
                    encoding.get_ids().iter().map(|value| i64::from(*value)).collect(),
                    encoding.get_attention_mask().iter().map(|value| i64::from(*value)).collect(),
                ))
            }
            Self::SentencePiece(tokenizer) => {
                let mut values = tokenizer
                    .encode_to_ids(query)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?
                    .into_iter()
                    .map(|value| i64::try_from(value).expect("token id fits i64"))
                    .collect::<Vec<_>>();
                if let Some(eos) = tokenizer.eos_id() {
                    values.push(i64::try_from(eos).expect("token id fits i64"));
                }
                let attention_mask = vec![1; values.len()];
                Ok((values, attention_mask))
            }
        }
    }
}

impl TokenEmbeddingTable {
    const HEADER_BYTES: usize = 24;

    fn load(path: &Path, model: &TokenEmbeddingModel) -> anyhow::Result<Self> {
        let file =
            File::open(path).with_context(|| format!("open token table {}", path.display()))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .with_context(|| format!("map token table {}", path.display()))?;
        ensure!(mmap.len() >= Self::HEADER_BYTES, "token table header is truncated");
        ensure!(&mmap[..8] == TOKEN_EMBEDDING_MAGIC, "unsupported token table");
        let rows = u32::from_le_bytes(mmap[8..12].try_into()?) as usize;
        let dimensions = u32::from_le_bytes(mmap[12..16].try_into()?) as usize;
        let scale = f32::from_le_bytes(mmap[16..20].try_into()?);
        let zero_point = mmap[20];
        ensure!(rows == model.rows, "token table row count mismatch");
        ensure!(dimensions == model.dimensions, "token table dimensions mismatch");
        ensure!((scale - model.scale).abs() <= f32::EPSILON, "token table scale mismatch");
        ensure!(zero_point == model.zero_point, "token table zero-point mismatch");
        let values = rows.checked_mul(dimensions).context("token table dimensions overflow")?;
        ensure!(mmap.len() == Self::HEADER_BYTES + values, "token table length mismatch");
        Ok(Self { mmap, input: model.input.clone(), rows, dimensions, scale, zero_point })
    }

    fn lookup(&self, tokens: &[i64]) -> anyhow::Result<Vec<f32>> {
        let mut output = Vec::with_capacity(tokens.len() * self.dimensions);
        for token in tokens {
            let row = usize::try_from(*token).context("negative token id")?;
            ensure!(row < self.rows, "token id {row} exceeds table size");
            let start = Self::HEADER_BYTES + row * self.dimensions;
            output.extend(
                self.mmap[start..start + self.dimensions].iter().map(|value| {
                    (i32::from(*value) - i32::from(self.zero_point)) as f32 * self.scale
                }),
            );
        }
        // SAFETY: no slices into the map survive this point.
        let _ = unsafe { self.mmap.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
        Ok(output)
    }
}

impl TextProjection {
    const HEADER_BYTES: usize = 16;

    fn load(path: &Path, model: &TextProjectionModel) -> anyhow::Result<Self> {
        let file =
            File::open(path).with_context(|| format!("open projection {}", path.display()))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .with_context(|| format!("map projection {}", path.display()))?;
        ensure!(mmap.len() >= Self::HEADER_BYTES, "projection header is truncated");
        ensure!(&mmap[..8] == TEXT_PROJECTION_MAGIC, "unsupported text projection");
        let input_dimensions = u32::from_le_bytes(mmap[8..12].try_into()?) as usize;
        let output_dimensions = u32::from_le_bytes(mmap[12..16].try_into()?) as usize;
        ensure!(input_dimensions == model.input_dimensions, "projection input mismatch");
        ensure!(output_dimensions == model.output_dimensions, "projection output mismatch");
        let values = input_dimensions
            .checked_mul(output_dimensions)
            .context("projection dimensions overflow")?;
        ensure!(mmap.len() == Self::HEADER_BYTES + values * 4, "projection length mismatch");
        Ok(Self { mmap, input_dimensions, output_dimensions })
    }

    fn apply(&self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        ensure!(input.len() == self.input_dimensions, "projection input shape mismatch");
        let mut output = vec![0.0_f32; self.output_dimensions];
        for (row, value) in input.iter().enumerate() {
            let start = Self::HEADER_BYTES + row * self.output_dimensions * 4;
            for (column, target) in output.iter_mut().enumerate() {
                let offset = start + column * 4;
                let weight = f32::from_le_bytes(self.mmap[offset..offset + 4].try_into()?);
                *target += value * weight;
            }
        }
        Ok(output)
    }
}

fn relevant_results(
    mut ranked: Vec<(&str, f32)>,
    score_window: Option<f32>,
    min_score_prominence: Option<f32>,
    min_results: usize,
    max_results: Option<usize>,
) -> Vec<(&str, f32)> {
    let Some((_, best_score)) = ranked.first().copied() else {
        return ranked;
    };
    let prominent = min_score_prominence.is_none_or(|minimum| {
        ranked.len() < 4 || best_score - ranked[ranked.len() / 2].1 >= minimum
    });
    let confidence_limit = score_window.map_or(ranked.len(), |window| {
        let cutoff = best_score - window;
        ranked.partition_point(|(_, score)| *score >= cutoff)
    });
    let minimum = min_results.min(ranked.len());
    let maximum = max_results.unwrap_or(ranked.len()).min(ranked.len());
    ranked.truncate(if prominent { confidence_limit.max(minimum).min(maximum) } else { minimum });
    ranked
}

fn load_cpu_session(path: &Path, threads: usize, model: &str) -> anyhow::Result<Session> {
    let mut builder = Session::builder()?
        .with_intra_threads(threads)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .with_inter_threads(1)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .with_parallel_execution(false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    builder
        .commit_from_file(path)
        .with_context(|| format!("load semantic {model} model {}", path.display()))
}

fn build_index(arguments: &Arguments) -> anyhow::Result<()> {
    let manifest_path = arguments
        .manifest
        .canonicalize()
        .with_context(|| format!("resolve semantic manifest {}", arguments.manifest.display()))?;
    let manifest = load_manifest(&manifest_path)?;
    let request: BuildRequest = serde_json::from_reader(std::io::stdin().lock())?;
    ensure!(!request.entries.is_empty(), "semantic catalog is empty");
    let catalog_total = request.entries.len();
    let stdout = std::io::stdout();
    let mut progress_writer = arguments.progress_json.then(|| BufWriter::new(stdout.lock()));
    let model = format!("{}@{}", manifest.id, manifest.version);
    let mut reusable = reusable_embeddings(&arguments.index, &model, manifest.dimensions);
    let mut planned = Vec::with_capacity(request.entries.len());
    let mut missing = 0;
    for entry in request.entries {
        let embedding = reusable.remove(&(entry.key.clone(), entry.fingerprint));
        missing += usize::from(embedding.is_none());
        planned.push((entry, embedding));
    }
    let reused = catalog_total - missing;
    let mut report_progress = |progress: usize, detail: &str| {
        if !progress.is_multiple_of(8) && progress != missing {
            return;
        }
        let Some(writer) = progress_writer.as_mut() else {
            return;
        };
        let update = BuildProgress { progress, total: missing, detail: detail.to_string() };
        if write_json_line(&mut *writer, &update).is_ok() {
            let _ = writer.flush();
        }
    };
    let initial_detail = if missing == 0 {
        format!("Reusing {reused} cached embeddings")
    } else {
        format!("Encoding {missing} new or changed embeddings; reusing {reused}")
    };
    report_progress(0, &initial_detail);
    let mut newly_encoded = 0;
    let mut processed_missing = 0;
    let mut encoded = Vec::with_capacity(catalog_total);
    let mut session = if missing == 0 {
        None
    } else {
        ort::init_from(&arguments.runtime)
            .with_context(|| format!("load ONNX Runtime {}", arguments.runtime.display()))?
            .commit();
        let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let model_path = resolve(root, &manifest.image.model);
        Some(load_cpu_session(&model_path, arguments.threads, "image")?)
    };
    for (entry, existing) in planned {
        let detail = entry.key.clone();
        if let Some(embedding) = existing {
            encoded.push((entry, embedding));
        } else {
            match encode_image(
                session.as_mut().expect("missing embeddings require a session"),
                &manifest.image,
                &entry.path,
                entry.view,
                manifest.dimensions,
            ) {
                Ok(mut embedding) => {
                    normalize(&mut embedding)?;
                    newly_encoded += 1;
                    encoded.push((entry, embedding));
                }
                Err(error) => eprintln!("semantic index skipped {}: {error}", entry.path.display()),
            }
            processed_missing += 1;
            report_progress(processed_missing, &detail);
        }
    }
    ensure!(!encoded.is_empty(), "no semantic catalog images could be encoded");
    eprintln!(
        "semantic index: reused {}, encoded {}, skipped {}",
        reused,
        newly_encoded,
        missing - newly_encoded
    );
    write_index(&arguments.index, &model, request.fingerprint, &encoded, manifest.dimensions)?;
    Ok(())
}

fn reusable_embeddings(
    path: &Path,
    model: &str,
    dimensions: usize,
) -> HashMap<(String, u64), Vec<f32>> {
    let Ok(index) = SemanticIndex::load(path) else { return HashMap::new() };
    if index.model != model || index.dimensions != dimensions {
        return HashMap::new();
    }
    index
        .keys
        .into_iter()
        .zip(index.fingerprints)
        .zip(index.embeddings.chunks_exact(dimensions))
        .map(|((key, fingerprint), embedding)| ((key, fingerprint), embedding.to_vec()))
        .collect()
}

fn encode_image(
    session: &mut Session,
    model: &ImageModel,
    path: &Path,
    view: ImageView,
    dimensions: usize,
) -> anyhow::Result<Vec<f32>> {
    let source = image::open(path).with_context(|| format!("decode {}", path.display()))?.to_rgb8();
    let source = image_view(source, view);
    if let (Some(patch_size), Some(max_num_patches)) = (model.patch_size, model.max_num_patches) {
        return encode_patch_image(
            session,
            model,
            &source,
            patch_size,
            max_num_patches,
            dimensions,
        );
    }
    let prepared = match model.resize_mode {
        ResizeMode::Stretch => image::imageops::resize(
            &source,
            model.width as u32,
            model.height as u32,
            model.resize_filter.image_filter(),
        ),
        ResizeMode::Cover => {
            let scale = (model.width as f64 / f64::from(source.width()))
                .max(model.height as f64 / f64::from(source.height()));
            let resized_width = (f64::from(source.width()) * scale).ceil() as u32;
            let resized_height = (f64::from(source.height()) * scale).ceil() as u32;
            let resized = image::imageops::resize(
                &source,
                resized_width,
                resized_height,
                model.resize_filter.image_filter(),
            );
            let left = (resized_width - model.width as u32) / 2;
            let top = (resized_height - model.height as u32) / 2;
            image::imageops::crop_imm(&resized, left, top, model.width as u32, model.height as u32)
                .to_image()
        }
    };
    let plane = model.width * model.height;
    let mut values = vec![0.0_f32; plane * 3];
    for (index, pixel) in prepared.pixels().enumerate() {
        for channel in 0..3 {
            values[channel * plane + index] =
                (f32::from(pixel[channel]) / 255.0 - model.mean[channel]) / model.std[channel];
        }
    }
    let tensor = Tensor::from_array(([1, 3, model.height, model.width], values))?;
    let outputs = session.run(ort::inputs![model.input.as_str() => tensor])?;
    let embedding = outputs[model.output.as_str()].try_extract_array::<f32>()?;
    ensure!(embedding.len() == dimensions, "unexpected image embedding shape");
    Ok(embedding.iter().copied().collect())
}

fn image_view(source: image::RgbImage, view: ImageView) -> image::RgbImage {
    let width = source.width();
    let height = source.height();
    let (left, top, crop_width, crop_height) = match view {
        ImageView::Full => return source,
        ImageView::Center => {
            let edge = width.min(height);
            ((width - edge) / 2, (height - edge) / 2, edge, edge)
        }
        ImageView::LeftThird => (0, 0, (width / 3).max(1), height),
        ImageView::RightThird => {
            let third = (width / 3).max(1);
            (width - third, 0, third, height)
        }
    };
    image::imageops::crop_imm(&source, left, top, crop_width, crop_height).to_image()
}

fn encode_patch_image(
    session: &mut Session,
    model: &ImageModel,
    source: &image::RgbImage,
    patch_size: usize,
    max_num_patches: usize,
    dimensions: usize,
) -> anyhow::Result<Vec<f32>> {
    let mask_input = model
        .pixel_attention_mask
        .as_deref()
        .context("patch image model is missing pixelAttentionMask")?;
    let shapes_input =
        model.spatial_shapes.as_deref().context("patch image model is missing spatialShapes")?;
    let [patch_height, patch_width] =
        patch_grid(source.height() as usize, source.width() as usize, patch_size, max_num_patches)?;
    ensure!(
        model.supported_shapes.contains(&[patch_height, patch_width]),
        "patch grid {patch_height}x{patch_width} is not supported by the model pack"
    );
    let prepared = image::imageops::resize(
        source,
        (patch_width * patch_size) as u32,
        (patch_height * patch_size) as u32,
        image::imageops::FilterType::Triangle,
    );
    let patch_values = patch_values(&prepared, model, patch_size, max_num_patches);
    let patch_count = patch_height * patch_width;
    let mut pixel_attention_mask = vec![0_i64; max_num_patches];
    pixel_attention_mask[..patch_count].fill(1);
    let spatial_shapes = vec![patch_height as i64, patch_width as i64];
    let pixel_tensor =
        Tensor::from_array(([1, max_num_patches, patch_size * patch_size * 3], patch_values))?;
    let mask_tensor = Tensor::from_array(([1, max_num_patches], pixel_attention_mask))?;
    let shapes_tensor = Tensor::from_array(([1, 2], spatial_shapes))?;
    let outputs = session.run(ort::inputs![
        model.input.as_str() => pixel_tensor,
        mask_input => mask_tensor,
        shapes_input => shapes_tensor,
    ])?;
    let embedding = outputs[model.output.as_str()].try_extract_array::<f32>()?;
    ensure!(embedding.len() == dimensions, "unexpected image embedding shape");
    Ok(embedding.iter().copied().collect())
}

fn patch_grid(
    source_height: usize,
    source_width: usize,
    patch_size: usize,
    max_num_patches: usize,
) -> anyhow::Result<[usize; 2]> {
    ensure!(source_height > 0 && source_width > 0, "patch image source is empty");
    ensure!(patch_size > 0 && max_num_patches > 0, "invalid patch image configuration");
    let scaled = |size: usize, scale: f64| {
        ((size as f64 * scale / patch_size as f64).ceil() as usize).max(1)
    };
    let mut minimum = 0.000_001_f64;
    let mut maximum = 100.0_f64;
    while maximum - minimum >= 0.000_01 {
        let scale = f64::midpoint(minimum, maximum);
        let height = scaled(source_height, scale);
        let width = scaled(source_width, scale);
        if height.checked_mul(width).is_some_and(|patches| patches <= max_num_patches) {
            minimum = scale;
        } else {
            maximum = scale;
        }
    }
    Ok([scaled(source_height, minimum), scaled(source_width, minimum)])
}

fn patch_values(
    source: &image::RgbImage,
    model: &ImageModel,
    patch_size: usize,
    max_num_patches: usize,
) -> Vec<f32> {
    let patch_height = source.height() as usize / patch_size;
    let patch_width = source.width() as usize / patch_size;
    let patch_stride = patch_size * patch_size * 3;
    let mut values = vec![0.0_f32; max_num_patches * patch_stride];
    for patch_y in 0..patch_height {
        for patch_x in 0..patch_width {
            let patch = patch_y * patch_width + patch_x;
            for y in 0..patch_size {
                for x in 0..patch_size {
                    let pixel = source.get_pixel(
                        (patch_x * patch_size + x) as u32,
                        (patch_y * patch_size + y) as u32,
                    );
                    for channel in 0..3 {
                        let offset = patch * patch_stride + (y * patch_size + x) * 3 + channel;
                        values[offset] = (f32::from(pixel[channel]) / 255.0 - model.mean[channel])
                            / model.std[channel];
                    }
                }
            }
        }
    }
    values
}

fn write_index(
    path: &Path,
    model: &str,
    fingerprint: u64,
    entries: &[(BuildEntry, Vec<f32>)],
    dimensions: usize,
) -> anyhow::Result<()> {
    let encoded_dimensions = index_u32(dimensions, "semantic index dimensions")?;
    let model_length = index_string_length(model.len(), "semantic index model length")?;
    let entry_count =
        u64::try_from(entries.len()).context("semantic index entry count exceeds format limit")?;
    for (entry, embedding) in entries {
        index_string_length(entry.key.len(), "semantic index key length")?;
        ensure!(
            embedding.len() == dimensions,
            "semantic index embedding for {} has {} dimensions, expected {}",
            entry.key,
            embedding.len(),
            dimensions
        );
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut writer = BufWriter::new(File::create(&temp)?);
    writer.write_all(INDEX_MAGIC)?;
    writer.write_all(&encoded_dimensions.to_le_bytes())?;
    writer.write_all(&model_length.to_le_bytes())?;
    writer.write_all(model.as_bytes())?;
    writer.write_all(&fingerprint.to_le_bytes())?;
    writer.write_all(&entry_count.to_le_bytes())?;
    for (entry, embedding) in entries {
        writer.write_all(
            &index_string_length(entry.key.len(), "semantic index key length")?.to_le_bytes(),
        )?;
        writer.write_all(entry.key.as_bytes())?;
        writer.write_all(&entry.fingerprint.to_le_bytes())?;
        for value in embedding {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    std::fs::rename(&temp, path).with_context(|| format!("install index {}", path.display()))?;
    std::fs::write(fingerprint_path(path), fingerprint.to_string())?;
    Ok(())
}

fn index_u32(value: usize, field: &str) -> anyhow::Result<u32> {
    u32::try_from(value).with_context(|| format!("{field} exceeds format limit"))
}

fn index_string_length(value: usize, field: &str) -> anyhow::Result<u32> {
    ensure!(value <= MAX_INDEX_STRING_BYTES, "{field} exceeds format limit");
    index_u32(value, field)
}

fn fingerprint_path(index: &Path) -> PathBuf {
    let mut name = index.as_os_str().to_os_string();
    name.push(".fingerprint");
    PathBuf::from(name)
}

impl Arguments {
    fn parse(values: &[String]) -> anyhow::Result<Self> {
        let manifest = if let Some(manifest) = value_after(values, "--manifest") {
            PathBuf::from(manifest)
        } else {
            let models_dir = required_path(values, "--models-dir")?;
            pack::active_manifest(&models_dir, value_after(values, "--pack-id"))?
        };
        let index = required_path(values, "--index")?;
        let runtime = runtime_path(values)?;
        let query = value_after(values, "--query").map(String::from);
        let serve = values.iter().any(|value| value == "--serve");
        let build_index = values.iter().any(|value| value == "--build-index");
        let progress_json = values.iter().any(|value| value == "--progress-json");
        ensure!(
            serve || build_index || query.is_some(),
            "use --serve, --build-index, or --query <natural language>"
        );
        ensure!(!(serve && query.is_some()), "--serve and --query are mutually exclusive");
        ensure!(
            !(build_index && (serve || query.is_some())),
            "--build-index cannot be combined with search modes"
        );
        ensure!(
            query.as_ref().is_none_or(|value| !value.trim().is_empty()),
            "query must not be empty"
        );
        let top_k = number_after(values, "--top-k")?.unwrap_or(8);
        let threads = number_after(values, "--threads")?.unwrap_or(4);
        ensure!(top_k > 0, "--top-k must be positive");
        ensure!(threads > 0, "--threads must be positive");
        Ok(Self {
            manifest,
            index,
            runtime,
            query,
            serve,
            build_index,
            progress_json,
            top_k,
            threads,
        })
    }
}

impl SemanticIndex {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path).with_context(|| format!("open index {}", path.display()))?;
        let mut reader = IndexReader::new(BufReader::new(file))?;
        let dimensions = reader.header().dimensions as usize;
        let count = usize::try_from(reader.header().count)
            .context("semantic index entry count is too large")?;
        ensure!(count > 0, "semantic index is empty");
        let value_count =
            count.checked_mul(dimensions).context("semantic index dimensions overflow")?;
        let mut keys = Vec::new();
        keys.try_reserve_exact(count).context("semantic index entry count is too large")?;
        let mut fingerprints = Vec::new();
        fingerprints.try_reserve_exact(count).context("semantic index entry count is too large")?;
        let mut embeddings = Vec::new();
        embeddings
            .try_reserve_exact(value_count)
            .context("semantic index dimensions are too large")?;
        for _ in 0..count {
            let entry = reader.read_entry_into(&mut embeddings)?;
            keys.push(entry.key);
            fingerprints.push(entry.fingerprint);
        }
        let model = reader.finish()?.model;
        let groups = group_keys(&keys);
        Ok(Self { model, keys, fingerprints, groups, embeddings, dimensions })
    }

    #[cfg(test)]
    fn rank(&self, query: &[f32], top_k: usize) -> Vec<(&str, f32)> {
        self.rank_excluding(query, top_k, None)
    }

    fn rank_excluding(
        &self,
        query: &[f32],
        top_k: usize,
        exclusion: Option<(&[f32], f32)>,
    ) -> Vec<(&str, f32)> {
        if top_k == 0 || self.groups.is_empty() {
            return Vec::new();
        }
        let mut ranked: Vec<_> = self
            .groups
            .iter()
            .filter_map(|(key, items)| {
                if let Some((negative, threshold)) = exclusion
                    && self.group_score(items, negative) >= threshold
                {
                    return None;
                }
                let score = self.group_score(items, query);
                Some((key.as_str(), score))
            })
            .collect();
        let limit = top_k.min(ranked.len());
        if limit == 0 {
            return Vec::new();
        }
        ranked.select_nth_unstable_by(limit - 1, |left, right| right.1.total_cmp(&left.1));
        ranked.truncate(limit);
        ranked.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        ranked
    }

    fn group_score(&self, items: &[usize], query: &[f32]) -> f32 {
        items.iter().fold(f32::NEG_INFINITY, |best, item| {
            let start = item * self.dimensions;
            let candidate = &self.embeddings[start..start + self.dimensions];
            let score = query.iter().zip(candidate).map(|(left, right)| left * right).sum();
            best.max(score)
        })
    }

    fn exclusion_threshold(&self, query: &[f32]) -> Option<f32> {
        if self.groups.len() < 4 {
            return None;
        }
        let mut scores = self
            .groups
            .iter()
            .map(|(_, items)| self.group_score(items, query))
            .filter(|score| score.is_finite())
            .collect::<Vec<_>>();
        if scores.len() < 4 {
            return None;
        }
        scores.sort_unstable_by(f32::total_cmp);
        let mut low = scores[scores.len() / 4];
        let mut high = scores[scores.len() * 3 / 4];
        if high <= low {
            return None;
        }
        for _ in 0..16 {
            let midpoint = (low + high) * 0.5;
            let mut low_sum = 0.0;
            let mut low_count = 0usize;
            let mut high_sum = 0.0;
            let mut high_count = 0usize;
            for score in &scores {
                if *score < midpoint {
                    low_sum += score;
                    low_count += 1;
                } else {
                    high_sum += score;
                    high_count += 1;
                }
            }
            if low_count == 0 || high_count == 0 {
                return None;
            }
            let next_low = low_sum / low_count as f32;
            let next_high = high_sum / high_count as f32;
            if (next_low - low).abs() + (next_high - high).abs() < 1e-6 {
                low = next_low;
                high = next_high;
                break;
            }
            low = next_low;
            high = next_high;
        }
        Some((low + high) * 0.5)
    }

    fn unique_items(&self) -> usize {
        self.groups.len()
    }
}

fn group_keys(keys: &[String]) -> Vec<(String, Vec<usize>)> {
    let mut positions = HashMap::with_capacity(keys.len());
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (item, key) in keys.iter().enumerate() {
        let group = *positions.entry(key.as_str()).or_insert_with(|| {
            groups.push((key.clone(), Vec::new()));
            groups.len() - 1
        });
        groups[group].1.push(item);
    }
    groups
}

fn load_manifest(path: &Path) -> anyhow::Result<Manifest> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read semantic manifest {}", path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse semantic manifest {}", path.display()))?;
    ensure!(manifest.format == 1, "unsupported semantic model-pack format");
    ensure!(!manifest.id.trim().is_empty(), "semantic model id is empty");
    ensure!(!manifest.version.trim().is_empty(), "semantic model version is empty");
    ensure!(manifest.dimensions > 0, "semantic model dimensions must be positive");
    ensure!(manifest.context_length > 1, "semantic context length must exceed one");
    ensure!(
        manifest.image.std.iter().all(|value| value.is_finite() && *value > 0.0),
        "semantic image standard deviations must be positive"
    );
    match (manifest.image.patch_size, manifest.image.max_num_patches) {
        (None, None) => ensure!(
            manifest.image.width > 0 && manifest.image.height > 0,
            "semantic image dimensions must be positive"
        ),
        (Some(patch_size), Some(max_num_patches)) => {
            ensure!(patch_size > 0 && max_num_patches > 0, "invalid patch image configuration");
            ensure!(
                manifest.image.pixel_attention_mask.is_some()
                    && manifest.image.spatial_shapes.is_some(),
                "patch image inputs are incomplete"
            );
            ensure!(!manifest.image.supported_shapes.is_empty(), "patch image shapes are empty");
            ensure!(
                manifest.image.supported_shapes.iter().all(|[height, width]| {
                    *height > 0
                        && *width > 0
                        && height
                            .checked_mul(*width)
                            .is_some_and(|patches| patches <= max_num_patches)
                }),
                "invalid supported patch image shape"
            );
        }
        _ => anyhow::bail!("patch image configuration is incomplete"),
    }
    Ok(manifest)
}

fn tokenize(
    tokenizer: &TextTokenizer,
    query: &str,
    context: usize,
    eos_token: u32,
    pad_token: u32,
) -> anyhow::Result<(Vec<i64>, Vec<i64>)> {
    let (mut values, mut attention_mask) = tokenizer.encode(query)?;
    if values.len() > context {
        values.truncate(context);
        attention_mask.truncate(context);
        values[context - 1] = i64::from(eos_token);
        attention_mask[context - 1] = 1;
    }
    values.resize(context, i64::from(pad_token));
    attention_mask.resize(context, 0);
    Ok((values, attention_mask))
}

fn normalize(embedding: &mut [f32]) -> anyhow::Result<()> {
    let length = embedding.iter().map(|value| value * value).sum::<f32>().sqrt();
    ensure!(length.is_finite() && length > f32::EPSILON, "empty semantic embedding");
    for value in embedding {
        *value /= length;
    }
    Ok(())
}

fn default_image_input() -> String {
    String::from("images")
}

fn default_image_output() -> String {
    String::from("image_embedding")
}

fn default_text_input() -> String {
    String::from("tokens")
}

fn default_text_output() -> String {
    String::from("text_embedding")
}

const fn default_mask_padding() -> bool {
    true
}

fn default_eos_token() -> u32 {
    49_407
}

fn runtime_path(arguments: &[String]) -> anyhow::Result<PathBuf> {
    runtime_path_with(arguments, |name| std::env::var_os(name))
}

fn runtime_path_with(
    arguments: &[String],
    mut environment: impl FnMut(&str) -> Option<std::ffi::OsString>,
) -> anyhow::Result<PathBuf> {
    value_after(arguments, "--runtime")
        .map(PathBuf::from)
        .or_else(|| environment("SKWD_LENS_ORT_DYLIB").map(PathBuf::from))
        .or_else(|| environment("SKWD_SEMANTIC_ORT_DYLIB").map(PathBuf::from))
        .context("use --runtime <libonnxruntime.so> or SKWD_LENS_ORT_DYLIB")
}

fn required_path(arguments: &[String], flag: &str) -> anyhow::Result<PathBuf> {
    value_after(arguments, flag).map(PathBuf::from).with_context(|| format!("use {flag} <path>"))
}

fn number_after(arguments: &[String], flag: &str) -> anyhow::Result<Option<usize>> {
    value_after(arguments, flag)
        .map(|value| value.parse().with_context(|| format!("invalid {flag} value: {value}")))
        .transpose()
}

fn value_after<'a>(arguments: &'a [String], flag: &str) -> Option<&'a str> {
    arguments
        .iter()
        .position(|argument| argument == flag)
        .and_then(|position| arguments.get(position + 1))
        .map(String::as_str)
}

fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() { path.to_path_buf() } else { root.join(path) }
}

fn memory_mib() -> (Option<f64>, Option<f64>) {
    let Ok(status) = std::fs::read_to_string("/proc/self/smaps_rollup") else {
        return (None, None);
    };
    let value = |field: &str| {
        status.lines().find_map(|line| {
            let (name, rest) = line.split_once(':')?;
            (name == field)
                .then(|| rest.split_whitespace().next()?.parse::<f64>().ok())
                .flatten()
                .map(|kib| kib / 1_024.0)
        })
    };
    (value("Rss"), value("Pss"))
}

#[cfg(test)]
mod pack_tests;

#[cfg(test)]
mod tests;
