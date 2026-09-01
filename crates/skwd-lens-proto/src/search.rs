use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SearchRequest {
    pub generation: u64,
    pub query: String,
    #[serde(default)]
    pub negative_query: Option<String>,
    #[serde(default = "default_negative_weight")]
    pub negative_weight: f32,
    pub top_k: usize,
    #[serde(default)]
    pub score_window: Option<f32>,
    #[serde(default)]
    pub min_score_prominence: Option<f32>,
    #[serde(default)]
    pub max_results: Option<usize>,
    #[serde(default)]
    pub min_results: usize,
    #[serde(default)]
    pub embedding_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Match {
    pub rank: usize,
    pub key: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SearchResponse {
    pub generation: u64,
    pub query_ms: f64,
    pub search_ms: f64,
    pub matches: Vec<Match>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    pub error: Option<String>,
}

impl SearchResponse {
    #[must_use]
    pub fn from_matches(
        generation: u64,
        matches: Vec<Match>,
        query_ms: f64,
        search_ms: f64,
    ) -> Self {
        Self { query_ms, search_ms, matches, ..Self::empty(generation) }
    }

    #[must_use]
    pub fn from_embedding(generation: u64, embedding: Vec<f32>, query_ms: f64) -> Self {
        Self { query_ms, embedding: Some(embedding), ..Self::empty(generation) }
    }

    pub fn failed(generation: u64, error: &impl std::fmt::Display) -> Self {
        Self { error: Some(error.to_string()), ..Self::empty(generation) }
    }

    fn empty(generation: u64) -> Self {
        Self {
            generation,
            query_ms: 0.0,
            search_ms: 0.0,
            matches: Vec::new(),
            embedding: None,
            error: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct BuildRequest {
    pub fingerprint: u64,
    pub entries: Vec<BuildEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BuildEntry {
    pub key: String,
    pub path: PathBuf,
    pub fingerprint: u64,
    #[serde(default, skip_serializing_if = "ImageView::is_full")]
    pub view: ImageView,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ImageView {
    #[default]
    Full,
    Center,
    LeftThird,
    RightThird,
}

impl ImageView {
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn is_full(&self) -> bool {
        matches!(self, Self::Full)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct BuildProgress {
    pub progress: usize,
    pub total: usize,
    pub detail: String,
}

const fn default_negative_weight() -> f32 {
    0.5
}

#[cfg(test)]
#[path = "search_tests.rs"]
mod tests;
