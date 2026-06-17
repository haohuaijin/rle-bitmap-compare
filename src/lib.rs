//! Shared workload definitions for the FTS access-plan benchmark.
//!
//! Both binaries (`generate`, `query`) import from this crate so the
//! synthetic dataset and the access plan rebuilt against it can't drift.
//!
//! Module overview:
//!   - this file: workload constants, [`Pattern`], [`schema`], [`file_path`].
//!   - [`access_plan`]: build a per-file [`ParquetAccessPlan`] in either
//!     `Vec<RowSelector>` (RLE) or `BooleanBuffer` (bitmap) form.
//!   - [`runner`]:      run the benchmark SQL through DataFusion and attach
//!     the access plans to the physical plan.
//!   - [`timing`]:      warmup + N timed iterations for a closure.
//!
//! [`ParquetAccessPlan`]:
//!   datafusion_datasource_parquet::ParquetAccessPlan

pub mod access_plan;
pub mod runner;
pub mod timing;

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema};

/// Hard-coded so the table numbers are comparable across runs.
pub const ROWS_PER_FILE: usize = 5_000_000;
pub const ROW_GROUP_SIZE: usize = 128 * 1024;

/// Mean run length for the clustered pattern (mirrors OpenObserve's
/// `CLUSTER_BLOCK` / arrow's default batch). Per-run length is jittered in
/// `[BASE_RUN * 3/4, BASE_RUN * 5/4)`.
pub const BASE_RUN: u64 = 8192;
/// One jittered select run per `SUPER_BLOCK` rows → `BASE_RUN / SUPER_BLOCK
/// = 1/3` mean selectivity.
pub const SUPER_BLOCK: u64 = 3 * BASE_RUN;

pub const HIT_TOKEN: &str = "svc-c3500";
pub const HIT_TOKEN_UPPER: &str = "SVC-C3500";
pub const MISS_TOKEN: &str = "svc-other";

/// The two synthetic hit patterns the benchmark contrasts.
#[derive(Clone, Copy, Debug)]
pub enum Pattern {
    /// `i % 3 == 0` — every 3rd row hits → worst-case `select(1)/skip(2)/...`.
    Fragmented,
    /// One jittered ~`BASE_RUN` run per `SUPER_BLOCK` rows → long select runs
    /// at irregular offsets, like real log-like data sorted by tag.
    Clustered,
}

impl Pattern {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "fragmented" => Ok(Self::Fragmented),
            "clustered" => Ok(Self::Clustered),
            _ => Err(format!("unknown pattern: {s} (fragmented|clustered)")),
        }
    }

    /// Body column that carries this pattern's `HIT_TOKEN`.
    pub fn body_column(self) -> &'static str {
        match self {
            Self::Fragmented => "body_fragmented",
            Self::Clustered => "body_clustered",
        }
    }

    /// One-line description shown in the header.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Fragmented => "fragmented (i%3==0)",
            Self::Clustered => "clustered (jittered runs, ~8K hits per 24K rows)",
        }
    }

    /// Whether row `i` carries `HIT_TOKEN` in this pattern's body column.
    pub fn is_hit(self, i: u64) -> bool {
        match self {
            Self::Fragmented => i.is_multiple_of(3),
            Self::Clustered => {
                let sb = i / SUPER_BLOCK;
                let p = i % SUPER_BLOCK;
                let h = splitmix64(sb ^ 0xCAFE_BABE_5EED_F00D);
                let start = h % BASE_RUN; // [0, BASE_RUN)
                let len = BASE_RUN * 3 / 4 + ((h >> 32) % (BASE_RUN / 2));
                p >= start && p < start + len
            }
        }
    }
}

#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// `service_name` (narrow-projection target) + one body column per pattern.
/// For the same `i`, `body_fragmented` and `body_clustered` are materialized
/// side by side so one dataset drives both query runs.
pub fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("service_name", DataType::Utf8, false),
        Field::new("body_fragmented", DataType::Utf8, false),
        Field::new("body_clustered", DataType::Utf8, false),
    ]))
}

/// Fixed dataset location — keeps both examples and `/usr/bin/time -l`
/// invocations pointing at the same parquet files without a CLI flag.
pub const DATA_DIR: &str = "data-fts";

pub fn file_path(f: usize) -> std::path::PathBuf {
    std::path::PathBuf::from(DATA_DIR).join(format!("part-{f:04}.parquet"))
}
