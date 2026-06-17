//! Build a per-file [`ParquetAccessPlan`] in one of two row-id encodings.
//!
//! - [`PlanRepr::Rle`]    — `Vec<RowSelector>` (the original arrow-rs path).
//!   Cost is O(matched rows) for scattered hits, O(runs) for clustered.
//! - [`PlanRepr::Bitmap`] — `BooleanBuffer` via the patched
//!   `RowSelection::from_boolean_buffer`. Constant 1 bit per row,
//!   selectivity-independent.

use datafusion::{
    arrow::array::builder::BooleanBufferBuilder,
    datasource::physical_plan::parquet::ParquetAccessPlan,
    parquet::arrow::arrow_reader::{RowSelection, RowSelector},
};
use rayon::prelude::*;

use crate::{Pattern, ROW_GROUP_SIZE, ROWS_PER_FILE};

#[derive(Clone, Copy)]
pub enum PlanRepr {
    Rle,
    Bitmap,
}

pub struct BuiltPlan {
    pub plan: ParquetAccessPlan,
    /// Total `RowSelector` count (RLE only; 0 for bitmap).
    pub selectors: usize,
}

/// Build one `ParquetAccessPlan` per file, in parallel across files.
pub fn build_plans(files: usize, pat: Pattern, repr: PlanRepr) -> Vec<BuiltPlan> {
    (0..files)
        .into_par_iter()
        .map(|f| {
            let base = (f * ROWS_PER_FILE) as u64;
            let ids: Vec<u32> = (0..ROWS_PER_FILE as u64)
                .filter(|k| pat.is_hit(base + k))
                .map(|k| k as u32)
                .collect();
            match repr {
                PlanRepr::Rle => build_rle(&ids),
                PlanRepr::Bitmap => build_bitmap(&ids),
            }
        })
        .collect()
}

/// Walk sorted `ids` one row group at a time. Calls
/// `f(rg, rg_start, rg_end, ids_in_rg)` for each row group that has at
/// least one matching id.
fn for_each_row_group(ids: &[u32], mut f: impl FnMut(usize, usize, usize, &[u32])) {
    let mut i = 0;
    while i < ids.len() {
        let rg = ids[i] as usize / ROW_GROUP_SIZE;
        let rg_start = rg * ROW_GROUP_SIZE;
        let rg_end = (rg_start + ROW_GROUP_SIZE).min(ROWS_PER_FILE);
        let j = i;
        while i < ids.len() && (ids[i] as usize) < rg_end {
            i += 1;
        }
        f(rg, rg_start, rg_end, &ids[j..i]);
    }
}

fn build_rle(ids: &[u32]) -> BuiltPlan {
    let rgs = ROWS_PER_FILE.div_ceil(ROW_GROUP_SIZE);
    let mut plan = ParquetAccessPlan::new_none(rgs);
    let mut selectors = 0;
    for_each_row_group(ids, |rg, rg_start, rg_end, ids_in_rg| {
        let runs = rle_runs(rg_start, rg_end, ids_in_rg);
        selectors += runs.len();
        plan.scan(rg);
        plan.scan_selection(rg, RowSelection::from(runs));
    });
    BuiltPlan { plan, selectors }
}

/// Convert sorted `ids` inside one row group into `select(N)/skip(N)` runs.
fn rle_runs(rg_start: usize, rg_end: usize, ids: &[u32]) -> Vec<RowSelector> {
    let mut out = Vec::new();
    let mut cursor = rg_start;
    let mut i = 0;
    while i < ids.len() {
        let run_start = ids[i] as usize;
        let mut run_end = run_start + 1;
        i += 1;
        while i < ids.len() && (ids[i] as usize) == run_end && run_end < rg_end {
            run_end += 1;
            i += 1;
        }
        if run_start > cursor {
            out.push(RowSelector::skip(run_start - cursor));
        }
        out.push(RowSelector::select(run_end - run_start));
        cursor = run_end;
    }
    if cursor < rg_end {
        out.push(RowSelector::skip(rg_end - cursor));
    }
    out
}

fn build_bitmap(ids: &[u32]) -> BuiltPlan {
    let rgs = ROWS_PER_FILE.div_ceil(ROW_GROUP_SIZE);
    let mut plan = ParquetAccessPlan::new_none(rgs);
    for_each_row_group(ids, |rg, rg_start, rg_end, ids_in_rg| {
        let rg_rows = rg_end - rg_start;
        let mut b = BooleanBufferBuilder::new(rg_rows);
        b.append_n(rg_rows, false);
        for &id in ids_in_rg {
            b.set_bit(id as usize - rg_start, true);
        }
        plan.scan(rg);
        plan.scan_selection(rg, RowSelection::from_boolean_buffer(b.finish()));
    });
    BuiltPlan { plan, selectors: 0 }
}
