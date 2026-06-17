//! Warmup + N timed iterations of an async benchmark closure.

use std::time::Instant;

use tokio::runtime::Runtime;

use crate::runner::ResultRow;

pub struct ExecStats {
    pub min: f64,
    pub mean: f64,
    pub max: f64,
}

/// Run `f` once as warmup (timing discarded), then `repeats` times timed.
/// Returns the final result + min/mean/max exec(ms).
pub fn time_runs<F, Fut>(rt: &Runtime, repeats: usize, mut f: F) -> (Vec<ResultRow>, ExecStats)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Vec<ResultRow>>,
{
    let mut last = rt.block_on(f());
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t = Instant::now();
        last = rt.block_on(f());
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let stats = ExecStats {
        min: samples.iter().cloned().fold(f64::INFINITY, f64::min),
        max: samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        mean: samples.iter().sum::<f64>() / samples.len() as f64,
    };
    (last, stats)
}
