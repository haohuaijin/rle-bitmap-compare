//! Run one strategy (`rle` / `bitmap` / `no-index`) against one pattern
//! (`fragmented` / `clustered`). Prints min/mean/max exec time for that
//! single strategy; meant to be invoked under `/usr/bin/time -l` so peak
//! RSS attributes to one code path.
//!
//! Usage:
//!   cargo run --release --bin query -- \
//!       --pattern fragmented|clustered --strategy rle|bitmap|no-index \
//!       [--files N] [--threads N] [--repeats N]
//!
//! Heavy lifting lives in the library:
//!   rle_bitmap_compare::access_plan — build RLE / bitmap ParquetAccessPlan.
//!   rle_bitmap_compare::runner      — run the SQL via DataFusion.
//!   rle_bitmap_compare::timing      — warmup + N timed iterations.

use std::time::Instant;

use rle_bitmap_compare::{
    Pattern, ROWS_PER_FILE,
    access_plan::{PlanRepr, build_plans},
    file_path,
    runner::{PlanList, ResultRow, run_query},
    timing::{ExecStats, time_runs},
};
use tokio::runtime::Runtime;

// ============================================================================
// CLI
// ============================================================================

#[derive(Clone, Copy)]
enum Strategy {
    IndexRle,
    IndexBitmap,
    NoIndex,
}

impl Strategy {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "rle" => Ok(Self::IndexRle),
            "bitmap" => Ok(Self::IndexBitmap),
            "no-index" => Ok(Self::NoIndex),
            _ => Err(format!("unknown --strategy: {s} (rle|bitmap|no-index)")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::IndexRle => "with FTS index (RLE)",
            Self::IndexBitmap => "with FTS index (bitmap)",
            Self::NoIndex => "no index (post-filter)",
        }
    }
}

struct Args {
    files: usize,
    threads: usize,
    /// Timed iterations (plus one untimed warmup).
    repeats: usize,
    pattern: Pattern,
    strategy: Strategy,
}

fn parse_args() -> Args {
    let mut files = 40;
    let mut threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut repeats = 5;
    let mut pattern = None;
    let mut strategy = None;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--files" => files = it.next().unwrap().parse().unwrap(),
            "--threads" => threads = it.next().unwrap().parse().unwrap(),
            "--repeats" => repeats = it.next().unwrap().parse().unwrap(),
            "--pattern" => pattern = Some(Pattern::parse(&it.next().unwrap()).unwrap()),
            "--strategy" => strategy = Some(Strategy::parse(&it.next().unwrap()).unwrap()),
            "-h" | "--help" => {
                eprintln!(
                    "usage: query --pattern fragmented|clustered \
                     --strategy rle|bitmap|no-index \
                     [--files N] [--threads N] [--repeats N]"
                );
                std::process::exit(0);
            }
            _ => panic!("unknown arg: {arg}"),
        }
    }
    Args {
        files,
        threads,
        repeats,
        pattern: pattern.expect("--pattern is required (fragmented|clustered)"),
        strategy: strategy.expect("--strategy is required (rle|bitmap|no-index)"),
    }
}

// ============================================================================
// main
// ============================================================================

fn main() {
    let args = parse_args();
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.threads)
        .enable_all()
        .build()
        .unwrap();
    for f in 0..args.files {
        let p = file_path(f);
        assert!(p.exists(), "missing {} — run `generate` first", p.display());
    }

    print_header(&args);
    let (rows, selectors_col, plan_ms, exec) = match args.strategy {
        Strategy::IndexRle => measure_indexed(&rt, &args, PlanRepr::Rle),
        Strategy::IndexBitmap => measure_indexed(&rt, &args, PlanRepr::Bitmap),
        Strategy::NoIndex => measure_no_index(&rt, &args),
    };
    print_row(
        args.strategy.label(),
        &selectors_col,
        plan_ms,
        &exec,
        rows.len(),
    );
    print_topk(&rows);
}

fn measure_indexed(
    rt: &Runtime,
    args: &Args,
    repr: PlanRepr,
) -> (Vec<ResultRow>, String, f64, ExecStats) {
    let t = Instant::now();
    let built = build_plans(args.files, args.pattern, repr);
    let plan_ms = t.elapsed().as_secs_f64() * 1e3;
    let selectors_col = match repr {
        PlanRepr::Rle => built.iter().map(|p| p.selectors).sum::<usize>().to_string(),
        PlanRepr::Bitmap => "-".into(),
    };
    let by_file: PlanList = built.into_iter().map(|p| Some(p.plan)).collect();
    let (rows, exec) = time_runs(rt, args.repeats, || {
        run_query(args.pattern, args.threads, true, Some(&by_file))
    });
    (rows, selectors_col, plan_ms, exec)
}

fn measure_no_index(rt: &Runtime, args: &Args) -> (Vec<ResultRow>, String, f64, ExecStats) {
    let (rows, exec) = time_runs(rt, args.repeats, || {
        run_query(args.pattern, args.threads, false, None)
    });
    (rows, "-".into(), 0.0, exec)
}

// ============================================================================
// formatting
// ============================================================================

fn print_header(args: &Args) {
    println!(
        "dataset : {} files x {} rows ({} total), threads={}",
        args.files,
        ROWS_PER_FILE,
        args.files * ROWS_PER_FILE,
        args.threads,
    );
    println!("pattern : {}", args.pattern.describe());
    println!("strategy: {}", args.strategy.label());
    println!(
        "timing  : 1 warmup + {} timed iterations (exec ms: min / mean / max)\n",
        args.repeats,
    );
    println!(
        "{:<28} | {:>13} | {:>9} | {:>30} | {:>8}",
        "strategy", "selectors", "plan(ms)", "exec(ms) min / mean / max", "groups",
    );
    println!("{}", "-".repeat(103));
}

fn print_row(label: &str, selectors: &str, plan_ms: f64, exec: &ExecStats, groups: usize) {
    let exec_col = format!("{:>7.1} / {:>7.1} / {:>7.1}", exec.min, exec.mean, exec.max);
    println!("{label:<28} | {selectors:>13} | {plan_ms:>9.1} | {exec_col:>30} | {groups:>8}",);
}

fn print_topk(rows: &[ResultRow]) {
    println!("\ntop-10 (asc by cnt):");
    println!("  {:<16} | {:>12}", "service_name", "cnt");
    println!("  {}", "-".repeat(33));
    for (svc, cnt) in rows {
        println!("  {svc:<16} | {cnt:>12}");
    }
}
