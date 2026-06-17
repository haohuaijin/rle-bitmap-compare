//! Generate a parquet dataset that carries BOTH hit patterns side by side as
//! two body columns. One dataset serves `--pattern fragmented` and
//! `--pattern clustered` runs of the `query` example.
//!
//! Schema (3 cols):
//!   service_name     Utf8   dictionary-encoded, narrow-projection target
//!   body_fragmented  Utf8   ~300B/row, `SVC-C3500` on every 3rd row
//!   body_clustered   Utf8   ~300B/row, `SVC-C3500` in jittered ~8K runs
//!
//! Usage:
//!   cargo run --release --example generate -- [--files N] [--threads N]

use std::{fs::File, path::Path, sync::Arc, time::Instant};

use datafusion::{
    arrow::array::{ArrayRef, RecordBatch, StringArray},
    parquet::{arrow::ArrowWriter, basic::Compression, file::properties::WriterProperties},
};
use rayon::prelude::*;

use rle_bitmap_compare::{
    DATA_DIR, HIT_TOKEN, MISS_TOKEN, Pattern, ROW_GROUP_SIZE, ROWS_PER_FILE, file_path, schema,
    splitmix64,
};

const TS_BASE: i64 = 1_760_000_000_000_000;
/// 50 service names with a triangular distribution — bucket 0 most common,
/// 49 rarest. Makes `GROUP BY service_name ORDER BY cnt LIMIT 10` interesting.
const SERVICE_BUCKETS: u64 = 50;
/// One Arrow `RecordBatch` per call to `build_batch`. Keeps each batch's
/// string allocations bounded.
const BATCH_ROWS: usize = 131_072;

fn service_name(i: u64) -> String {
    let total = SERVICE_BUCKETS * (SERVICE_BUCKETS + 1) / 2;
    let mut h = splitmix64(i ^ 0xD1CE_5EED_C0DE_BEEF) % total;
    for b in 0..SERVICE_BUCKETS {
        let weight = SERVICE_BUCKETS - b;
        if h < weight {
            return format!("svc-{b:02}");
        }
        h -= weight;
    }
    unreachable!()
}

/// ~300B pseudo-log line; `hit` controls whether `HIT_TOKEN` is embedded.
fn body(i: u64, hit: bool) -> String {
    let trace = splitmix64(i ^ 0x7AC3_7AC3_7AC3_7AC3);
    let span = splitmix64(i ^ 0x0123_4567_89AB_CDEF) & 0xFFFF_FFFF;
    let latency = splitmix64(i ^ 0x1111_2222_3333_4444) % 2000;
    let tag = if hit { HIT_TOKEN } else { MISS_TOKEN }.to_uppercase();
    let ts = TS_BASE + i as i64 * 137;
    format!(
        "{ts} info [{tag}] GET /api/v1/object/list?offset={off} status=200 \
         latency={latency}ms trace_id={trace:016x} span_id={span:08x} \
         caller=handler/object.go:188 msg=\"request finished successfully, \
         cache=miss backend=primary retry=0\"",
        off = i % 1000,
    )
}

fn build_batch(start: u64, n: usize) -> RecordBatch {
    let rows = || start..start + n as u64;
    let svc: StringArray = rows().map(service_name).collect::<Vec<_>>().into();
    let body_f: StringArray = rows()
        .map(|i| body(i, Pattern::Fragmented.is_hit(i)))
        .collect::<Vec<_>>()
        .into();
    let body_c: StringArray = rows()
        .map(|i| body(i, Pattern::Clustered.is_hit(i)))
        .collect::<Vec<_>>()
        .into();
    let cols: Vec<ArrayRef> = vec![Arc::new(svc), Arc::new(body_f), Arc::new(body_c)];
    RecordBatch::try_new(schema(), cols).unwrap()
}

fn writer_props() -> WriterProperties {
    WriterProperties::builder()
        .set_write_batch_size(8192)
        .set_max_row_group_row_count(Some(ROW_GROUP_SIZE))
        .set_compression(Compression::ZSTD(Default::default()))
        .build()
}

/// Write one parquet file if it doesn't already exist (atomic: tmp + rename).
fn write_file(path: &Path, file_idx: usize) {
    if path.exists() {
        return;
    }
    let tmp = path.with_extension("parquet.tmp");
    let mut writer = ArrowWriter::try_new(
        File::create(&tmp).expect("create parquet tmp"),
        schema(),
        Some(writer_props()),
    )
    .unwrap();

    let base = (file_idx * ROWS_PER_FILE) as u64;
    let mut written = 0;
    while written < ROWS_PER_FILE {
        let n = BATCH_ROWS.min(ROWS_PER_FILE - written);
        writer
            .write(&build_batch(base + written as u64, n))
            .unwrap();
        written += n;
    }
    writer.close().unwrap();
    std::fs::rename(&tmp, path).unwrap();
}

struct Args {
    files: usize,
    threads: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        files: 40,
        threads: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--files" => a.files = it.next().unwrap().parse().unwrap(),
            "--threads" => a.threads = it.next().unwrap().parse().unwrap(),
            "-h" | "--help" => {
                eprintln!("usage: generate [--files N] [--threads N]");
                std::process::exit(0);
            }
            _ => panic!("unknown arg: {arg}"),
        }
    }
    a
}

fn main() {
    let args = parse_args();
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap();

    std::fs::create_dir_all(DATA_DIR).expect("create data dir");
    eprintln!(
        "generating {} files x {} rows (row_group={}) into {DATA_DIR} ...",
        args.files, ROWS_PER_FILE, ROW_GROUP_SIZE,
    );

    let start = Instant::now();
    let done = std::sync::atomic::AtomicUsize::new(0);
    (0..args.files).into_par_iter().for_each(|f| {
        write_file(&file_path(f), f);
        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        eprintln!(
            "  {n}/{} done ({:.0}s)",
            args.files,
            start.elapsed().as_secs_f64()
        );
    });

    let on_disk: u64 = (0..args.files)
        .map(|f| {
            std::fs::metadata(file_path(f))
                .map(|m| m.len())
                .unwrap_or(0)
        })
        .sum();
    eprintln!(
        "done: {} files x {} rows, on-disk {:.2} GB, took {:.0}s",
        args.files,
        ROWS_PER_FILE,
        on_disk as f64 / (1u64 << 30) as f64,
        start.elapsed().as_secs_f64(),
    );
}
