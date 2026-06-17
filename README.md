# rle-bitmap-compare

Micro-benchmark for `ParquetAccessPlan` representations on a synthetic FTS workload — RLE (`Vec<RowSelector>`) vs bitmap (`BooleanBuffer`) vs no index.

The bitmap path needs the patched arrow-rs branch (`export-mask`, exposes `RowSelection::from_boolean_buffer`) and the matching datafusion fork (`use-mask-in-datafusion`) pinned in `Cargo.toml`.

## Workload

- 200 M rows (40 files × 5 M), row group 128 K, hard-coded.
- Columns: `service_name`, `body_fragmented`, `body_clustered` (all Utf8).
- `body_*` is ~300 B per row. The FTS token `SVC-C3500` appears in:
  - `body_fragmented` on every 3rd row → `select(1)/skip(2)/...`
  - `body_clustered` in jittered ~8 K runs (one per 24 K rows).
- SQL (`<col>` = `body_fragmented` or `body_clustered`):
  ```sql
  SELECT service_name, COUNT(*) AS cnt
  FROM t WHERE <col> ILIKE '%SVC-C3500%'
  GROUP BY service_name ORDER BY cnt LIMIT 10
  ```
  The `rle` / `bitmap` strategies drop the `WHERE` (the access plan answers it); `no-index` keeps it and disables `pushdown_filters`.

## Running

```bash
# 1. Build (first run pulls the patched forks, ~3 min).
cargo build --release

# 2. Generate the dataset (~9.8 GB on disk).
cargo run --release --bin generate

# 3. Measure peak heap per strategy.
for pat in fragmented clustered; do
  for st in rle bitmap no-index; do
    echo "=== $pat / $st ==="
    /usr/bin/time -l ./target/release/query --pattern "$pat" --strategy "$st"
  done
done

# 4. Cleanup.
rm -rf data-fts/
```

**peak heap** below = the `peak memory footprint` line from `/usr/bin/time -l` (bytes, macOS).

## Sample numbers (M1-class CPU, defaults)

### `--pattern fragmented` (scattered 1-row hits)

| strategy | selectors | exec mean (ms) | peak heap |
|---|---:|---:|---:|
| `rle` | 133 333 854 | **1441** | **7.95 GB** |
| `bitmap` | — | **481** | **201 MB** |
| `no-index` | — | **4782** | **269 MB** |

### `--pattern clustered` (jittered ~8 K runs)

| strategy | selectors | exec mean (ms) | peak heap |
|---|---:|---:|---:|
| `rle` | 17 834 | **145** | **162 MB** |
| `bitmap` | — | **246** | **193 MB** |
| `no-index` | — | **4530** | **260 MB** |

## Scaling to 500 M rows (`--files 100`)

Same workload, same machine, 2.5× the data (~24 GB on disk). Pass `--files 100` to both `generate` and `query`.

### `--pattern fragmented`

| strategy | selectors | exec mean (ms) | peak heap |
|---|---:|---:|---:|
| `rle` | 333 334 634 | **5848** | **17.10 GB** |
| `bitmap` | — | **1286** | **272 MB** |
| `no-index` | — | **11715** | **307 MB** |

### `--pattern clustered`

| strategy | selectors | exec mean (ms) | peak heap |
|---|---:|---:|---:|
| `rle` | 44 586 | **331** | **189 MB** |
| `bitmap` | — | **565** | **263 MB** |
| `no-index` | — | **10781** | **303 MB** |

Bitmap, no-index, and clustered-RLE all scale roughly linearly (~2.3-2.7× exec for 2.5× data). **Scattered-RLE goes superlinear** — 4.06× exec and 2.15× heap — because the 17 GB plan starts to swamp page cache. At 1 B rows it would OOM on a 32 GB machine; bitmap would still fit in ~500 MB.

### Takeaways

- **Scattered + RLE blows up:** 7.95 GB at 200 M, 17.10 GB at 500 M, OOM-class beyond. One 16-byte `RowSelector` per matched row.
- **Bitmap is selectivity-independent and scales linearly** — ~200 MB at 200 M, ~270 MB at 500 M.
- **Clustered + RLE wins** by ~15-40% on both axes — 8 K runs collapse to a single selector each.
- **No-index is ~10× slower** than the indexed paths regardless of pattern; narrow projection means the access-plan paths never decode `body_*` at all.

**Bottom line:** a bitmap-backed `RowSelection` is the right default — flat memory, no GB cliff, within ~1.7× of RLE on its best case.

## Layout

- `src/lib.rs` — workload constants, `Pattern::is_hit`, `schema()`, `file_path()`.
- `src/access_plan.rs` — build RLE / bitmap `ParquetAccessPlan`.
- `src/runner.rs` — run the SQL via DataFusion; attach access plans to the physical plan.
- `src/timing.rs` — warmup + N timed iterations (`ExecStats`, `time_runs`).
- `src/bin/generate.rs` — rayon-parallel parquet writer.
- `src/bin/query.rs` — CLI + orchestration for one strategy per invocation.
