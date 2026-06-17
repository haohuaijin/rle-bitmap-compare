//! Run the benchmark SQL through DataFusion and (optionally) attach
//! pre-built per-file [`ParquetAccessPlan`]s to the physical plan.

use std::sync::Arc;

use datafusion::{
    arrow::array::{Int64Array, StringArray},
    catalog::memory::DataSourceExec,
    common::tree_node::{Transformed, TreeNode},
    datasource::physical_plan::{FileGroup, FileScanConfig, parquet::ParquetAccessPlan},
    physical_plan::{ExecutionPlan, execute_stream},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use futures::StreamExt;

use crate::{DATA_DIR, HIT_TOKEN_UPPER, Pattern, schema};

const BATCH_SIZE: usize = 8192;

/// One [`ParquetAccessPlan`] per file (or `None` if that file has no hits).
pub type PlanList = Vec<Option<ParquetAccessPlan>>;
/// One result row of the benchmark SQL — `(service_name, count)`.
pub type ResultRow = (String, i64);

/// Run the benchmark query once.
///
/// `with_index = true` drops the `WHERE` clause (the access plan answers it)
/// and expects `plans` to be `Some(&PlanList)`. `with_index = false` keeps the
/// `WHERE`, disables `pushdown_filters`, and ignores `plans` — this is the
/// no-index cost ceiling.
pub async fn run_query(
    pattern: Pattern,
    threads: usize,
    with_index: bool,
    plans: Option<&PlanList>,
) -> Vec<ResultRow> {
    let ctx = make_ctx(threads).await;
    let sql = build_sql(pattern, with_index);
    let df = ctx.sql(&sql).await.expect("plan sql");
    let mut plan = df.create_physical_plan().await.expect("physical plan");
    if let Some(plans) = plans {
        plan = attach_access_plans(plan, plans);
    }
    collect_rows(plan, &ctx).await
}

async fn make_ctx(threads: usize) -> SessionContext {
    let mut sc = SessionConfig::new()
        .with_batch_size(BATCH_SIZE)
        .with_target_partitions(threads);
    sc.options_mut().execution.parquet.pushdown_filters = false;
    let ctx = SessionContext::new_with_config(sc);
    let schema = schema();
    let opts = ParquetReadOptions::default().schema(&schema);
    ctx.register_parquet("t", DATA_DIR, opts)
        .await
        .expect("register parquet dir");
    ctx
}

fn build_sql(pattern: Pattern, with_index: bool) -> String {
    if with_index {
        // The index answered the predicate — drop the WHERE.
        "SELECT service_name, COUNT(*) AS cnt FROM t \
         GROUP BY service_name ORDER BY cnt LIMIT 10"
            .to_string()
    } else {
        format!(
            "SELECT service_name, COUNT(*) AS cnt FROM t \
             WHERE {col} ILIKE '%{HIT_TOKEN_UPPER}%' \
             GROUP BY service_name ORDER BY cnt LIMIT 10",
            col = pattern.body_column(),
        )
    }
}

async fn collect_rows(plan: Arc<dyn ExecutionPlan>, ctx: &SessionContext) -> Vec<ResultRow> {
    let mut stream = execute_stream(plan, ctx.task_ctx()).expect("execute");
    let mut out = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = batch.expect("batch");
        let svc = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let cnt = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for r in 0..batch.num_rows() {
            out.push((svc.value(r).to_string(), cnt.value(r)));
        }
    }
    out
}

/// Walk the physical plan; on each parquet `DataSourceExec` attach the per-file
/// [`ParquetAccessPlan`] to its `FileScanConfig` extension.
///
/// Datafusion 54+ keys file extensions by their concrete Rust type, so the
/// access plan must be passed by **value**, not as `Arc<ParquetAccessPlan>` —
/// otherwise the parquet reader's `extension::<ParquetAccessPlan>()` lookup
/// returns `None` and the plan is silently ignored.
fn attach_access_plans(plan: Arc<dyn ExecutionPlan>, plans: &PlanList) -> Arc<dyn ExecutionPlan> {
    plan.transform_down(|node| {
        let Some(dse) = node.downcast_ref::<DataSourceExec>() else {
            return Ok(Transformed::no(node));
        };
        let Some(config) = dse.data_source().downcast_ref::<FileScanConfig>() else {
            return Ok(Transformed::no(node));
        };
        let mut config = config.clone();
        config.file_groups = config
            .file_groups
            .iter()
            .map(|group| {
                FileGroup::new(
                    group
                        .iter()
                        .map(|file| {
                            let mut file = file.clone();
                            let id = file_id(file.object_meta.location.as_ref());
                            if let Some(Some(p)) = plans.get(id) {
                                file = file.with_extension(p.clone());
                            }
                            file
                        })
                        .collect(),
                )
            })
            .collect();
        Ok(Transformed::yes(
            Arc::new(DataSourceExec::new(Arc::new(config))) as Arc<dyn ExecutionPlan>,
        ))
    })
    .expect("attach access plans")
    .data
}

fn file_id(path: &str) -> usize {
    path.rsplit('/')
        .next()
        .unwrap_or(path)
        .trim_start_matches("part-")
        .trim_end_matches(".parquet")
        .parse()
        .expect("file id from path")
}
