use apotheosis3::controllers::apotheosis::Apotheosis;
use apotheosis3::controllers::nndescent::NndParams;
use apotheosis3::controllers::nsg::{BuildConfig, NndInit, NsgParams};
use apotheosis3::datalayer::algorithms::TlshDistance;
use apotheosis3::datalayer::record::{ApotheosisRecord, SimpleTlshRecord};
use clap::{Parser, ValueEnum};
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;
use tlsh2::TlshDefault;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InitArg {
    Random,
    MetricTree,
}

impl From<InitArg> for NndInit {
    fn from(a: InitArg) -> Self {
        match a {
            InitArg::Random => NndInit::Random,
            InitArg::MetricTree => NndInit::MetricTree,
        }
    }
}

/// Benchmark APOTHEOSIS + NSG on a flat JSON array of TLSH hashes (file_hashes.json)
#[derive(Parser, Debug)]
#[command(name = "test_tlsh", about = "Tune build modes/params and measure recall@1 vs brute force (TLSH)")]
struct Args {
    /// Flat JSON file: an array of objects each with a TLSH field.
    #[arg(long, default_value = "file_hashes.json")]
    dataset: String,
    /// JSON field holding the TLSH hash in each object.
    #[arg(long, default_value = "TLSH")]
    field: String,
    /// Drop duplicate hashes.
    #[arg(long, default_value_t = false)]
    dedup: bool,
    /// Number of dataset points to index.
    #[arg(long, default_value_t = 42000)]
    dataset_size: usize,
    /// First query index (defaults to dataset_size, i.e. disjoint from the dataset).
    #[arg(long)]
    query_start: Option<usize>,
    /// Number of queries.
    #[arg(long, default_value_t = 1000)]
    query_count: usize,
    /// k for the search (recall@1 only inspects the top-1).
    #[arg(long, default_value_t = 1)]
    search_k: usize,
    /// Search-time ef. Omitted => use the build ef (--ef).
    #[arg(long)]
    ef_search: Option<usize>,
    /// Path for the cached brute-force ground truth (auto-derived if omitted).
    #[arg(long)]
    brute_cache: Option<String>,

    // ---- build mode ----
    #[arg(long, value_enum, default_value_t = InitArg::MetricTree)]
    init: InitArg,
    #[arg(long, default_value_t = 10)]
    iter: usize,
    #[arg(long, default_value_t = 16)]
    n_trees: usize,
    #[arg(long, default_value_t = 32)]
    leaf_size: usize,

    // ---- NSG sizes (M, C, EF) ----
    #[arg(long, default_value_t = 32)]
    m: usize,
    #[arg(long, default_value_t = 200)]
    c: usize,
    #[arg(long, default_value_t = 64)]
    ef: usize,

    // ---- NN-descent sizes (K, L, S, R) ----
    #[arg(long = "nnd-k", default_value_t = 50)]
    nnd_k: usize,
    #[arg(long = "nnd-l", default_value_t = 400)]
    nnd_l: usize,
    #[arg(long = "nnd-s", default_value_t = 10)]
    nnd_s: usize,
    #[arg(long = "nnd-r", default_value_t = 200)]
    nnd_r: usize,
}

fn push_field(obj: &Value, field: &str, out: &mut Vec<String>) {
    if let Some(s) = obj.get(field).and_then(|x| x.as_str()) {
        if !s.is_empty() && s != "TNULL" {
            out.push(s.to_string());
        }
    }
}

fn read_hashes(path: &str, field: &str, dedup: bool) -> Vec<String> {
    let data = fs::read_to_string(path).expect("Failed to read JSON file");
    let v: Value = serde_json::from_str(&data).expect("Failed to parse JSON");
    let mut out = Vec::new();
    match &v {
        Value::Array(arr) => {
            for item in arr {
                push_field(item, field, &mut out);
            }
        }
        Value::Object(map) => {
            for val in map.values() {
                push_field(val, field, &mut out);
            }
        }
        _ => {}
    }
    out.retain(|s| TlshDefault::from_str(s).is_ok());
    if dedup {
        let mut seen = HashSet::new();
        out.retain(|s| seen.insert(s.clone()));
    }
    out
}

fn create_tlsh_object(hash: &str) -> TlshDefault {
    TlshDefault::from_str(hash).unwrap()
}

fn brute_cache_path(args: &Args, dataset_size: usize, qs: usize, qc: usize) -> String {
    if let Some(p) = &args.brute_cache {
        return p.clone();
    }
    let base = Path::new(&args.dataset)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset");
    let dd = if args.dedup { "dedup" } else { "all" };
    format!(".brute_cache_tlsh_{base}_{dd}_{dataset_size}_{qs}_{qc}.json")
}

fn compute_brute(dataset: &[String], queries: &[String]) -> Vec<(u32, String)> {
    let cand_objs: Vec<TlshDefault> = dataset.iter().map(|s| create_tlsh_object(s)).collect();
    queries
        .iter()
        .map(|q| {
            let tq = create_tlsh_object(q);
            let mut best = (u32::MAX, String::new());
            for (i, c) in cand_objs.iter().enumerate() {
                let d = tq.diff(c, true) as u32;
                if d < best.0 {
                    best = (d, dataset[i].clone());
                }
            }
            best
        })
        .collect()
}

fn load_or_compute_brute(path: &str, dataset: &[String], queries: &[String]) -> Vec<(u32, String)> {
    if Path::new(path).exists() {
        if let Ok(data) = fs::read_to_string(path) {
            if let Ok(v) = serde_json::from_str::<Vec<(u32, String)>>(&data) {
                if v.len() == queries.len() {
                    eprintln!("Loaded brute-force ground truth from {path}");
                    return v;
                }
            }
        }
        eprintln!("Ignoring stale/invalid brute cache at {path}");
    }
    eprintln!(
        "Computing brute-force ground truth ({} queries x {} candidates)...",
        queries.len(),
        dataset.len()
    );
    let v = compute_brute(dataset, queries);
    if let Err(e) = fs::write(path, serde_json::to_string(&v).unwrap()) {
        eprintln!("Warning: could not write brute cache {path}: {e}");
    } else {
        eprintln!("Saved brute-force ground truth to {path}");
    }
    v
}

// cargo run --release --bin test_tlsh_exams -- --help (to see all flags)
pub fn main() {
    let args = Args::parse();

    let init: NndInit = args.init.into();
    let config = BuildConfig {
        nsg: NsgParams { m: args.m, c: args.c, ef: args.ef },
        nnd: NndParams { k: args.nnd_k, l: args.nnd_l, s: args.nnd_s, r: args.nnd_r },
        init,
        iter: args.iter,
        n_trees: args.n_trees,
        leaf_size: args.leaf_size,
    };

    let hashes = read_hashes(&args.dataset, &args.field, args.dedup);
    let n_total = hashes.len();
    eprintln!("Number of usable TLSH hashes: {n_total}");

    let dataset_size = args.dataset_size.min(n_total);
    let query_start = args.query_start.unwrap_or(dataset_size).min(n_total);
    let query_end = (query_start + args.query_count).min(n_total);
    assert!(dataset_size > 0, "dataset is empty");

    let dataset: Vec<String> = hashes[..dataset_size].to_vec();
    let queries: Vec<String> = hashes[query_start..query_end].to_vec();
    assert!(!queries.is_empty(), "no queries selected");

    let eff_ef = args.ef_search.unwrap_or(config.nsg.ef);
    let init_label = match init {
        NndInit::Random => "random",
        NndInit::MetricTree => "metric_tree",
    };

    eprintln!(
        "Config: init={}, iter={}, n_trees={}, leaf_size={} | M={} C={} EF={} | K={} L={} S={} R={}",
        init_label, config.iter, config.n_trees, config.leaf_size,
        config.nsg.m, config.nsg.c, config.nsg.ef,
        config.nnd.k, config.nnd.l, config.nnd.s, config.nnd.r
    );
    eprintln!(
        "Dataset size: {}, Queries: {} (range {}..{}), search_k={}, ef_search={}",
        dataset.len(), queries.len(), query_start, query_end, args.search_k, eff_ef
    );

    let records: Vec<SimpleTlshRecord> = dataset
        .iter()
        .map(|s| SimpleTlshRecord::create(s.clone()))
        .collect();

    let mut apotheosis = Apotheosis::<SimpleTlshRecord, TlshDistance>::new(config);

    eprintln!("Inserting into APOTHEOSIS / NSG model...");
    let creation_start = Instant::now();
    apotheosis.insert(records);
    let creation_time = creation_start.elapsed();

    let cache = brute_cache_path(&args, dataset_size, query_start, args.query_count);
    let brute_start = Instant::now();
    let brute = load_or_compute_brute(&cache, &dataset, &queries);
    let brute_time = brute_start.elapsed();

    eprintln!("Running APOTHEOSIS search...");
    let nsg_start = Instant::now();
    let mut apo_results: Vec<(u32, TlshDefault)> = Vec::with_capacity(queries.len());
    for q in &queries {
        let qh = TlshDefault::from_str(q).unwrap();
        let res = apotheosis.search(&qh, args.search_k, args.ef_search);
        let top = res.first().expect("empty search result");
        apo_results.push((top.0, top.1.search_id()));
    }
    let nsg_time = nsg_start.elapsed();

    let mut recall_at_1 = 0usize;
    let mut exact_same = 0usize;
    let mut genuine_miss = 0usize;
    for (apo, br) in apo_results.iter().zip(brute.iter()) {
        if apo.0 == br.0 {
            recall_at_1 += 1;
            let br_obj = create_tlsh_object(&br.1);
            if apo.1.hash() == br_obj.hash() {
                exact_same += 1;
            }
        } else {
            genuine_miss += 1;
        }
    }

    let n = apo_results.len();
    eprintln!("Recall@1 (distance): {recall_at_1}/{n}");
    eprintln!("Exact same point:    {exact_same}/{n}");
    eprintln!("Genuine misses:      {genuine_miss}/{n}");
    eprintln!("Creation time:    {:?}", creation_time);
    eprintln!("Brute force time: {:?}", brute_time);
    eprintln!("NSG search time:  {:?}", nsg_time);

    println!(
        "RESULT,{init},{iter},{ntrees},{leaf},{m},{c},{ef},{k},{l},{s},{r},\
         {dsize},{nq},{sk},{effef},{recall},{exact},{miss},{cms:.3},{bms:.3},{nms:.3}",
        init = init_label,
        iter = config.iter,
        ntrees = config.n_trees,
        leaf = config.leaf_size,
        m = config.nsg.m,
        c = config.nsg.c,
        ef = config.nsg.ef,
        k = config.nnd.k,
        l = config.nnd.l,
        s = config.nnd.s,
        r = config.nnd.r,
        dsize = dataset.len(),
        nq = n,
        sk = args.search_k,
        effef = eff_ef,
        recall = recall_at_1,
        exact = exact_same,
        miss = genuine_miss,
        cms = creation_time.as_secs_f64() * 1000.0,
        bms = brute_time.as_secs_f64() * 1000.0,
        nms = nsg_time.as_secs_f64() * 1000.0,
    );
}
