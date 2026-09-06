// =============================================================================
// NSG recall/latency benchmark — "exams" dataset (flat file_hashes.json), ssdeep
// =============================================================================
//
// Overview
//   Indexes a slice of ssdeep signatures into the APOTHEOSIS / NSG model, runs a set
//   of queries, and measures recall against a brute-force ground truth.
//
//   With --model-cache the built model is dumped to disk and reused on the next
//   run that requests the same build config, replacing a rebuild with a load.
//   This is what makes the ef-search sweep cheap: the first ef-search point
//   builds and dumps the model; the remaining points load it.
//
// Cache integrity
//   Each cache (model, brute, hash) has a companion "<file>.fp" sidecar holding
//   a stable FNV-1a fingerprint of the data it was derived from:
//     * model .fp  fingerprint of the indexed slice
//     * brute .fp  fingerprint of (indexed slice, query slice)
//     * hash  .fp  signature of the dataset file on disk (path, length, mtime)
//   The fingerprint is verified on load; a mismatch forces a rebuild or
//   recompute rather than trusting a same-shaped but incorrect cache. A missing
//   sidecar (for example, a cache written by an older build, or by the HNSW
//   binary before it was updated) is accepted by shape, with a warning, and a
//   sidecar is then written so subsequent runs are protected. The brute and hash
//   caches are shared with the HNSW harness, so the HNSW binary must use the
//   same fingerprint functions (see MIGRATION_NOTES) for the cross-tool sharing
//   to remain safe.
//
// Timing
//   The search is timed over --repeats passes (default 1) and the reported
//   nsg_ms is the median, with an optional untimed --warmup pass. creation_ms
//   and brute_ms are single measurements whose meaning depends on whether the
//   cache was hit, so the RESULT line also carries model_cached and brute_cached
//   flags (0/1) to distinguish "built" from "loaded". A derived qps column is
//   emitted for convenience.
//
// Harness interface
//   The sweep harness (scripts/lib_nsg.sh) invokes this binary once per
//   (build config, ef_search) point and parses the single stdout line:
//
//     RESULT,init,iter,n_trees,leaf_size,M,C,EF,alpha,K,L,S,R,
//            dataset_size,num_queries,search_k,ef_search,
//            recall_at_k,exact_same,genuine_miss,creation_ms,brute_ms,nsg_ms,
//            model_cached,brute_cached,repeats,qps,incomparable
//
//   All other output is written to stderr. The RESULT columns and their order
//   must stay in sync with the CSV header in lib_nsg.sh, plus the trailing `incomparable` column the ssdeep wrapper appends.
//
//   Column-name note: recall_at_k and genuine_miss are complementary (they sum
//   to 1); exact_same (exact-item recall) is independent of both.
//
// Parameter groups (all runtime flags; nothing is compile-time)
//   build mode : --init {random|metric-tree}, --iter, --n-trees, --leaf-size
//   NSG sizes  : --m (M), --c (C), --ef (EF, the build-time candidate pool)
//   pruning    : --alpha (scale of the MRNG occlusion test; 1.0 is the plain RNG
//                rule, >1 occludes fewer candidates and keeps more short edges,
//                <1 occludes more and keeps only diversified directions). Degree
//                stays ~M either way, since the pruned candidates refill the list.
//                It is a build parameter: a model cached under one alpha must
//                not be reused for another, which is why the harness puts it in
//                the dump filename.
//   NN-descent : --nnd-k (K), --nnd-l (L), --nnd-s (S), --nnd-r (R)
//   search     : --search-k, --ef-search (omit to reuse the build --ef)
//   timing     : --repeats (median over N passes), --warmup
//   sampling   : --shuffle/--seed (representative sample), --dedup
//   caching    : --hash-cache (skip JSON parse), --brute-cache, --model-cache, --rescan
//
// Recall definitions (mean over queries, each in [0, 1])
//   distance-recall@k : fraction of returned items within the k-th true distance
//                       (tie-tolerant).
//   exact-item recall : additionally requires the returned point to be a true
//                       top-k point (compared by hash bytes, so duplicates of
//                       the right hash still count).
//
// Help:  cargo run --release --bin test_ssdeep_exams -- --help

use apotheosis3::controllers::apotheosis::Apotheosis;
use apotheosis3::controllers::nndescent::NndParams;
use apotheosis3::controllers::nsg::{BuildConfig, NndInit, NsgParams};
use apotheosis3::datalayer::algorithms::{DistanceAlgorithm, SsdeepDistance, SsdeepHash};
use apotheosis3::datalayer::record::{ApotheosisRecord, SimpleSsdeepRecord};
use clap::{Parser, ValueEnum};
use serde_json::Value;
use std::collections::{BinaryHeap, HashSet};
use std::fs;
use std::path::Path;
use std::time::Instant;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// ssdeep glue. `ssdeep::compare` (libfuzzy) returns a 0..=100 *similarity*
// (100 == identical); NSG wants a distance where smaller == closer, so
// distance = 100 - similarity.
//
// CAVEAT: ssdeep returns 0 once block sizes differ by more than 2x -- that is
// structural, whatever the content -- collapsing those pairs to the max
// distance. On a corpus with a spread of file sizes that is most pairs, so
// expect lower recall than TLSH, and read the `incomparable` column before
// trusting recall_at_k.
// ---------------------------------------------------------------------------

type SsdeepRecord = SimpleSsdeepRecord;

/// True when libfuzzy will accept `s` as a signature. Comparing a signature
/// with itself succeeds exactly when it parses, so this asks libfuzzy instead
/// of guessing at the format. The NUL check comes first because
/// `ssdeep::compare` *panics*, rather than erroring, on an interior NUL byte.
fn is_valid_ssdeep(s: &str) -> bool {
    !s.bytes().any(|b| b == 0) && ssdeep::compare(s, s).is_ok()
}

// CLI-facing copy of NndInit so clap can derive a --init value parser; the
// From impl below converts it into the controller's own enum.
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

// ---------------------------------------------------------------------------
// Determinism and cache-integrity helpers.
//   * SplitMix64  a fully specified PRNG that keeps --shuffle reproducible
//                 across compilers, rand versions, and both the NSG and HNSW
//                 crates.
//   * FNV-1a      a stable content hash. std's DefaultHasher is not stable
//                 across Rust versions and must not be used for a cross-build
//                 cache key.
//   * .fp sidecar every cache (model, brute, hash) has a companion <file>.fp
//                 holding a hex fingerprint, verified on load; a mismatch is
//                 refused.
// ---------------------------------------------------------------------------

struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

fn fnv1a64(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Stable fingerprint over one or more ordered lists of hash strings.
fn fingerprint_slices(slices: &[&[String]]) -> u64 {
    let mut h = FNV_OFFSET;
    for sl in slices {
        h = fnv1a64(&(sl.len() as u64).to_le_bytes(), h);
        for s in *sl {
            h = fnv1a64(s.as_bytes(), h);
            h = fnv1a64(&[0xff], h); // record separator (prevents concat collisions)
        }
    }
    h
}

fn read_fp(path: &str) -> Option<u64> {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| u64::from_str_radix(s.trim(), 16).ok())
}

fn write_fp(path: &str, fp: u64) {
    if let Err(e) = fs::write(path, format!("{:016x}", fp)) {
        eprintln!("Warning: could not write fingerprint sidecar {path}: {e}");
    }
}

/// Median of a slice of f64 (sorts in place). 0.0 for an empty slice.
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Cheap dataset signature: path + length + mtime of the dataset file. Detects a
/// regenerated dataset on disk (which would invalidate a hash cache) without
/// re-reading the file. mtime-based, so it errs on the safe side: a touched file
/// forces a rescan rather than risking a stale cache.
fn dataset_signature(path: &str) -> u64 {
    let mut h = FNV_OFFSET;
    h = fnv1a64(path.as_bytes(), h);
    if let Ok(md) = fs::metadata(path) {
        h = fnv1a64(&md.len().to_le_bytes(), h);
        if let Ok(t) = md.modified() {
            if let Ok(d) = t.duration_since(std::time::UNIX_EPOCH) {
                h = fnv1a64(&d.as_secs().to_le_bytes(), h);
            }
        }
    }
    h
}

/// Benchmark APOTHEOSIS + NSG on a flat JSON array of ssdeep signatures (file_hashes.json).
#[derive(Parser, Debug)]
#[command(name = "test_ssdeep_exams", about = "Tune build modes/params and measure recall@k vs brute force (ssdeep)")]
struct Args {
    /// Flat JSON file: an array of objects each with an ssdeep field.
    #[arg(long, default_value = "file_hashes.json")]
    dataset: String,
    /// JSON field holding the ssdeep signature in each object.
    #[arg(long, default_value = "ssdeep")]
    field: String,
    /// Drop duplicate hashes.
    #[arg(long, default_value_t = false)]
    dedup: bool,
    /// Shuffle the full hash list (seeded) before slicing dataset/queries, so the
    /// indexed set and queries are a representative sample rather than the first
    /// records in file order.
    #[arg(long, default_value_t = false)]
    shuffle: bool,
    /// RNG seed for --shuffle (fixed so runs are reproducible).
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Number of dataset points to index (0 = use all available).
    #[arg(long, default_value_t = 42000)]
    dataset_size: usize,
    /// First query index (defaults to dataset_size, i.e. disjoint from the dataset).
    #[arg(long)]
    query_start: Option<usize>,
    /// Number of queries.
    #[arg(long, default_value_t = 1000)]
    query_count: usize,
    /// k for the search (recall@k).
    #[arg(long, default_value_t = 1)]
    search_k: usize,
    /// Search-time ef. Omitted => use the build ef (--ef).
    #[arg(long)]
    ef_search: Option<usize>,
    /// Number of timed search passes; the reported nsg_ms is the median.
    #[arg(long, default_value_t = 1)]
    repeats: usize,
    /// Run one untimed warmup pass before the timed passes.
    #[arg(long, default_value_t = false)]
    warmup: bool,
    /// Path for the cached brute-force ground truth (auto-derived if omitted).
    #[arg(long)]
    brute_cache: Option<String>,
    /// Cached model dump. If it exists and matches the dataset size + fingerprint,
    /// load it instead of building; otherwise build and write it here.
    #[arg(long)]
    model_cache: Option<String>,
    /// Cache file for the extracted + validated hash list. Only used when given;
    /// without it the dataset is always re-read (no auto-cache is written).
    #[arg(long)]
    hash_cache: Option<String>,
    /// Force a re-read of the dataset, ignoring any existing hash cache.
    #[arg(long, default_value_t = false)]
    rescan: bool,

    // ---- build mode ----
    #[arg(long, value_enum, default_value_t = InitArg::MetricTree)]
    init: InitArg,
    #[arg(long, default_value_t = 10)]
    iter: usize,
    #[arg(long, default_value_t = 16)]
    n_trees: usize,
    #[arg(long, default_value_t = 32)]
    leaf_size: usize,

    // ---- NSG sizes (M, C, EF) and the pruning factor (alpha) ----
    #[arg(long, default_value_t = 32)]
    m: usize,
    #[arg(long, default_value_t = 200)]
    c: usize,
    #[arg(long, default_value_t = 64)]
    ef: usize,
    /// Scale of the MRNG occlusion test: a candidate is dropped when
    /// alpha*d(selected,candidate) < d(node,candidate). 1.0 is the plain
    /// relative-neighborhood rule; above it fewer candidates are occluded (more
    /// near-redundant edges kept), below it more are (only well-spread
    /// directions survive). Out-degree stays ~M regardless. Must be > 0.
    #[arg(long, default_value_t = 1.2)]
    alpha: f32,

    // ---- NN-descent sizes (K, L, S, R) ----
    #[arg(long = "nnd-k", default_value_t = 50)]
    nnd_k: usize,
    #[arg(long = "nnd-l", default_value_t = 400)]
    nnd_l: usize,
    #[arg(long = "nnd-s", default_value_t = 10)]
    nnd_s: usize,
    #[arg(long = "nnd-r", default_value_t = 200)]
    nnd_r: usize,

    /// PRNG seed for the index construction (NSG + NN-descent + VP-forest).
    /// Distinct from --seed, which only shuffles the data before slicing: this
    /// one leaves the indexed points untouched and redraws the algorithm's own
    /// randomness, so repeating a run over several --build-seed values samples
    /// the build variance (see the harness --trials flag).
    #[arg(long, default_value_t = 42)]
    build_seed: u64,
}

// ---------------------------------------------------------------------------
// Dataset reading: pull the ssdeep field out of the flat JSON (array, or a keyed
// object) and keep only strings that parse as valid ssdeep signatures.
// ---------------------------------------------------------------------------

fn push_field(obj: &Value, field: &str, out: &mut Vec<String>) {
    if let Some(s) = obj.get(field).and_then(|x| x.as_str()) {
        if !s.is_empty() && s != "3::" {
            out.push(s.to_string());
        }
    }
}

fn read_hashes(path: &str, field: &str, dedup: bool) -> Vec<String> {
    let data = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read JSON file {path}: {e}"));
    let v: Value = serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("Failed to parse JSON {path}: {e}"));
    let mut out = Vec::new();
    match &v {
        Value::Array(arr) => {
            for item in arr {
                push_field(item, field, &mut out);
            }
        }
        // Tolerate a keyed object too ({ name -> record }).
        Value::Object(map) => {
            for val in map.values() {
                push_field(val, field, &mut out);
            }
        }
        _ => {}
    }
    out.retain(|s| is_valid_ssdeep(s));
    if dedup {
        let mut seen = HashSet::new();
        out.retain(|s| seen.insert(s.clone()));
    }
    out
}

// ---------------------------------------------------------------------------
// Hash-list cache: active only when --hash-cache is given. In that case the
// extracted and validated list is cached to that file so repeated runs skip the
// JSON parse; --rescan bypasses a stale one. A "<cache>.fp" sidecar stores the
// dataset signature, so a dataset regenerated on disk (same path) forces a
// rescan even without --rescan. Without --hash-cache the dataset is always
// re-read and no cache is written.
// ---------------------------------------------------------------------------

fn load_or_scan_hashes(args: &Args) -> Vec<String> {
    let cache = match &args.hash_cache {
        Some(p) => p.clone(),
        None => return read_hashes(&args.dataset, &args.field, args.dedup),
    };
    let sig_path = format!("{cache}.fp");
    let sig_now = dataset_signature(&args.dataset);
    if !args.rescan && Path::new(&cache).exists() {
        if let Ok(data) = fs::read_to_string(&cache) {
            let v: Vec<String> = data.lines().map(|l| l.to_string()).collect();
            if !v.is_empty() {
                match read_fp(&sig_path) {
                    Some(s) if s == sig_now => {
                        eprintln!("Loaded {} cached hashes from {cache} (signature ok)", v.len());
                        return v;
                    }
                    Some(_) => {
                        eprintln!("Hash cache {cache} is stale (dataset changed on disk) — rescanning");
                    }
                    None => {
                        eprintln!(
                            "Hash cache {cache} has no signature sidecar; using it and writing one \
                             (pass --rescan if the dataset changed)."
                        );
                        write_fp(&sig_path, sig_now);
                        return v;
                    }
                }
            } else {
                eprintln!("Ignoring empty/invalid hash cache at {cache}");
            }
        }
    }
    let v = read_hashes(&args.dataset, &args.field, args.dedup);
    match fs::write(&cache, v.join("\n")) {
        Ok(_) => {
            eprintln!("Saved {} hashes to {cache}", v.len());
            write_fp(&sig_path, sig_now);
        }
        Err(e) => eprintln!("Warning: could not write hash cache {cache}: {e}"),
    }
    v
}

// ---------------------------------------------------------------------------
// Brute-force ground truth: exact top-k per query, cached to disk so every
// (build config, ef_search) point reuses it. Validated by a "<path>.fp" sidecar
// holding the fingerprint of (dataset, queries), so a same-shaped but wrong
// cache (different data, seed, or extraction order) can never be reused.
// ---------------------------------------------------------------------------

fn create_ssdeep_object(hash: &str) -> SsdeepHash {
    SsdeepHash(hash.to_string())
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
    let shuf = if args.shuffle { format!("shuf{}", args.seed) } else { "noshuf".to_string() };
    format!(".brute_cache_ssdeep_{base}_{dd}_{shuf}_{dataset_size}_{qs}_{qc}.json")
}

fn compute_brute(dataset: &[String], queries: &[String], k: usize) -> Vec<Vec<(u32, String)>> {
    // Parse all candidates in parallel (millions of hashes).
    let cand_objs: Vec<SsdeepHash> = dataset.par_iter().map(|s| create_ssdeep_object(s)).collect();
    queries
        .par_iter()
        .map(|q| {
            let tq = create_ssdeep_object(q);
            // Max-heap of the k smallest (distance, index) pairs seen so far.
            let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(k + 1);
            for (i, c) in cand_objs.iter().enumerate() {
                let d = SsdeepDistance.calculate_distance(&tq, c);
                if heap.len() < k {
                    heap.push((d, i));
                } else if let Some(&(worst, _)) = heap.peek() {
                    if d < worst {
                        heap.pop();
                        heap.push((d, i));
                    }
                }
            }
            let mut v: Vec<(u32, usize)> = heap.into_vec();
            v.sort_unstable();
            v.into_iter().map(|(d, i)| (d, dataset[i].clone())).collect()
        })
        .collect()
}

fn load_or_compute_brute(
    path: &str,
    dataset: &[String],
    queries: &[String],
    k: usize,
    fp: u64,
) -> (Vec<Vec<(u32, String)>>, bool) {
    let fp_path = format!("{path}.fp");
    if Path::new(path).exists() {
        if let Ok(data) = fs::read_to_string(path) {
            if let Ok(v) = serde_json::from_str::<Vec<Vec<(u32, String)>>>(&data) {
                // Reuse only if it covers every query at the required depth k.
                let shape_ok = v.len() == queries.len() && v.first().map_or(true, |row| row.len() >= k);
                if shape_ok {
                    match read_fp(&fp_path) {
                        Some(s) if s == fp => {
                            eprintln!("Loaded brute-force ground truth (top-{k}, fp ok) from {path}");
                            return (v, true);
                        }
                        Some(_) => {
                            eprintln!(
                                "Ignoring brute cache at {path}: fingerprint mismatch \
                                 (different data/queries) — recomputing"
                            );
                        }
                        None => {
                            eprintln!(
                                "Brute cache at {path} has no fingerprint sidecar; accepting by shape \
                                 and writing one (older or HNSW-written cache?)."
                            );
                            write_fp(&fp_path, fp);
                            return (v, true);
                        }
                    }
                } else {
                    eprintln!("Ignoring brute cache at {path}: shape/depth mismatch — recomputing");
                }
            } else {
                eprintln!("Ignoring invalid brute cache at {path} — recomputing");
            }
        }
    }
    eprintln!(
        "Computing brute-force ground truth top-{k} ({} queries x {} candidates)...",
        queries.len(),
        dataset.len()
    );
    let v = compute_brute(dataset, queries, k);
    match fs::write(path, serde_json::to_string(&v).unwrap()) {
        Ok(()) => {
            eprintln!("Saved brute-force ground truth to {path}");
            write_fp(&fp_path, fp);
        }
        Err(e) => eprintln!("Warning: could not write brute cache {path}: {e}"),
    }
    (v, false)
}

// ---------------------------------------------------------------------------
// Model cache: load a previously dumped model if one exists at --model-cache and
// its indexed count and fingerprint match; otherwise build it and dump it there
// (with a "<path>.fp" sidecar). lib_nsg.sh keys the path on the build-config
// slug and invalidates any dump older than the freshly compiled binary, so a
// recompile triggers a rebuild.
// ---------------------------------------------------------------------------

fn load_or_build_model(
    args: &Args,
    config: BuildConfig,
    records: Vec<SsdeepRecord>,
    expected_len: usize,
    fp: u64,
) -> (Apotheosis<SsdeepRecord, SsdeepDistance>, bool) {
    if let Some(path) = &args.model_cache {
        if Path::new(path).exists() {
            let fp_path = format!("{path}.fp");
            let sidecar = read_fp(&fp_path);
            match Apotheosis::<SsdeepRecord, SsdeepDistance>::load(path) {
                // Apotheosis::insert dedups by radix key, so a model legitimately
                // holds FEWER records than were handed to it — an exact record-count
                // match is the wrong test and never succeeds on a corpus with
                // duplicate hashes. The fingerprint is taken over the *input* slice
                // and is unaffected by the dedup, so that is the check that matters;
                // the count is only a shape sanity test for dumps written before the
                // sidecar existed.
                Ok(m) => match sidecar {
                    Some(s) if s == fp => {
                        eprintln!(
                            "Loaded cached model ({} unique records indexed from {expected_len}, fp ok) from {path}",
                            m.records.len()
                        );
                        return (m, true);
                    }
                    Some(_) => {
                        eprintln!(
                            "Ignoring cached model at {path}: fingerprint mismatch (data changed) — rebuilding"
                        );
                    }
                    None if !m.records.is_empty() && m.records.len() <= expected_len => {
                        eprintln!(
                            "Cached model at {path} has no fingerprint sidecar; accepting by shape \
                             and writing one (use --no-model-cache to force a rebuild if unsure)."
                        );
                        write_fp(&fp_path, fp);
                        return (m, true);
                    }
                    None => eprintln!(
                        "Ignoring cached model at {path}: no fingerprint sidecar, and {} records \
                         is not a plausible subset of {expected_len}",
                        m.records.len()
                    ),
                },
                Err(e) => eprintln!("Ignoring unreadable cached model at {path}: {e}"),
            }
        }
    }
    eprintln!("Building APOTHEOSIS / NSG model...");
    let mut apotheosis = Apotheosis::<SsdeepRecord, SsdeepDistance>::new(config);
    apotheosis.insert(records);
    if let Some(path) = &args.model_cache {
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        match apotheosis.dump(path) {
            Ok(()) => {
                eprintln!("Saved model to {path}");
                write_fp(&format!("{path}.fp"), fp);
            }
            Err(e) => eprintln!("Warning: could not write model cache {path}: {e}"),
        }
    }
    (apotheosis, false)
}

// ---------------------------------------------------------------------------
// Entry point: assemble the build config, load + slice the data, build the NSG
// index, compute/reuse ground truth, run the search, score recall, print RESULT.
// ---------------------------------------------------------------------------
pub fn main() {
    let args = Args::parse();

    let init: NndInit = args.init.into();
    // A non-positive or non-finite alpha occludes every candidate in sync_prune,
    // so the run would silently measure a degenerate graph. Reject it up front.
    assert!(
        args.alpha.is_finite() && args.alpha > 0.0,
        "--alpha must be finite and > 0 (got {}); 1.0 is the plain RNG rule",
        args.alpha
    );
    let config = BuildConfig {
        nsg: NsgParams { m: args.m, c: args.c, ef: args.ef, alpha: args.alpha },
        nnd: NndParams { k: args.nnd_k, l: args.nnd_l, s: args.nnd_s, r: args.nnd_r },
        init,
        iter: args.iter,
        n_trees: args.n_trees,
        leaf_size: args.leaf_size,
        seed: args.build_seed,
    };

    let mut hashes = load_or_scan_hashes(&args);
    if args.shuffle {
        // Self-contained SplitMix64 Fisher-Yates shuffle: reproducible across
        // compilers, rand versions, and both the NSG and HNSW crates, so
        // --shuffle yields an identical permutation everywhere. That identity is
        // a hard requirement for the two harnesses to share a brute-force cache.
        let mut rng = SplitMix64::new(args.seed);
        for i in (1..hashes.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            hashes.swap(i, j);
        }
        eprintln!("Shuffled {} hashes with seed {} (SplitMix64)", hashes.len(), args.seed);
    }
    let n_total = hashes.len();
    eprintln!("Number of usable ssdeep signatures: {n_total}");

    let dataset_size = if args.dataset_size == 0 {
        n_total
    } else {
        args.dataset_size.min(n_total)
    };
    let query_start = args.query_start.unwrap_or(dataset_size).min(n_total);
    let query_end = (query_start + args.query_count).min(n_total);
    assert!(dataset_size > 0, "no usable ssdeep signatures found in {}", args.dataset);

    let dataset: Vec<String> = hashes[..dataset_size].to_vec();
    let queries: Vec<String> = hashes[query_start..query_end].to_vec();
    assert!(
        !queries.is_empty(),
        "no queries selected (dataset_size={dataset_size}, query_start={query_start}, corpus={n_total})"
    );
    if queries.len() < args.query_count {
        eprintln!(
            "warning: requested {} queries but only {} are available after the dataset slice \
             (corpus has {} usable hashes)",
            args.query_count,
            queries.len(),
            n_total
        );
    }

    let k_eff = args.search_k.min(dataset.len());
    let eff_ef = args.ef_search.unwrap_or(config.nsg.ef);
    if eff_ef < args.search_k {
        eprintln!(
            "warning: ef_search ({eff_ef}) < search_k ({}); recall@{} is capped by the pool size",
            args.search_k, args.search_k
        );
    }
    let init_label = match init {
        NndInit::Random => "random",
        NndInit::MetricTree => "metric_tree",
    };

    eprintln!(
        "Config: init={}, iter={}, n_trees={}, leaf_size={} | M={} C={} EF={} alpha={} | K={} L={} S={} R={}",
        init_label, config.iter, config.n_trees, config.leaf_size,
        config.nsg.m, config.nsg.c, config.nsg.ef, config.nsg.alpha,
        config.nnd.k, config.nnd.l, config.nnd.s, config.nnd.r
    );
    eprintln!(
        "Dataset size: {}, Queries: {} (range {}..{}), search_k={}, ef_search={}",
        dataset.len(), queries.len(), query_start, query_end, args.search_k, eff_ef
    );

    // Content fingerprints for cache integrity (stable FNV-1a over the sliced
    // hash strings). The model dump is keyed on the indexed slice; the brute
    // truth on the indexed and query slices.
    let model_fp = fingerprint_slices(&[dataset.as_slice()]);
    let brute_fp = fingerprint_slices(&[dataset.as_slice(), queries.as_slice()]);

    let records: Vec<SsdeepRecord> = dataset
        .iter()
        .map(|s| SimpleSsdeepRecord::create(s.clone()))
        .collect();

    eprintln!("Preparing APOTHEOSIS / NSG model...");
    let creation_start = Instant::now();
    let (mut apotheosis, model_cached) =
        load_or_build_model(&args, config, records, dataset.len(), model_fp);
    let creation_time = creation_start.elapsed();
    // Put the built and the loaded path on the same RNG stream: the model dump
    // carries no PRNG state, so without this a cached model would search with a
    // different stream than the one that just built it.
    apotheosis.set_seed(args.build_seed);
    // insert() collapses records that share a radix key, so the graph is usually
    // smaller than dataset_size. dataset_size in the RESULT line is the number of
    // rows fed in, NOT the number of nodes in the index.
    eprintln!(
        "Indexed {} unique signatures from {} rows ({} collapsed as duplicates)",
        apotheosis.records.len(),
        dataset.len(),
        dataset.len() - apotheosis.records.len()
    );

    let cache = brute_cache_path(&args, dataset_size, query_start, args.query_count);
    let brute_start = Instant::now();
    let (brute, brute_cached) = load_or_compute_brute(&cache, &dataset, &queries, k_eff, brute_fp);
    let brute_time = brute_start.elapsed();

    // ---- timed search: optional warmup + `repeats` passes, report median -----
    let repeats = args.repeats.max(1);
    eprintln!("Running APOTHEOSIS search (repeats={repeats}, warmup={})...", args.warmup);
    if args.warmup {
        for q in &queries {
            let qh = SsdeepHash(q.clone());
            let _ = apotheosis.search(&qh, args.search_k, args.ef_search);
        }
    }
    let mut nsg_times_ms: Vec<f64> = Vec::with_capacity(repeats);
    // Approx top-k per query, kept from the first pass for scoring (the search is
    // deterministic, so every pass yields the same result set).
    let mut approx: Vec<Vec<(u32, Vec<u8>)>> = Vec::new();
    for r in 0..repeats {
        let start = Instant::now();
        let mut pass: Vec<Vec<(u32, Vec<u8>)>> = Vec::with_capacity(queries.len());
        for q in &queries {
            let qh = SsdeepHash(q.clone());
            let res = apotheosis.search(&qh, args.search_k, args.ef_search);
            pass.push(res.iter().take(k_eff).map(|&(d, rec)| (d, rec.search_id().0.into_bytes())).collect());
        }
        nsg_times_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        if r == 0 {
            approx = pass;
        }
    }
    let nsg_ms = median(&mut nsg_times_ms);

    // recall@k, averaged over queries. Per query, distance-recall@k is the
    // fraction of returned items whose distance is within the k-th true distance
    // (tie-tolerant); exact-item recall additionally requires the same point.
    // The reported value is the MEAN of these per-query recalls.
    let mut recall_sum = 0.0f64; // sum of per-query distance recall@k
    let mut exact_sum = 0.0f64;  // sum of per-query exact-item recall@k
    let mut counted = 0usize;    // queries that contributed (kk > 0)
    // ssdeep-only: queries whose true k-th distance is already the maximum,
    // i.e. the corpus holds nothing with a compatible block size. Distance
    // recall is trivially 1.0 for these (everything returned is <= 100), so a
    // high recall_at_k means little unless this count is small. Read
    // exact_same instead when it is not.
    let mut incomparable = 0usize;
    for (tlist, alist) in brute.iter().zip(approx.iter()) {
        let kk = k_eff.min(tlist.len());
        if kk == 0 {
            continue;
        }
        let true_kth = tlist[kk - 1].0;
        if true_kth >= 100 {
            incomparable += 1;
        }
        let true_ids: HashSet<Vec<u8>> =
            tlist[..kk].iter().map(|(_, h)| create_ssdeep_object(h).0.into_bytes()).collect();
        let mut found = 0usize;
        let mut exact = 0usize;
        for (d, h) in alist.iter() {
            if *d <= true_kth {
                found += 1;
            }
            if true_ids.contains(h) {
                exact += 1;
            }
        }
        recall_sum += found as f64 / kk as f64;
        exact_sum += exact as f64 / kk as f64;
        counted += 1;
    }

    let n = queries.len();
    let denom = counted.max(1) as f64;
    let recall_at_k = recall_sum / denom; // mean recall@k over queries, in [0,1]
    let exact_at_k = exact_sum / denom;   // mean exact-item recall@k
    let miss_at_k = 1.0 - recall_at_k;
    let qps = if nsg_ms > 0.0 { n as f64 * 1000.0 / nsg_ms } else { 0.0 };

    eprintln!(
        "Recall@{} (mean over {} queries): {:.4} ({:.1}%)",
        args.search_k, counted, recall_at_k, 100.0 * recall_at_k
    );
    eprintln!("Exact-item recall@{}: {:.4}", args.search_k, exact_at_k);
    eprintln!(
        "Incomparable queries (no compatible block size anywhere in the corpus): {}/{} ({:.1}%)",
        incomparable,
        counted,
        100.0 * incomparable as f64 / counted.max(1) as f64
    );
    eprintln!("Miss rate@{}:         {:.4}", args.search_k, miss_at_k);
    eprintln!(
        "Creation time:    {:?} ({})",
        creation_time,
        if model_cached { "loaded from cache" } else { "built" }
    );
    eprintln!(
        "Brute force time: {:?} ({})",
        brute_time,
        if brute_cached { "loaded from cache" } else { "computed" }
    );
    eprintln!(
        "NSG search time:  {:.3} ms (median of {} pass(es), ~{:.0} qps)",
        nsg_ms, repeats, qps
    );

    // Machine-readable line for the sweep harness (stdout).
    println!(
        "RESULT,{init},{iter},{ntrees},{leaf},{m},{c},{ef},{alpha},{k},{l},{s},{r},\
         {dsize},{nq},{sk},{effef},{recall:.6},{exact:.6},{miss:.6},\
         {cms:.3},{bms:.3},{nms:.3},{mc},{bc},{rep},{qps:.1},{incomp}",
        init = init_label,
        iter = config.iter,
        ntrees = config.n_trees,
        leaf = config.leaf_size,
        m = config.nsg.m,
        c = config.nsg.c,
        ef = config.nsg.ef,
        alpha = config.nsg.alpha,
        k = config.nnd.k,
        l = config.nnd.l,
        s = config.nnd.s,
        r = config.nnd.r,
        dsize = dataset.len(),
        nq = n,
        sk = args.search_k,
        effef = eff_ef,
        recall = recall_at_k,
        exact = exact_at_k,
        miss = miss_at_k,
        cms = creation_time.as_secs_f64() * 1000.0,
        bms = brute_time.as_secs_f64() * 1000.0,
        nms = nsg_ms,
        mc = model_cached as u8,
        bc = brute_cached as u8,
        rep = repeats,
        qps = qps,
        incomp = incomparable,
    );
}
