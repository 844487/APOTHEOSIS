use crate::datalayer::algorithms::DistanceAlgorithm;
use crate::datalayer::algorithms::Medoid;
use crate::datalayer::nodes::NsgNode;
use crate::controllers::nndescent::NNDescent;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::u32;
use std::collections::HashSet;
use tracing::debug;


fn default_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

// Reusable visited-set for a search
#[derive(Default)]
pub struct SearchScratch {
    visited: Vec<u32>,
    epoch: u32,
}

impl SearchScratch {
    pub fn new(n: usize) -> Self {
        Self { visited: vec![0u32; n], epoch: 0 }
    }

    #[inline]
    fn begin(&mut self, n: usize) {
        if self.visited.len() != n {
            self.visited.clear();
            self.visited.resize(n, 0);
            self.epoch = 0;
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.visited.iter_mut().for_each(|v| *v = 0);
            self.epoch = 1;
        }
    }

    #[inline]
    fn mark(&mut self, idx: usize) {
        self.visited[idx] = self.epoch;
    }

    // True if `idx` was newly visited
    #[inline]
    fn visit(&mut self, idx: usize) -> bool {
        if self.visited[idx] != self.epoch {
            self.visited[idx] = self.epoch;
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NndInit {
    Random,
    MetricTree,
}

// Runtime NSG structural sizes (see the comment on the `Nsg` struct)
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct NsgParams {
    pub m: usize,
    pub c: usize,
    pub ef: usize,
    pub alpha: f32,
}

impl Default for NsgParams {
    fn default() -> Self {
        Self { m: 16, c: 500, ef: 400, alpha: 1.2 }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct BuildConfig {
    pub nsg: NsgParams,
    pub nnd: crate::controllers::nndescent::NndParams,
    pub init: NndInit,
    pub iter: usize,
    pub n_trees: usize,
    pub leaf_size: usize,
    pub seed: u64,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            nsg: NsgParams::default(),
            nnd: crate::controllers::nndescent::NndParams::default(),
            init: NndInit::MetricTree,
            iter: 10,
            n_trees: 16,
            leaf_size: 32,
            seed: 42,
        }
    }
}

// m: max out-degree
// c: max candidates considered in sync prune
// ef: candidate pool size for greedy search
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "ID: serde::Serialize, D: DistanceAlgorithm<ID> + serde::Serialize",
    deserialize = "ID: serde::Deserialize<'de>, D: DistanceAlgorithm<ID> + serde::Deserialize<'de>"
))]
pub struct Nsg<D, ID>
where
    ID: Clone,
    D: DistanceAlgorithm<ID> + Default,
{
    features: Vec<ID>,
    nnd_graph: Vec<NsgNode>,
    navigating_node: usize,
    #[serde(skip, default = "default_rng")]
    prng: StdRng,
    distance: D,
    alpha: f32,
    m: usize,
    c: usize,
    ef: usize,
}

#[allow(non_snake_case)]
impl<D, ID> Nsg<D, ID>
where
    ID: Clone,
    D: DistanceAlgorithm<ID> + Default,
{
    pub fn new() -> Self {
        let p = NsgParams::default();
        Self {
            features: vec![],
            nnd_graph: vec![],
            navigating_node: usize::MAX,
            prng: StdRng::seed_from_u64(42),
            distance: D::default(),
            alpha: p.alpha,
            m: p.m,
            c: p.c,
            ef: p.ef,
        }
    }

    #[inline]
    pub fn default_ef(&self) -> usize {
        self.ef
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.prng = StdRng::seed_from_u64(seed);
    }

    #[inline]
    fn random_node(&mut self) -> usize {
        let n = self.features.len();
        if n == 0 {
            return usize::MAX;
        }

        self.prng.next_u64() as usize % n
    }

    fn get_neighbors(
        &mut self,
        query_id: &ID,
        ef: usize,
        scratch: &mut SearchScratch,
    ) -> (Vec<(usize, u32)>, Vec<(usize, u32)>) {
        scratch.begin(self.features.len());
        let enter_point = self.navigating_node as usize;

        let mut knn_neighbors: Vec<(usize, u32, bool)> = Vec::with_capacity(ef + 1);

        // (idx, score) of every visited node
        let mut fullset: Vec<(usize, u32)> = Vec::new();

        for &neighbor in &self.nnd_graph[enter_point].neighbors {
            if knn_neighbors.len() >= ef {
                break;
            }

            let neighbor_feature_index = neighbor as usize;
            scratch.mark(neighbor_feature_index);
            let score = self
                .distance
                .calculate_distance(&self.features[neighbor_feature_index], query_id);
            let pos = knn_neighbors.partition_point(|n| n.1 <= score);
            knn_neighbors.insert(pos, (neighbor_feature_index, score, true));
            fullset.push((neighbor_feature_index, score));
        }

        let ef = ef.min(self.features.len()); // In case features.len() < ef
        while knn_neighbors.len() < ef {
            let neighbor_feature_index = self.random_node();
            if scratch.visit(neighbor_feature_index) {
                let score = self
                    .distance
                    .calculate_distance(&self.features[neighbor_feature_index], query_id);
                let pos = knn_neighbors.partition_point(|n| n.1 <= score);
                knn_neighbors.insert(pos, (neighbor_feature_index, score, true));
                fullset.push((neighbor_feature_index, score));
            }
        }

        // Greedy best-first search
        let mut current_neighbor_to_expand = 0;
        while current_neighbor_to_expand < ef {
            let mut earliest_insertion = ef;
            let (candidate, _, needs_expansion) = knn_neighbors[current_neighbor_to_expand];
            if needs_expansion {
                knn_neighbors[current_neighbor_to_expand].2 = false;
                for &neighbor in &self.nnd_graph[candidate].neighbors {
                    let neighbor_feature_index = neighbor as usize;
                    if scratch.visit(neighbor_feature_index) {
                        let score = self
                            .distance
                            .calculate_distance(&self.features[neighbor_feature_index], query_id);
                        fullset.push((neighbor_feature_index, score));

                        if score < knn_neighbors[ef - 1].1 {
                            let pos = knn_neighbors.partition_point(|n| n.1 <= score);
                            if knn_neighbors.len() == ef {
                                knn_neighbors.pop();
                            }
                            knn_neighbors.insert(pos, (neighbor_feature_index, score, true));

                            if pos < earliest_insertion {
                                earliest_insertion = pos;
                            }
                        }
                    }
                }
            }

            if earliest_insertion <= current_neighbor_to_expand {
                current_neighbor_to_expand = earliest_insertion;
            } else {
                current_neighbor_to_expand += 1;
            }
        }

        let retset = knn_neighbors
            .into_iter()
            .map(|(i, d, _)| (i, d))
            .collect();

        (retset, fullset)
    }

    fn init_graph(&mut self)
    where
        ID: Medoid,
    {
        let EF = self.ef;
        let center = ID::medoid(&self.features);
        self.navigating_node = self.random_node();
        debug!("init_graph: random entry point → {}", self.navigating_node);
        let results = self.knn_search(&center, 1, EF);
        if let Some(&(_, idx, _)) = results.first() {
            self.navigating_node = idx;
        }
        debug!("init_graph: navigating_node -> {}", self.navigating_node);
    }

    fn sync_prune(&self, node: usize, candidates: &mut Vec<(usize, u32)>) -> Vec<(usize, u32)> {
        let M = self.m;
        let C = self.c;
        let mut present: HashSet<usize> = candidates.iter().map(|&(idx, _)| idx).collect();
        for &neighbor in &self.nnd_graph[node].neighbors {
            let neighbor_feature_index = neighbor as usize;
            if !present.insert(neighbor_feature_index) {
                continue;
            }
            let score = self.distance.calculate_distance(
                &self.features[neighbor_feature_index],
                &self.features[node],
            );
            candidates.push((neighbor_feature_index, score));
        }

        candidates.sort_unstable_by_key(|&(_, distance)| distance);

        let mut selected: Vec<(usize, u32)> = Vec::with_capacity(M);
        let mut pruned: Vec<(usize, u32)> = Vec::new();

        if candidates.is_empty() {
            return selected;
        }

        let mut start = 0;
        if candidates[start].0 == node {
            start += 1;
        }
        if start >= candidates.len() {
            return selected;
        }
        selected.push(candidates[start]);

        loop {
            start += 1;
            if selected.len() >= M || start >= candidates.len() || start >= C {
                break;
            }

            let (cand_idx, cand_score) = candidates[start];
            let mut occlude = false;

            for &(sel_idx, _) in &selected {
                if sel_idx == cand_idx {
                    occlude = true;
                    break;
                }
                let d = self
                    .distance
                    .calculate_distance(&self.features[sel_idx], &self.features[cand_idx]);
                // [MRNG]
                if (d as f32) * self.alpha < cand_score as f32 {
                    occlude = true;
                    break;
                }
            }

            if occlude {
                pruned.push((cand_idx, cand_score));
            } else {
                selected.push((cand_idx, cand_score));
            }
        }

        // keepPrunedConnections: refill to M, nearest-first
        for c in pruned {
            if selected.len() >= M {
                break;
            }
            selected.push(c);
        }

        selected
    }

    // selected is the result of calling sync_prune for node
    // cut_graph is a temporal graph
    // for every edge [node -> neighbor], it adds the edge [neighbor -> node]
    fn inter_insert(&self, node: usize, selected: &Vec<(usize, u32)>, cut_graph: &mut Vec<Vec<(usize, u32)>>) {
        let M = self.m;
        for &(neighbor, distance) in selected {
            if cut_graph[neighbor].iter().any(|&(idx, _)| idx == node) {
                continue;
            }

            cut_graph[neighbor].push((node, distance));

            if cut_graph[neighbor].len() > M {
                let mut temp_pool = std::mem::take(&mut cut_graph[neighbor]);
                temp_pool.sort_unstable_by_key(|&(_, distance)| distance);

                let mut result: Vec<(usize, u32)> = Vec::with_capacity(M);
                let mut pruned: Vec<(usize, u32)> = Vec::new();
                let mut start = 0;
                result.push(temp_pool[start]);

                loop {
                    start += 1;
                    if result.len() >= M || start >= temp_pool.len() {
                        break;
                    }

                    let (cand_idx, cand_score) = temp_pool[start];
                    let mut occlude = false;

                    for &(sel_idx, _) in &result {
                        if sel_idx == cand_idx {
                            occlude = true;
                            break;
                        }
                        let d = self
                            .distance
                            .calculate_distance(&self.features[sel_idx], &self.features[cand_idx]);
                        if (d as f32) * self.alpha < cand_score as f32 {
                            occlude = true;
                            break;
                        }
                    }

                    if occlude {
                        pruned.push((cand_idx, cand_score));
                    } else {
                        result.push((cand_idx, cand_score));
                    }
                }

                // keepPrunedConnections: refill to M, nearest-first
                for c in pruned {
                    if result.len() >= M {
                        break;
                    }
                    result.push(c);
                }

                cut_graph[neighbor] = result;
            }
        }
    }

    fn link(&mut self) {
        let EF = self.ef;
        let n = self.features.len();
        let mut cut_graph: Vec<Vec<(usize, u32)>> = vec![Vec::new(); n];
        let mut scratch = SearchScratch::new(n);
        debug!("link: building NSG edges for {} nodes", n);
        let profile = std::env::var("NSG_PROFILE").is_ok();

        let t = std::time::Instant::now();
        for node in 0..n {
            let node_feature = self.features[node].clone();
            let (_, mut fullset) = self.get_neighbors(&node_feature, EF, &mut scratch);
            cut_graph[node] = self.sync_prune(node, &mut fullset);
        }
        let t_prune = t.elapsed();

        let t = std::time::Instant::now();
        for node in 0..n {
            let selected = std::mem::take(&mut cut_graph[node]);
            self.inter_insert(node, &selected, &mut cut_graph);
            cut_graph[node] = selected;
        }
        let t_insert = t.elapsed();

        // Write cut_graph back into nnd_graph
        debug!("link: writing edges back into nnd_graph");
        let t = std::time::Instant::now();
        for node in 0..n {
            let selected = &cut_graph[node];
            let node_entry = &mut self.nnd_graph[node];

            node_entry.neighbors = selected.iter().map(|&(idx, _)| idx as u32).collect();
            node_entry.neighbor_distances = selected.iter().map(|&(_, d)| d).collect();
        }
        let t_writeback = t.elapsed();

        if profile {
            eprintln!(
                "[profile][nsg/link]   get_neighbors+sync_prune={:.3?}  inter_insert={:.3?}  writeback={:.3?}",
                t_prune, t_insert, t_writeback,
            );
        }
    }

    fn tree_grow(&mut self) {
        let n = self.features.len();
        // Stores if a node is reachable from the root in the DFS tree
        let mut flags = vec![false; n];
        let mut root = self.navigating_node;
        let mut unlinked_count = 0;
        let mut scratch = SearchScratch::new(n);
        debug!("tree_grow: checking connectivity from navigating_node={}", self.navigating_node);

        while unlinked_count < n {
            self.dfs(&mut flags, root, &mut unlinked_count);
            debug!("tree_grow: {}/{} nodes reachable", unlinked_count, n);
            if unlinked_count >= n {
                break;
            }
            root = self.find_root(&mut flags, &mut scratch);
            debug!("tree_grow: new root → {}", root);
        }

        let max_degree = self.nnd_graph
            .iter()
            .map(|node| node.neighbor_count())
            .max()
            .unwrap_or(0);
        debug!("tree_grow: all {} nodes connected — max degree = {}", n, max_degree);
    }

    fn dfs(&self, flags: &mut Vec<bool>, root: usize, count: &mut usize) {
        let mut stack: Vec<(usize, usize)> = Vec::new();
        if !flags[root] {
            *count += 1;
        }
        flags[root] = true;
        stack.push((root, 0));

        while let Some(&(node, cursor)) = stack.last() {
            let neighbors = &self.nnd_graph[node].neighbors;
            let mut next = None;
            let mut c = cursor;
            while c < neighbors.len() {
                let neighbor = neighbors[c] as usize;
                c += 1;
                if !flags[neighbor] {
                    next = Some(neighbor);
                    break;
                }
            }
            stack.last_mut().unwrap().1 = c; // Update cursor for this node

            match next {
                Some(neighbor) => {
                    flags[neighbor] = true;
                    *count += 1;
                    stack.push((neighbor, 0));
                }
                None => {
                    stack.pop();
                }
            }
        }
    }

    fn find_root(&mut self, flags: &mut Vec<bool>, scratch: &mut SearchScratch) -> usize {
        let EF = self.ef;
        // Find first unlinked node
        let unlinked_node = match flags.iter().position(|&flag| !flag) {
            Some(idx) => idx,
            None => return self.navigating_node, // All linked
        };
        debug!("find_root: connecting unlinked node {}", unlinked_node);

        let unlinked_node_feature = self.features[unlinked_node].clone();
        let (_, mut fullset) = self.get_neighbors(&unlinked_node_feature, EF, scratch);
        fullset.sort_unstable_by_key(|&(_, distance)| distance);

        let root = fullset.iter()
            .find(|&&(idx, _)| flags[idx])
            .map(|&(idx, _)| idx)
            .unwrap_or_else(|| loop {
                let r = self.random_node();
                if flags[r] {
                    break r;
                }
            });

        debug!("find_root: linking {} → {}", root, unlinked_node);

        let distance = self
            .distance
            .calculate_distance(&self.features[root], &self.features[unlinked_node]);

        let node = &mut self.nnd_graph[root];
        node.neighbors.push(unlinked_node as u32);
        node.neighbor_distances.push(distance);

        root
    }

    pub fn build(&mut self, features: Vec<ID>, cfg: &BuildConfig) -> std::io::Result<()>
    where
        ID: Medoid,
    {
        self.features = features;
        self.m = cfg.nsg.m;
        self.c = cfg.nsg.c;
        self.ef = cfg.nsg.ef;
        self.alpha = if cfg.nsg.alpha.is_finite() && cfg.nsg.alpha > 0.0 {
            cfg.nsg.alpha
        } else {
            let fallback = NsgParams::default().alpha;
            debug!(
                "build: ignoring invalid alpha={} — using the default {}",
                cfg.nsg.alpha, fallback
            );
            fallback
        };
        self.prng = StdRng::seed_from_u64(cfg.seed);

        if self.features.is_empty() {
            self.nnd_graph.clear();
            self.navigating_node = usize::MAX;
            return Ok(());
        }

        let iter = cfg.iter;
        let profile = std::env::var("NSG_PROFILE").is_ok();

        // Build KNN graph with NNDescent
        debug!("Building KNN graph with NNDescent (iter={iter})...");
        let start = std::time::Instant::now();
        // let mut nnd = NNDescent::<D, ID, 50, 400, 10, 200>::new(&self.features);
        // let built = nnd.build(iter);

        let mut nnd = NNDescent::<D, ID>::new_with_seed(&self.features, cfg.nnd, cfg.seed);
        // VP-tree initialisation parameters: number of randomized trees and leaf size
        let (n_trees, leaf_size) = (cfg.n_trees, cfg.leaf_size);
        let built = match cfg.init {
            NndInit::Random => nnd.build(iter),
            NndInit::MetricTree => nnd.build_with_metric_tree(iter, n_trees, leaf_size),
        };
        let t_nndescent = start.elapsed();

        debug!("Built in {:.3}s", start.elapsed().as_secs_f64());

        let t = std::time::Instant::now();
        self.nnd_graph = built
            .iter()
            .enumerate()
            .map(|(node_idx, neighbors)| {
                let mut node = NsgNode::new_empty(node_idx as u32);
                node.neighbors = neighbors.iter().map(|&(idx, _)| idx as u32).collect();
                node.neighbor_distances = neighbors.iter().map(|&(_, dist)| dist).collect();
                node
            })
            .collect();
        let t_convert = t.elapsed();

        debug!("build: loaded {} nodes from NNDescent", self.nnd_graph.len());

        debug!("build: computing navigating node");
        let t = std::time::Instant::now();
        self.init_graph();
        let t_init_graph = t.elapsed();

        debug!("build: linking graph (sync_prune + inter_insert)");
        let t = std::time::Instant::now();
        self.link();
        let t_link = t.elapsed();

        debug!("build: ensuring full connectivity");
        let t = std::time::Instant::now();
        self.tree_grow();
        let t_tree_grow = t.elapsed();

        let max = self.nnd_graph.iter().map(|n| n.neighbor_count()).max().unwrap_or(0);
        let min = self.nnd_graph.iter().map(|n| n.neighbor_count()).min().unwrap_or(0);
        let avg = self.nnd_graph.iter().map(|n| n.neighbor_count()).sum::<usize>() / self.nnd_graph.len();
        debug!("build: done — degree stats: max={max}, min={min}, avg={avg}");

        if profile {
            let total = t_nndescent + t_convert + t_init_graph + t_link + t_tree_grow;
            let pct = |d: std::time::Duration| 100.0 * d.as_secs_f64() / total.as_secs_f64().max(1e-9);
            eprintln!("[profile][nsg] ===== build breakdown (n={}) =====", self.nnd_graph.len());
            eprintln!("[profile][nsg]   nndescent (knn graph) = {:>10.3?}  ({:5.1}%)", t_nndescent, pct(t_nndescent));
            eprintln!("[profile][nsg]   convert -> nnd_graph   = {:>10.3?}  ({:5.1}%)", t_convert, pct(t_convert));
            eprintln!("[profile][nsg]   init_graph (entrypt)   = {:>10.3?}  ({:5.1}%)", t_init_graph, pct(t_init_graph));
            eprintln!("[profile][nsg]   link (prune+insert)    = {:>10.3?}  ({:5.1}%)", t_link, pct(t_link));
            eprintln!("[profile][nsg]   tree_grow (connect)    = {:>10.3?}  ({:5.1}%)", t_tree_grow, pct(t_tree_grow));
            eprintln!("[profile][nsg]   TOTAL build            = {:>10.3?}", total);
        }

        Ok(())
    }

    pub fn knn_search(&mut self, query_id: &ID, k: usize, ef: usize) -> Vec<(u32, usize, &ID)> {
        if self.features.is_empty() {
            return Vec::new();
        }

        let mut scratch = SearchScratch::new(self.features.len());
        scratch.begin(self.features.len());

        // (index, score, needs_expansion)
        let mut knn_neighbors: Vec<(usize, u32, bool)> = Vec::with_capacity(ef + 1);

        let enter_point = self.navigating_node as usize;

        for &neighbor in &self.nnd_graph[enter_point].neighbors {
            if knn_neighbors.len() >= ef {
                break;
            }

            let neighbor_feature_index = neighbor as usize;
            scratch.mark(neighbor_feature_index);
            let score = self
                .distance
                .calculate_distance(&self.features[neighbor_feature_index], query_id);
            let pos = knn_neighbors.partition_point(|n| n.1 <= score);
            knn_neighbors.insert(pos, (neighbor_feature_index, score, true));
        }

        let ef = ef.max(k).min(self.features.len());
        while knn_neighbors.len() < ef {
            let neighbor_feature_index = self.random_node();
            if scratch.visit(neighbor_feature_index) {
                let score = self
                    .distance
                    .calculate_distance(&self.features[neighbor_feature_index], query_id);
                let pos = knn_neighbors.partition_point(|n| n.1 <= score);
                knn_neighbors.insert(pos, (neighbor_feature_index, score, true));
            }
        }

        // Greedy best-first search
        let mut current_neighbor_to_expand = 0;
        while current_neighbor_to_expand < ef {
            let mut earliest_insertion = ef;

            let (candidate, _, needs_expansion) = knn_neighbors[current_neighbor_to_expand];
            if needs_expansion {
                knn_neighbors[current_neighbor_to_expand].2 = false;

                for &neighbor in &self.nnd_graph[candidate].neighbors {
                    let neighbor_feature_index = neighbor as usize;
                    if scratch.visit(neighbor_feature_index) {
                        let score = self
                            .distance
                            .calculate_distance(&self.features[neighbor_feature_index], query_id);

                        if score < knn_neighbors[ef - 1].1 {
                            let pos = knn_neighbors.partition_point(|n| n.1 <= score);
                            if knn_neighbors.len() == ef {
                                knn_neighbors.pop();
                            }
                            knn_neighbors.insert(pos, (neighbor_feature_index, score, true));

                            if pos < earliest_insertion {
                                earliest_insertion = pos;
                            }
                        }
                    }
                }
            }

            if earliest_insertion <= current_neighbor_to_expand {
                current_neighbor_to_expand = earliest_insertion;
            } else {
                current_neighbor_to_expand += 1;
            }
        }

        knn_neighbors
            .into_iter()
            .take(k)
            .map(|(index, distance, _)| (distance, index, &self.features[index]))
            .collect()
    }
}

