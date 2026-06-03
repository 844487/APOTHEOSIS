use crate::datalayer::algorithms::DistanceAlgorithm;
use crate::datalayer::algorithms::Centroid;
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

// M: max out-degree (R in the reference NSG)
// C: max candidates considered in sync prune
// EF: candidate pool size for greedy search
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "ID: serde::Serialize, D: DistanceAlgorithm<ID> + serde::Serialize",
    deserialize = "ID: serde::Deserialize<'de>, D: DistanceAlgorithm<ID> + serde::Deserialize<'de>"
))]
pub struct Nsg<D, ID, const M: usize, const C: usize, const EF: usize = 400>
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
    alpha: f32, // TODO: Move this
}

impl<D, ID, const M: usize, const C: usize, const EF: usize> Nsg<D, ID, M, C, EF>
where
    ID: Clone,
    D: DistanceAlgorithm<ID> + Default,
{
    pub fn new() -> Self {
        Self {
            features: vec![],
            nnd_graph: vec![],
            navigating_node: usize::MAX,
            prng: StdRng::seed_from_u64(42),
            distance: D::default(),
            alpha: 1.2,
        }
    }

    #[inline]
    fn random_node(&mut self) -> usize {
        let n = self.features.len();
        self.prng.next_u64() as usize % n
    }

    fn get_neighbors(
        &mut self,
        query_id: &ID,
        ef: usize,
    ) -> (Vec<(usize, u32)>, Vec<(usize, u32)>) {
        let enter_point = self.navigating_node as usize;

        let mut visited_neighbors: HashSet<usize> = HashSet::new();

        let mut knn_neighbors: Vec<(usize, u32, bool)> = Vec::with_capacity(ef + 1);

        // (idx, score) of every visited node
        let mut fullset: Vec<(usize, u32)> = Vec::new();

        for neighbor in self.nnd_graph[enter_point].active_neighbors() {
            if knn_neighbors.len() >= ef {
                break;
            }

            let neighbor_feature_index = *neighbor as usize;
            visited_neighbors.insert(neighbor_feature_index);
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
            if visited_neighbors.insert(neighbor_feature_index) {
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
                for neighbor in self.nnd_graph[candidate].active_neighbors() {
                    let neighbor_feature_index = *neighbor as usize;
                    if visited_neighbors.insert(neighbor_feature_index) {
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
        ID: Centroid,
    {
        let center = ID::centroid(&self.features);
        self.navigating_node = self.random_node();
        debug!("init_graph: random entry point → {}", self.navigating_node);
        let results = self.knn_search(&center, 1, EF);
        if let Some(&(_, idx, _)) = results.first() {
            self.navigating_node = idx;
        }
        debug!("init_graph: navigating_node -> {}", self.navigating_node);
    }

    fn sync_prune(&self, node: usize, candidates: &mut Vec<(usize, u32)>) -> Vec<(usize, u32)> {
        for neighbor in self.nnd_graph[node].active_neighbors() {
            let neighbor_feature_index = *neighbor as usize;
            if candidates.iter().any(|(idx, _)| *idx == neighbor_feature_index) {
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
        for &(neighbor, distance) in selected {
            let neighbor_pool = &cut_graph[neighbor];

            if neighbor_pool.iter().any(|&(idx, _)| idx == node) {
                continue;
            }

            let mut temp_pool = neighbor_pool.clone();
            temp_pool.push((node, distance));

            if temp_pool.len() > M {
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
            } else {
                cut_graph[neighbor] = temp_pool;
            }
        }
    }

    fn link(&mut self) {
        let n = self.features.len();
        let mut cut_graph: Vec<Vec<(usize, u32)>> = vec![Vec::new(); n];
        debug!("link: building NSG edges for {} nodes", n);

        for node in 0..n {
            let node_feature = self.features[node].clone();
            let (_, mut fullset) = self.get_neighbors(&node_feature, EF);
            cut_graph[node] = self.sync_prune(node, &mut fullset);
        }

        for node in 0..n {
            let selected = cut_graph[node].clone();
            self.inter_insert(node, &selected, &mut cut_graph);
        }

        // Write cut_graph back into nnd_graph
        debug!("link: writing edges back into nnd_graph");
        for node in 0..n {
            let selected = &cut_graph[node];
            let node_entry = &mut self.nnd_graph[node];

            node_entry.neighbors = selected.iter().map(|&(idx, _)| idx as u32).collect();
            node_entry.neighbor_distances = selected.iter().map(|&(_, d)| d).collect();
        }
    }

    fn tree_grow(&mut self) {
        let n = self.features.len();
        // Stores if a node is reachable from the root in the DFS tree
        let mut flags = vec![false; n];
        let mut root = self.navigating_node;
        let mut unlinked_count = 0;
        debug!("tree_grow: checking connectivity from navigating_node={}", self.navigating_node);

        while unlinked_count < n {
            self.dfs(&mut flags, root, &mut unlinked_count);
            debug!("tree_grow: {}/{} nodes reachable", unlinked_count, n);
            if unlinked_count >= n {
                break;
            }
            root = self.find_root(&mut flags);
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
        let mut stack = vec![root];
        if !flags[root] {
            *count += 1;
        }
        flags[root] = true;

        while let Some(node) = stack.last().copied() {
            let next = self.nnd_graph[node].active_neighbors()
                .iter()
                .map(|&neighbor| neighbor as usize)
                .find(|&neighbor| !flags[neighbor]);

            match next {
                Some(neighbor) => {
                    flags[neighbor] = true;
                    *count += 1;
                    stack.push(neighbor);
                }
                None => {
                    // All neighbors have already been visited in the search
                    stack.pop();
                }
            }
        }
    }

    fn find_root(&mut self, flags: &mut Vec<bool>) -> usize {
        // Find first unlinked node
        let unlinked_node = match flags.iter().position(|&flag| !flag) {
            Some(idx) => idx,
            None => return self.navigating_node, // All linked
        };
        debug!("find_root: connecting unlinked node {}", unlinked_node);

        let unlinked_node_feature = self.features[unlinked_node].clone();
        let (_, mut fullset) = self.get_neighbors(&unlinked_node_feature, EF);
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

    pub fn build(&mut self, features: Vec<ID>) -> std::io::Result<()>
    where
        ID: Centroid,
    {
        self.features = features;
        let iter = 10;

        // Build KNN graph with NNDescent
        debug!("Building KNN graph with NNDescent (iter={iter})...");
        let start = std::time::Instant::now();
        // let mut nnd = NNDescent::<D, ID, 50, 400, 10, 200>::new(&self.features);
        // let built = nnd.build(iter);

        let mut nnd = NNDescent::<D, ID, 50, 400, 10, 200>::new(&self.features);
        // VP-tree initialisation parameters: number of randomized trees and leaf size
        let (n_trees, leaf_size) = (16, 32);
        let built = nnd.build_with_metric_tree(iter, n_trees, leaf_size);

        debug!("Built in {:.3}s", start.elapsed().as_secs_f64());

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

        debug!("build: loaded {} nodes from NNDescent", self.nnd_graph.len());

        debug!("build: computing navigating node");
        self.init_graph();

        debug!("build: linking graph (sync_prune + inter_insert)");
        self.link();

        debug!("build: ensuring full connectivity");
        self.tree_grow();

        let max = self.nnd_graph.iter().map(|n| n.neighbor_count()).max().unwrap_or(0);
        let min = self.nnd_graph.iter().map(|n| n.neighbor_count()).min().unwrap_or(0);
        let avg = self.nnd_graph.iter().map(|n| n.neighbor_count()).sum::<usize>() / self.nnd_graph.len();
        debug!("build: done — degree stats: max={max}, min={min}, avg={avg}");

        Ok(())
    }

    pub fn knn_search(&mut self, query_id: &ID, k: usize, ef: usize) -> Vec<(u32, usize, &ID)> {
        let mut visited_neighbors: HashSet<usize> = HashSet::new();

        // (index, score, needs_expansion)
        let mut knn_neighbors: Vec<(usize, u32, bool)> = Vec::with_capacity(ef + 1);

        let enter_point = self.navigating_node as usize;

        for neighbor in self.nnd_graph[enter_point].active_neighbors() {
            if knn_neighbors.len() >= ef {
                break;
            }

            let neighbor_feature_index = *neighbor as usize;
            visited_neighbors.insert(neighbor_feature_index);
            let score = self
                .distance
                .calculate_distance(&self.features[neighbor_feature_index], query_id);
            let pos = knn_neighbors.partition_point(|n| n.1 <= score);
            knn_neighbors.insert(pos, (neighbor_feature_index, score, true));
        }

        let ef = ef.min(self.features.len());
        while knn_neighbors.len() < ef {
            let neighbor_feature_index = self.random_node();
            if visited_neighbors.insert(neighbor_feature_index) {
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

                for neighbor in self.nnd_graph[candidate].active_neighbors() {
                    let neighbor_feature_index = *neighbor as usize;
                    if visited_neighbors.insert(neighbor_feature_index) {
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

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;
        let mut buf4;

        // width (M)
        buf4 = (M as u32).to_le_bytes();
        file.write_all(&buf4)?;

        // navigating_node (ep_)
        buf4 = (self.navigating_node as u32).to_le_bytes();
        file.write_all(&buf4)?;

        // for each node: k + neighbors
        for node in &self.nnd_graph {
            buf4 = (node.neighbor_count() as u32).to_le_bytes();
            file.write_all(&buf4)?;
            for &nb in node.active_neighbors() {
                file.write_all(&nb.to_le_bytes())?;
            }
        }

        debug!("save: wrote {} nodes to {}", self.nnd_graph.len(), path);
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> std::io::Result<()> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)?;
        let mut buf4 = [0u8; 4];

        file.read_exact(&mut buf4)?;
        let width = u32::from_le_bytes(buf4) as usize;
        debug!("load: width={width}");
        if width != M {
            debug!("load: note — graph was built with R={width}, current M={M}");
        }

        file.read_exact(&mut buf4)?;
        self.navigating_node = u32::from_le_bytes(buf4) as usize;
        debug!("load: navigating_node={}", self.navigating_node);

        self.nnd_graph.clear();
        loop {
            if file.read_exact(&mut buf4).is_err() { break; }
            let k = u32::from_le_bytes(buf4) as usize;

            let mut node = NsgNode::new_empty(self.nnd_graph.len() as u32);
            node.neighbors.reserve(k);
            node.neighbor_distances.reserve(k);

            for _ in 0..k {
                file.read_exact(&mut buf4)?;
                let nb = u32::from_le_bytes(buf4);
                node.neighbors.push(nb);
                node.neighbor_distances.push(u32::MAX);
            }
            self.nnd_graph.push(node);
        }

        debug!("load: loaded {} nodes", self.nnd_graph.len());
        Ok(())
    }

    // TODO: Rethink this
    pub fn get_neighbors_node(&self, index: usize) -> Vec<(u32, usize, &ID)> {
        let mut results: Vec<(u32, usize, &ID)> = vec![];

        let node = &self.nnd_graph[index];

        let neigbors = node.active_neighbors();

        for neighbor_index in neigbors {
            let score = self.distance.calculate_distance(
                &self.features[*neighbor_index as usize],
                &self.features[node.feature_index as usize],
            );
            results.push((
                score,
                *neighbor_index as usize,
                &self.features[*neighbor_index as usize],
            ));
        }

        results.insert(0, (0, index, &self.features[index]));

        results
    }
}
