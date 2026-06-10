use crate::controllers::metric_tree::MetricTreeInit;
use crate::datalayer::algorithms::DistanceAlgorithm;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use tracing::debug;

/// Runtime NN-descent sizes (see the comment on `Nhood`)
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct NndParams {
    pub k: usize,
    pub l: usize,
    pub s: usize,
    pub r: usize,
}

impl Default for NndParams {
    fn default() -> Self {
        Self { k: 50, l: 400, s: 10, r: 200 }
    }
}

// K: number of final neighbors per node
// L: candidate pool size per node
// S: number of initial random neighbors
// R: max reverse neighbors size
pub struct Nhood {
    // pool: (idx, dist, is_new)
    // is_new=true if neighbor was added in this iteration
    pub pool: Vec<(usize, u32, bool)>,
    pub nn_new: Vec<usize>, // New added candidate neighbors
    pub nn_old: Vec<usize>, // Old candidate neighbors at previous iterations
    pub rnn_new: Vec<usize>, // New added erverse candidate neighbors
    pub rnn_old: Vec<usize>, // Old reverse candidate neighbors
    pub m: usize, // Processed entries in this iteration
}

pub struct NNDescent<'a, D, ID>
where
    D: DistanceAlgorithm<ID> + Default,
{
    features: &'a [ID],
    graph: Vec<Nhood>,
    distance: D,
    prng: StdRng,
    params: NndParams,
}

#[allow(non_snake_case)]
impl<'a, D, ID> NNDescent<'a, D, ID>
where
    D: DistanceAlgorithm<ID> + Default,
{
    pub fn new(features: &'a [ID], params: NndParams) -> Self {
        let n = features.len();
        Self {
            features,
            graph: Vec::with_capacity(n),
            distance: D::default(),
            prng: StdRng::seed_from_u64(42),
            params,
        }
    }

    #[inline]
    fn random_node(&mut self) -> usize {
        let n = self.features.len();
        self.prng.next_u64() as usize % n
    }

    fn initialize_graph(&mut self) {
        let S = self.params.s;
        let n = self.features.len();
        self.graph.clear();

        for i in 0..n {
            // S random neighbors
            let mut nn_new: Vec<usize> = Vec::with_capacity(S);
            while nn_new.len() < S {
                let r = self.random_node();
                if r != i && !nn_new.contains(&r) {
                    nn_new.push(r);
                }
            }

            // Compute distances and fill pool
            let pool: Vec<(usize, u32, bool)> = nn_new.iter().map(|&nb| {
                let dist = self
                    .distance
                    .calculate_distance(&self.features[i], &self.features[nb]);
                (nb, dist, true)
            }).collect();

            self.graph.push(Nhood {
                pool,
                nn_new,
                nn_old: Vec::new(),
                rnn_new: Vec::new(),
                rnn_old: Vec::new(),
                m: 0,
            });
        }

        debug!("initialize_graph: {} nodes initialized with S={S} random neighbors", n);
    }

    fn insert(&mut self, node: usize, neighbor: usize, distance: u32) {
        let L = self.params.l;
        let pool = &mut self.graph[node].pool;

        // Pool is full and the candidate is no better than the current worst
        if pool.len() >= L && distance >= pool.last().unwrap().1 {
            return;
        }
    
        // Already in the pool
        if pool.iter().any(|&(idx, _, _)| idx == neighbor) { return; }
    
        let pos = pool.partition_point(|&(_, d, _)| d <= distance);
    
        if pool.len() < L {
            pool.insert(pos, (neighbor, distance, true));
        } else if pos < L {
            pool.pop();
            pool.insert(pos, (neighbor, distance, true));
        }
    }

    fn join(&mut self) {
        let n = self.features.len();
        for node in 0..n {
            let nn_new = self.graph[node].nn_new.clone();
            let nn_old = self.graph[node].nn_old.clone();
    
            for &i in &nn_new {
                for &j in &nn_new {
                    if i < j {
                        let d = self.distance.calculate_distance(&self.features[i], &self.features[j]);
                        self.insert(i, j, d);
                        self.insert(j, i, d);
                    }
                }
                for &j in &nn_old {
                    if i != j {
                        let d = self.distance.calculate_distance(&self.features[i], &self.features[j]);
                        self.insert(i, j, d);
                        self.insert(j, i, d);
                    }
                }
            }
        }
    }    

    fn update(&mut self) {
        let S = self.params.s;
        let L = self.params.l;
        let R = self.params.r;
        let n = self.features.len();
    
        // Clear nn_new/nn_old
        for node in 0..n {
            self.graph[node].nn_new.clear();
            self.graph[node].nn_old.clear();
        }
    
        for node in 0..n {
            // TODO: Pool is already sorted. Maybe do not insert at partition
            // point and then sort here?
            // self.graph[node].pool.sort_unstable_by_key(|&(_, d, _)| d);
            self.graph[node].pool.truncate(L);
    
            // Advance until we have S new neighbors
            let pool_len = self.graph[node].pool.len();
            let maxl = (self.graph[node].m + S).min(pool_len);
            let mut m = 0;
            let mut new_count = 0;
            while m < maxl && new_count < S {
                if self.graph[node].pool[m].2 {
                    new_count += 1;
                }
                m += 1;
            }
            self.graph[node].m = m;
        }
    
        let mut rnn_new_updates: Vec<(usize, usize)> = Vec::new();
        let mut rnn_old_updates: Vec<(usize, usize)> = Vec::new();
    
        // TODO: Parallelise this 
        for node in 0..n {
            let m = self.graph[node].m;
            for l in 0..m {
                let (neighbor_idx, neighbor_distance, is_new) = self.graph[node].pool[l];
                let worst_distance_in_neighbor_pool = self.graph[neighbor_idx].pool.last().map_or(u32::MAX, |&(_, d, _)| d);
    
                if is_new {
                    self.graph[node].nn_new.push(neighbor_idx);
                    self.graph[node].pool[l].2 = false;
                    // node distance to neighbor is worse than the the distance
                    // of the worst element of the neighbor, it is a great
                    // candidate
                    if neighbor_distance > worst_distance_in_neighbor_pool {
                        rnn_new_updates.push((neighbor_idx, node));
                    }
                } else {
                    self.graph[node].nn_old.push(neighbor_idx);
                    if neighbor_distance > worst_distance_in_neighbor_pool {
                        rnn_old_updates.push((neighbor_idx, node));
                    }
                }
            }
        }
    
        // Apply reverse neighbor updates (promising candidates)
        for (target, source) in rnn_new_updates {
            if self.graph[target].rnn_new.len() < R {
                self.graph[target].rnn_new.push(source);
            } else {
                // Reservoir sampling
                let pos = self.prng.next_u64() as usize % R;
                self.graph[target].rnn_new[pos] = source;
            }
        }
    
        for (target, source) in rnn_old_updates {
            if self.graph[target].rnn_old.len() < R {
                self.graph[target].rnn_old.push(source);
            } else {
                let pos = self.prng.next_u64() as usize % R;
                self.graph[target].rnn_old[pos] = source;
            }
        }
    
        for node in 0..n {
            let mut rnn_new = std::mem::take(&mut self.graph[node].rnn_new);
            let mut rnn_old = std::mem::take(&mut self.graph[node].rnn_old);
    
            // TODO: Shuffle rnn. This should not be reached (it is the same
            // in the C++ code). Explore parallelisation.
            if rnn_new.len() > R {
                debug!("update: rnn_new exceeded R={R}, shuffling");
                for i in 0..R {
                    let j = i + self.prng.next_u64() as usize % (rnn_new.len() - i);
                    rnn_new.swap(i, j);
                }
                rnn_new.truncate(R);
            }
            self.graph[node].nn_new.extend(rnn_new);
    
            if rnn_old.len() > R {
                // Shuffle
                for i in 0..R {
                    let j = i + self.prng.next_u64() as usize % (rnn_old.len() - i);
                    rnn_old.swap(i, j);
                }
                rnn_old.truncate(R);
            }
            self.graph[node].nn_old.extend(rnn_old);
    
            // Max size for old candidates is 2 * R
            if self.graph[node].nn_old.len() > R * 2 {
                self.graph[node].nn_old.truncate(R * 2);
            }
        }
    }

    pub fn build(&mut self, iter: usize) -> Vec<Vec<(usize, u32)>> {
        let K = self.params.k;
        let L = self.params.l;
        let S = self.params.s;
        let R = self.params.r;
        let n = self.features.len();
        debug!("build: initializing graph with {} nodes, K={K}, L={L}, S={S}, R={R}", n);
        self.initialize_graph();
    
        for it in 0..iter {
            debug!("build: iteration {}/{}", it + 1, iter);
            self.join();
            self.update();
        }
    
        debug!("build: extracting K={K} best neighbors");
        let mut final_graph: Vec<Vec<(usize, u32)>> = Vec::with_capacity(n);
        for node in 0..n {
            let neighbors = self.graph[node].pool.iter()
                .take(K)
                .map(|&(idx, dist, _)| (idx, dist))
                .collect();
            final_graph.push(neighbors);
        }
    
        debug!("build: done");
        final_graph
    }
}

// VP-tree (metric-tree) initialisation
// TODO: Check this
#[allow(non_snake_case)]
impl<'a, D, ID> NNDescent<'a, D, ID>
where
    D: DistanceAlgorithm<ID> + Default,
{
    fn initialize_graph_metric_tree(&mut self, n_trees: usize, leaf_size: usize) {
        let L = self.params.l;
        let S = self.params.s;
        let n = self.features.len();
        self.graph.clear();

        let mt = MetricTreeInit::<D, ID>::new(self.features, n_trees, leaf_size);
        let candidates = mt.candidate_neighbors(&mut self.prng);

        for i in 0..n {
            // Rank the VP-tree candidates by the real distance metric, keep L
            let mut scored: Vec<(usize, u32)> = candidates[i]
                .iter()
                .filter(|&&nb| nb != i)
                .map(|&nb| {
                    let dist = self
                        .distance
                        .calculate_distance(&self.features[i], &self.features[nb]);
                    (nb, dist)
                })
                .collect();
            scored.sort_unstable_by_key(|&(_, d)| d);
            scored.truncate(L);

            // Top up with random neighbors so every node still starts with at
            // least S new candidates
            let target = S.min(n.saturating_sub(1));
            while scored.len() < target {
                let r = self.random_node();
                if r != i && !scored.iter().any(|&(idx, _)| idx == r) {
                    let dist = self
                        .distance
                        .calculate_distance(&self.features[i], &self.features[r]);
                    let pos = scored.partition_point(|&(_, d)| d <= dist);
                    scored.insert(pos, (r, dist));
                }
            }

            // Full candidate list stays in the pool (already capped at L above),
            // but only the S nearest seed nn_new
            let nn_new: Vec<usize> = scored.iter().take(S).map(|&(idx, _)| idx).collect();
            let pool: Vec<(usize, u32, bool)> =
                scored.iter().map(|&(idx, d)| (idx, d, true)).collect();

            self.graph.push(Nhood {
                pool,
                nn_new,
                nn_old: Vec::new(),
                rnn_new: Vec::new(),
                rnn_old: Vec::new(),
                m: 0,
            });
        }

        debug!(
            "initialize_graph_metric_tree: {} nodes initialized from {} VP-trees (leaf_size={})",
            n, n_trees, leaf_size
        );
    }

    pub fn build_with_metric_tree(
        &mut self,
        iter: usize,
        n_trees: usize,
        leaf_size: usize,
    ) -> Vec<Vec<(usize, u32)>> {
        let K = self.params.k;
        let L = self.params.l;
        let S = self.params.s;
        let R = self.params.r;
        let n = self.features.len();
        debug!(
            "build_with_metric_tree: VP-tree init then NN-descent, n={}, K={K}, L={L}, S={S}, R={R}, n_trees={}, leaf_size={}",
            n, n_trees, leaf_size
        );
        self.initialize_graph_metric_tree(n_trees, leaf_size);

        for it in 0..iter {
            debug!("build_with_metric_tree: iteration {}/{}", it + 1, iter);
            self.join();
            self.update();
        }

        debug!("build_with_metric_tree: extracting K={K} best neighbors");
        let mut final_graph: Vec<Vec<(usize, u32)>> = Vec::with_capacity(n);
        for node in 0..n {
            let neighbors = self.graph[node]
                .pool
                .iter()
                .take(K)
                .map(|&(idx, dist, _)| (idx, dist))
                .collect();
            final_graph.push(neighbors);
        }

        debug!("build_with_metric_tree: done");
        final_graph
    }
}
