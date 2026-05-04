use crate::datalayer::algorithms::DistanceAlgorithm;
use crate::datalayer::nodes::NsgNode;
use core::cmp::min;
use std::thread::current;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::{cmp, u32};
use std::collections::HashSet;
use tracing::debug;


fn default_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

// M: max out-degree
// C: max candidates considered in sync prune
// EF: candidate pool size for greedy search
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "ID: serde::Serialize, D: DistanceAlgorithm<ID> + serde::Serialize",
    deserialize = "ID: serde::Deserialize<'de>, D: DistanceAlgorithm<ID> + serde::Deserialize<'de>"
))]
pub struct Nsg<D, ID, const M: usize, const C: usize, const EF: usize = 400>
where
    D: DistanceAlgorithm<ID> + Default,
{
    features: Vec<ID>,
    nnd_graph: Vec<NsgNode<M>>,
    navigating_node: usize, // TODO: u32? 
    #[serde(skip, default = "default_rng")]
    prng: StdRng,
    distance: D,
}

impl<D, ID, const M: usize, const C: usize, const EF: usize> Nsg<D, ID, M, C, EF>
where
    D: DistanceAlgorithm<ID> + Default,
{
    pub fn new() -> Self {
        Self {
            features: vec![],
            nnd_graph: vec![],
            navigating_node: usize::MAX,
            prng: StdRng::seed_from_u64(42),
            distance: D::default(),
        }
    }

    // TODO: Provisional, just to work like NSG
    pub fn load_fvecs(path: &str) -> std::io::Result<Vec<Vec<f32>>> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut features = Vec::new();
        let mut buf4 = [0u8; 4];
    
        loop {
            if file.read_exact(&mut buf4).is_err() { break; }
            let dim = u32::from_le_bytes(buf4) as usize;
    
            let mut vec = vec![0f32; dim];
            let byte_slice = unsafe {
                std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, dim * 4)
            };
            file.read_exact(byte_slice)?;
            features.push(vec);
        }
    
        Ok(features)
    }

    // TODO: Again, provisional
    pub fn load_nn_graph(path: &str) -> std::io::Result<Vec<NsgNode<M>>> {
        use std::io::Read;
    
        let mut file = std::fs::File::open(path)?;
        let mut buf4 = [0u8; 4];
    
        // First 4 bytes = K (same for all nodes)
        file.read_exact(&mut buf4)?;
        let k = u32::from_le_bytes(buf4) as usize;
        debug!("KNN graph K={k}");
    
        // Seek back — each node repeats its own k
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(0))?;
    
        let mut nnd_graph: Vec<NsgNode<M>> = Vec::new();
        loop {
            if file.read_exact(&mut buf4).is_err() { break; }
            let node_k = u32::from_le_bytes(buf4) as usize;
    
            let mut node = NsgNode::new_empty(nnd_graph.len() as u32);
            node.neighbor_count = node_k.min(M) as u16;
    
            for i in 0..node_k {
                file.read_exact(&mut buf4)?;
                let nb = u32::from_le_bytes(buf4);
                if i < M { node.neighbors[i] = nb; }
            }
            debug!(
                "node {}: {} neighbors → {:?}",
                nnd_graph.len(),
                node.neighbor_count,
                &node.neighbors[..node.neighbor_count as usize]
            );
            nnd_graph.push(node);
        }
    
        Ok(nnd_graph)
    }

    pub fn from_nn_graph(nn_graph_path: &str, fvecs_path: &str, distance: D) -> std::io::Result<Self>
    where
        ID: From<Vec<f32>>,
    {
        debug!("Before reading features");
        let features: Vec<ID> = Self::load_fvecs(fvecs_path)?
            .into_iter()
            .map(ID::from)
            .collect();
    
        let nnd_graph = Self::load_nn_graph(nn_graph_path)?;
    
        Ok(Self {
            features,
            nnd_graph,
            navigating_node: usize::MAX, // We will have to compute this later
            prng: StdRng::seed_from_u64(42),
            distance,
        })
    }

    pub fn knn_search_exhaustive(&self, query_id: &ID, k: usize, ef: usize) -> Vec<(u32, usize)> {
        let mut visited_neighbors: HashSet<usize> = HashSet::new();
        // TODO: hnsw??
        visited_neighbors.insert(self.navigating_node);

        let mut candidates: Vec<usize> = vec![self.navigating_node];

        let (enter_point, score) = {
            let initial_distance = self
                .distance
                .calculate_distance(&self.features[self.navigating_node], query_id);
            (self.navigating_node, initial_distance)
        };

        let mut knn_neigbors: Vec<(usize, u32)> = vec![(enter_point, score)];

        while let Some(candidate) = candidates.pop() {
            let neighbors = self.nnd_graph[candidate].active_neighbors();

            for neighbor in neighbors {
                let neighbor_feature_index = *neighbor as usize;

                if visited_neighbors.insert(neighbor_feature_index) {
                    let score = self
                        .distance
                        .calculate_distance(&self.features[neighbor_feature_index], query_id);

                    let pos = knn_neigbors.partition_point(|n| n.1 <= score);
                    if pos != ef {
                        if knn_neigbors.len() == ef {
                            knn_neigbors.pop();
                        }
                        knn_neigbors.insert(pos, (neighbor_feature_index, score));
                        candidates.push(neighbor_feature_index);
                    }
                }
            }
        }
        knn_neigbors
            .into_iter()
            .take(k)
            .map(|(index, distance)| (distance, index))
            .collect()
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
        let mut visited_neighbors: HashSet<usize> = HashSet::new();
        // TODO: Insert enter_point?

        let mut knn_neighbors: Vec<(usize, u32, bool)> = Vec::with_capacity(ef + 1);

        // (idx, score) de todos los nodos visitados
        let mut fullset: Vec<(usize, u32)> = Vec::new();
    
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
            fullset.push((neighbor_feature_index, score));
        }
    
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
                            if pos != ef {
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
            }

            if earliest_insertion <= current_neighbor_to_expand {
                current_neighbor_to_expand = earliest_insertion;
            } else {
                current_neighbor_to_expand += 1;
            }
        }
    
        let retset = knn_neighbors
            .into_iter()
            .map(|(i, d, _)| (i, d)).
            collect();

        (retset, fullset)
    }

    fn sync_prune(&self, node: usize, candidates: &mut Vec<(usize, u32)>) -> Vec<(usize, u32)> {
        for neighbor in self.nnd_graph[node].active_neighbors() {
            let neighbor_feature_index = *neighbor as usize;
            if candidates.iter().any(|(idx, _)| *idx == neighbor_feature_index) {
                continue;
            }
            
            let score = self
                .distance
                .calculate_distance(&self.features[neighbor_feature_index], &self.features[node]);

            candidates.push((neighbor_feature_index, score));
        }

        candidates.sort_unstable_by_key(|&(_, distance)| distance);

        let mut selected: Vec<(usize, u32)> = Vec::with_capacity(M);
        let mut start = 0;

        // If node is the best candidate, we skip to the next best candidate
        if candidates[start].0 == node {
            start += 1;
        }

        selected.push(candidates[start]);

        // For each next candidate p (ordered by distance to node), we accept it
        // only if there is no neighbor r in selected such that dist(r, p) < dist(node, p)
        // (we could reach p via r) [MRNG]
        'outer: for i in (start + 1)..candidates.len().min(C) {
            if selected.len() >= M {
                break;
            }

            let (candidate_feature_index, candidate_score) = candidates[i];
            for &(selected_feature_index, _) in &selected {
                let distance_selected_to_candidate = self
                    .distance
                    .calculate_distance(&self.features[selected_feature_index], &self.features[candidate_feature_index]);

                if distance_selected_to_candidate < candidate_score {
                    continue 'outer
                }
            }
            selected.push((candidate_feature_index, candidate_score));
        }

        selected
    }




    pub fn knn_search(&mut self, query_id: &ID, k: usize, ef: usize) -> Vec<(u32, usize)> {
        let mut visited_neighbors: HashSet<usize> = HashSet::new();
        // TODO: Insert enter_point?

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
            // TODO: Is it faster to do an unstable sort at the end?
            let pos = knn_neighbors.partition_point(|n| n.1 <= score);
            knn_neighbors.insert(pos, (neighbor_feature_index, score, true));
        }

        // TODO: What if our graph has less than ef nodes?
        while knn_neighbors.len() < ef {
            let neighbor_feature_index = self.random_node();
            if visited_neighbors.insert(neighbor_feature_index) {
                let score = self
                    .distance
                    .calculate_distance(&self.features[neighbor_feature_index], query_id);
                // TODO: Is it faster to do an unstable sort at the end?
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
                            if pos != ef {
                                if knn_neighbors.len() == ef {
                                    knn_neighbors.pop();
                                }
                                knn_neighbors.insert(pos, (neighbor_feature_index, score, true));
                            }

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
            .map(|(index, distance, _)| (distance, index))
            .collect()
    }
}
