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

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "ID: serde::Serialize, D: DistanceAlgorithm<ID> + serde::Serialize",
    deserialize = "ID: serde::Deserialize<'de>, D: DistanceAlgorithm<ID> + serde::Deserialize<'de>"
))]
pub struct Nsg<D, ID, const M: usize, const EF: usize = 400>
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

impl<D, ID, const M: usize, const EF: usize> Nsg<D, ID, M, EF>
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
