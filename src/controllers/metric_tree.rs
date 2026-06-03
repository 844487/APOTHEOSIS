use crate::datalayer::algorithms::DistanceAlgorithm;
use rand::rngs::StdRng;
use rand::RngCore;

// Randomized VP-tree (vantage-point tree) used to bootstrap the initial KNN
// candidate set for NN-descent
pub struct MetricTreeInit<'a, D, ID>
where
    D: DistanceAlgorithm<ID> + Default,
{
    features: &'a [ID],
    distance: D,
    n_trees: usize,
    leaf_size: usize,
}

impl<'a, D, ID> MetricTreeInit<'a, D, ID>
where
    D: DistanceAlgorithm<ID> + Default,
{
    pub fn new(features: &'a [ID], n_trees: usize, leaf_size: usize) -> Self {
        Self {
            features,
            distance: D::default(),
            n_trees: n_trees.max(1),
            leaf_size: leaf_size.max(2),
        }
    }

    // Pick a random pivot in the subrange, reorder `indices` by distance to that
    // pivot, and split at the median. Returns the number of points sent to the
    // left child. Because the split is always at count/2, both children are
    // non-empty whenever a node is split (count > leaf_size >= 2).
    fn vp_split(&self, indices: &mut [usize], rng: &mut StdRng) -> usize {
        let count = indices.len();
        let pivot = indices[rng.next_u64() as usize % count];

        // Distance from the pivot to every point in the subrange
        let mut scored: Vec<(u32, usize)> = indices
            .iter()
            .map(|&idx| {
                let d = self
                    .distance
                    .calculate_distance(&self.features[pivot], &self.features[idx]);
                (d, idx)
            })
            .collect();
        scored.sort_unstable_by_key(|&(d, _)| d);

        for (i, &(_, idx)) in scored.iter().enumerate() {
            indices[i] = idx;
        }

        count / 2
    }

    // Recursively split the index permutation; every leaf's members are pushed
    // into `leaves` as a group of mutual candidates.
    fn collect_leaves(&self, indices: &mut [usize], rng: &mut StdRng, leaves: &mut Vec<Vec<usize>>) {
        if indices.len() <= self.leaf_size {
            leaves.push(indices.to_vec());
            return;
        }

        let lim = self.vp_split(indices, rng);
        let (left, right) = indices.split_at_mut(lim);
        self.collect_leaves(left, rng, leaves);
        self.collect_leaves(right, rng, leaves);
    }

    // Build `n_trees` randomized VP-trees and return, for each node, the deduped
    // set of points that shared a leaf with it in at least one tree.
    pub fn candidate_neighbors(&self, rng: &mut StdRng) -> Vec<Vec<usize>> {
        let n = self.features.len();
        let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); n];
        if n == 0 {
            return candidates;
        }

        for _ in 0..self.n_trees {
            // Each tree starts from a fresh shuffled permutation so the trees
            // differ
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.next_u64() as usize % (i + 1);
                perm.swap(i, j);
            }

            let mut leaves: Vec<Vec<usize>> = Vec::new();
            self.collect_leaves(&mut perm, rng, &mut leaves);

            for leaf in &leaves {
                for &p in leaf {
                    for &q in leaf {
                        if p != q {
                            candidates[p].push(q);
                        }
                    }
                }
            }
        }

        for c in candidates.iter_mut() {
            c.sort_unstable();
            c.dedup();
        }

        candidates
    }
}
