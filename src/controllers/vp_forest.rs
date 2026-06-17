use crate::datalayer::algorithms::DistanceAlgorithm;
use rand::rngs::StdRng;
use rand::RngCore;

/// A randomized forest of VP-trees (vantage-point trees)
pub struct VpForest<'a, D, ID>
where
    D: DistanceAlgorithm<ID> + Default,
{
    features: &'a [ID], // Borrowed feature set
    distance: D,
    n_trees: usize, // Number of randomized trees to build
    leaf_size: usize, // A subrange this size or smaller becomes a leaf (splitting stops)
}

impl<'a, D, ID> VpForest<'a, D, ID>
where
    D: DistanceAlgorithm<ID> + Default,
{
    /// Creates a new VpForest over the given feature slice.
    ///
    /// # Parameters
    /// * `features` - The feature set to index. Borrowed for the forest's lifetime.
    /// * `n_trees` - Number of randomized VP-trees to build. Clamped to at least 1.
    /// * `leaf_size` - Maximum number of points in a leaf. Clamped to at least 2 so
    ///   that every split produces two non-empty children.
    ///
    /// # Returns
    /// * `Self` - A forest ready to produce candidate neighbors.
    pub fn new(features: &'a [ID], n_trees: usize, leaf_size: usize) -> Self {
        Self {
            features,
            distance: D::default(),
            n_trees: n_trees.max(1),
            leaf_size: leaf_size.max(2),
        }
    }

    /// Splits a subrange of `indices` around a random pivot (the vantage point).
    ///
    /// The points are reordered in place by their distance to the pivot, so the
    /// near half ends up on the left and the far half on the right. The split is
    /// always taken at the median (`count / 2`).
    ///
    /// # Parameters
    /// * `indices` - The subrange of point indices to reorder in place.
    /// * `rng` - Random source used to pick the pivot.
    /// * `scored` - Reusable scratch buffer of `(distance, index)` pairs.
    ///
    /// # Returns
    /// * `usize` - The number of points sent to the left child. Because the cut is
    ///   at `count / 2` and a node is only split when `count > leaf_size >= 2`, both
    ///   children are guaranteed non-empty.
    fn vp_split(
        &self,
        indices: &mut [usize],
        rng: &mut StdRng,
        scored: &mut Vec<(u32, usize)>,
    ) -> usize {
        let count = indices.len();

        // Pick a random vantage point from the current subrange
        let pivot = indices[rng.next_u64() as usize % count];

        scored.clear();
        scored.extend(indices.iter().map(|&idx| {
            let d = self
                .distance
                .calculate_distance(&self.features[pivot], &self.features[idx]);
            (d, idx)
        }));

        // Order the points from nearest to farthest from the pivot
        scored.sort_unstable_by_key(|&(d, _)| d);

        // Write the sorted order back into `indices` so the subrange is now
        // arranged by proximity to the pivot
        for (i, &(_, idx)) in scored.iter().enumerate() {
            indices[i] = idx;
        }

        // Split at the median
        count / 2
    }

    /// Recursively partitions `indices` into leaves and records each leaf's members
    /// as a group of mutually-close candidate neighbors.
    ///
    /// # Parameters
    /// * `indices` - The subrange to partition in place.
    /// * `rng` - Random source threaded down into every split.
    /// * `leaves` - Output: each leaf's members are pushed here as a group.
    /// * `scored` - Reusable scratch buffer shared with `vp_split`.
    fn collect_leaves(
        &self,
        indices: &mut [usize],
        rng: &mut StdRng,
        leaves: &mut Vec<Vec<usize>>,
        scored: &mut Vec<(u32, usize)>,
    ) {
        // Small enough: this subrange becomes a leaf, and its members are treated
        // as mutual candidates
        if indices.len() <= self.leaf_size {
            leaves.push(indices.to_vec());
            return;
        }

        // Otherwise split around a random vantage point and recurse into both halves
        let lim = self.vp_split(indices, rng, scored);
        let (left, right) = indices.split_at_mut(lim);
        self.collect_leaves(left, rng, leaves, scored);
        self.collect_leaves(right, rng, leaves, scored);
    }

    /// Builds `n_trees` randomized VP-trees and returns, for every point, the
    /// deduplicated set of points that shared a leaf with it in at least one tree.
    ///
    /// # Parameters
    /// * `rng` - Random source.
    ///
    /// # Returns
    /// * `Vec<Vec<usize>>` - For each point index, its sorted, deduped candidate list.
    pub fn candidate_neighbors(&self, rng: &mut StdRng) -> Vec<Vec<usize>> {
        let n = self.features.len();

        // One candidate list per point
        let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); n];

        if n == 0 {
            return candidates;
        }

        // Single distance buffer reused by every vp_split across every tree
        let mut scored: Vec<(u32, usize)> = Vec::with_capacity(n);

        for _ in 0..self.n_trees {
            // Each tree starts from a fresh shuffled permutation so
            // the trees differ from one another
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.next_u64() as usize % (i + 1);
                perm.swap(i, j);
            }

            // Partition this permutation into leaves
            let mut leaves: Vec<Vec<usize>> = Vec::new();
            self.collect_leaves(&mut perm, rng, &mut leaves, &mut scored);

            // Every pair of points that shared a leaf becomes a mutual candidate
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

        // Collapse the duplicates accumulated across the different trees
        for c in candidates.iter_mut() {
            c.sort_unstable();
            c.dedup();
        }

        candidates
    }
}
