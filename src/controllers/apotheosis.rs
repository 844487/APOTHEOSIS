// FIXME
use crate::datalayer::record::{self, RadixKeyMapping};

use crate::controllers::nsg::Nsg;
use crate::controllers::radix_tree::RadixNode;
use crate::datalayer::algorithms::DistanceAlgorithm;
use crate::datalayer::algorithms::Centroid;
use crate::datalayer::record::ApotheosisRecord;
use gexf::{Edge, EdgeType, Gexf, Node as GefxNode};
use std::fs::{self};
use std::path::{Path, PathBuf};

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "R: ApotheosisRecord + serde::Serialize, D: DistanceAlgorithm<R::MetricId> + serde::Serialize, R::MetricId: serde::Serialize",
    deserialize = "R: ApotheosisRecord + serde::Deserialize<'de>, D: DistanceAlgorithm<R::MetricId> + serde::Deserialize<'de>, R::MetricId: serde::Deserialize<'de>"
))]
pub struct Apotheosis<R, D, const M: usize = 16, const C: usize = 500, const EF: usize = 400>
where
    R: ApotheosisRecord,
    D: DistanceAlgorithm<R::MetricId> + Default,
{
    pub nsg: Nsg<D, R::MetricId, M, C, EF>,
    pub radix: RadixNode<u8, Option<usize>>,
    pub records: Vec<R>,
}

impl<R, D, const M: usize, const C: usize, const EF: usize> Apotheosis<R, D, M, C, EF>
where
    R: ApotheosisRecord,
    D: DistanceAlgorithm<R::MetricId> + Default,
    R::MetricId: Centroid
{

    pub fn new() -> Self {
        Self {
            // hnsw: Hnsw::new(),
            nsg: Nsg::new(),
            radix: RadixNode::<u8, Option<usize>>::new(vec![], None),
            records: vec![],
        }
    }

    pub fn insert(&mut self, records: Vec<R>) -> bool {
        let features: Vec<R::MetricId> = records    
            .iter()
            .map(|r| r.search_id())
            .collect();

        self.records = records;

        self.nsg.set_features(features);

        let _ = self.nsg.build("built.txt");

        for (index, record) in self.records.iter().enumerate() {
            if let Some(key) = record.search_id().to_radix_key() {
                if self.radix.find(&key).is_some() {
                    println!("Key already exists in radix tree: {:?}", key);
                    continue;
                }
                self.radix.insert(key, Some(index));
            }
        }

        true
    }

    /// Performs an approximate k-NN search. If the `MetricId` natively maps to a Radix Tree key 
    /// (e.g. TLSH string), it jumps directly to the NSG node to retrieve neighbors, bypassing full search.
    /// Otherwise, it performs a standard NSG k-NN search using the `query`.
    ///
    /// # Parameters
    /// * `query` - The native HNSW numerical or hash representation
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Returns
    /// * `Vec<(u32, &R)>` - List of tuples containing:
    ///   - `u32`: Distance/score to the query
    ///   - `&R`: Reference to the actual retrieved Record item
    pub fn search(
        &mut self, // FIXME: self must be a mutable reference
        query: &R::MetricId,
        k: usize,
        ef_search: Option<usize>,
    ) -> Vec<(u32, &R)> {
        let ef_search = ef_search.unwrap_or(24);

        let nsg_results: Vec<(u32, usize, &R::MetricId)> = if let Some(key) = query.to_radix_key() {
            if let Some(radix_node) = self.radix.find(&key) {
                if let Some(Some(node_index)) = radix_node.data {
                    self.nsg.get_neighbors_node(node_index)
                } else {
                    self.nsg.knn_search(query, k, ef_search)
                }
            } else {
                self.nsg.knn_search(query, k, ef_search)
            }
        } else {
            self.nsg.knn_search(query, k, ef_search)
        };

        nsg_results
            .into_iter()
            .map(|(distance, index, _id)| (distance, &self.records[index]))
            .collect()
    }

    /// Dumps the Apotheosis model to a binary file using Bincode.
    pub fn dump<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>>
    where
        Self: serde::Serialize,
    {
        let encoded = bincode::serialize(self)?;
        fs::write(path, encoded)?;
        Ok(())
    }
}
