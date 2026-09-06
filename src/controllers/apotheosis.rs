use crate::datalayer::record::{RadixKeyMapping};

use crate::controllers::nsg::{BuildConfig, Nsg};
use crate::controllers::radix_tree::RadixNode;
use crate::datalayer::algorithms::DistanceAlgorithm;
use crate::datalayer::algorithms::Medoid;
use crate::datalayer::record::ApotheosisRecord;
// use gexf::{Edge, EdgeType, Gexf, Node as GefxNode};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "R: ApotheosisRecord + serde::Serialize, D: DistanceAlgorithm<R::MetricId> + serde::Serialize, R::MetricId: serde::Serialize",
    deserialize = "R: ApotheosisRecord + serde::Deserialize<'de>, D: DistanceAlgorithm<R::MetricId> + serde::Deserialize<'de>, R::MetricId: serde::Deserialize<'de>"
))]
pub struct Apotheosis<R, D>
where
    R: ApotheosisRecord,
    D: DistanceAlgorithm<R::MetricId> + Default,
{
    pub nsg: Nsg<D, R::MetricId>,
    pub radix: RadixNode<u8, Option<usize>>,
    pub records: Vec<R>,
    #[serde(skip, default)]
    pub config: BuildConfig,
}

impl<R, D> Apotheosis<R, D>
where
    R: ApotheosisRecord,
    D: DistanceAlgorithm<R::MetricId> + Default,
    R::MetricId: Medoid
{

    pub fn new(config: BuildConfig) -> Self {
        Self {
            nsg: Nsg::new(),
            radix: RadixNode::<u8, Option<usize>>::new(vec![], None),
            records: vec![],
            config,
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.config.seed = seed;
        self.nsg.set_seed(seed);
    }

    pub fn insert(&mut self, records: Vec<R>) -> bool {
        self.records = Vec::with_capacity(records.len());
        let mut features: Vec<R::MetricId> = Vec::with_capacity(records.len());

        for record in records {
            let feature = record.search_id();

            match feature.to_radix_key() {
                Some(key) => {
                    if self.radix.find(&key).is_some() {
                        // println!("Key already exists in radix tree: {:?}", key);
                        continue;
                    }

                    let index = self.records.len();
                    self.radix.insert(key, Some(index));
                    features.push(feature);
                    self.records.push(record);
                }
                None => {
                    features.push(feature);
                    self.records.push(record);
                }
            }
        }

        let cfg = self.config;
        let _ = self.nsg.build(features, &cfg);

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
        &mut self,
        query: &R::MetricId,
        k: usize,
        ef_search: Option<usize>,
    ) -> Vec<(u32, &R)> {
        let ef_search = ef_search.unwrap_or_else(|| self.nsg.default_ef());

        let nsg_results: Vec<(u32, usize, &R::MetricId)> = if let Some(key) = query.to_radix_key() {
            if let Some(radix_node) = self.radix.find(&key) {
                if let Some(Some(node_index)) = radix_node.data {
                    let mut results = self.nsg.knn_search(query, k, ef_search);
                    if !results.iter().any(|&(_, i, _)| i == node_index) {
                        results.insert(0, (0, node_index, query));
                        results.truncate(k);
                    }

                    results
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
        let mut file = File::create(path)?;
        file.write_all(b"APOT")?;
        bincode::serialize_into(file, self)?;
        Ok(())
    }

    /// Loads a model previously written by `dump`.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: serde::de::DeserializeOwned,
    {
        let mut file = File::open(path)?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"APOT" {
            return Err("Invalid Apotheosis model file (missing magic bytes)".into());
        }

        let model = bincode::deserialize_from(file)?;
        Ok(model)
    }
}

