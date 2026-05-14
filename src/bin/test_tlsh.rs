use apotheosis2::controllers::apotheosis::Apotheosis;
use apotheosis2::datalayer::algorithms::TlshDistance;
use apotheosis2::datalayer::record::{ApotheosisRecord, SimpleTlshRecord};
use serde_json::Value;
use std::fs;
use std::str::FromStr;
use std::time::Instant;
use tlsh2::TlshDefault;

fn read_hashes_from_json<P: AsRef<std::path::Path>>(path: P) -> Vec<String> {
    let data = fs::read_to_string(path).expect("Failed to read JSON file");
    let v: Value = serde_json::from_str(&data).expect("Failed to parse JSON");

    v.as_array()
        .map(|array| {
            array.iter()
                .filter_map(|item| {
                    item.get("TLSH")
                        .and_then(|t| t.as_str())
                        .filter(|&s| s != "TNULL") 
                        .map(|s| s.to_string())
                })
                .collect()
        })
        .unwrap_or_else(Vec::new)
}

fn create_tlsh_object(hash: String) -> TlshDefault {
    TlshDefault::from_str(&hash).unwrap()
}

// cargo run --bin test_tlsh
pub fn main() {
    let hashes = read_hashes_from_json("file_hashes.json");
    println!("Number of hashes: {:?}", hashes.len());

    // Initialize vectors before pushing
    let dataset: Vec<String> = hashes[..42000].to_vec();
    let dataset_copy: Vec<String> = dataset.clone();
    let queries: Vec<String> = hashes[42000..43000].to_vec();
    let mut apotheosis = Apotheosis::<SimpleTlshRecord, TlshDistance, 32, 200, 64>::new();
    let creation_start: Instant = Instant::now();

    println!(
        "Dataset size: {}, Queries size: {}",
        dataset.len(),
        queries.len()
    );

    let mut records= vec![];

    for f in dataset_copy {
        println!("{:?}", f);
        records.push(SimpleTlshRecord::create(f));
    }

    println!("Inserting into NSG model...");
    apotheosis.insert(records);

    let query_hash = TlshDefault::from_str(
        "T1008100007FFA5C48F0F33EB5AEB455158576FE205AB2CA6D51A4828F24B2B408961F3B",
    )
    .unwrap();
    let result = apotheosis.search(&query_hash, 1, None);
    println!("Distance: {}", result[0].0);
    let creation_time: std::time::Duration = creation_start.elapsed();

    let mut brute_results: Vec<(u32, TlshDefault)> = Vec::new();
    let mut apo_results: Vec<(u32, TlshDefault)> = Vec::new();

    let brute_start = Instant::now();

    println!("Starting brute force search...");
    // --- Brute Force Search ---
    for query in queries.iter() {
        let mut closest: (u32, Option<&String>) = (u32::MAX, None);
        let tlsh_query = create_tlsh_object(query.to_string());
        for candidate in dataset.iter() {
            let tlsh_candidate = create_tlsh_object(candidate.to_string());

            let diff = tlsh_query.diff(&tlsh_candidate, true) as u32;
            if diff < closest.0 {
                closest.0 = diff;
                closest.1 = Some(&candidate);
            }
        }

        if let Some(hash) = closest.1 {
            let obj = create_tlsh_object(hash.to_string());
            brute_results.push((closest.0, obj));
        }
    }

    let brute_time: std::time::Duration = brute_start.elapsed();

    let nsg_start = Instant::now();

    println!("Starting APOTHEOSIS search...");

    for n in &queries {
        let record_hash = TlshDefault::from_str(n).unwrap();
        let results = apotheosis.search(&record_hash, 42, None);
        apo_results.push((results[0].0, results[0].1.search_id()));
    }

    let nsg_time: std::time::Duration = nsg_start.elapsed();

    let mut matches = 0;
    for i in 0..apo_results.len() {
        if apo_results[i].0 == brute_results[i].0
            && apo_results[i].1.hash() == brute_results[i].1.hash()
        {
            matches += 1;
        }
    }

    println!("Matches: {}/{}", matches, apo_results.len());
    println!("Creation time: {:?}", creation_time);
    println!("Brute force time: {:?}", brute_time);
    println!("NSG search time: {:?}", nsg_time);
}
