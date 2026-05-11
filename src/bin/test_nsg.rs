// src/bin/test_nsg.rs
use apotheosis2::datalayer::algorithms::L2Distance;
use apotheosis2::controllers::nsg::Nsg;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt::init();

    let fvecs_path    = std::env::args().nth(1).expect("usage: test_nsg <data.fvecs> <query.fvecs> <nsg_path> <nn_graph> <search_L> <search_K>");
    let query_path    = std::env::args().nth(2).expect("usage: test_nsg <data.fvecs> <query.fvecs> <nsg_path> <nn_graph> <search_L> <search_K>");
    let nsg_path      = std::env::args().nth(3).expect("usage: test_nsg <data.fvecs> <query.fvecs> <nsg_path> <nn_graph> <search_L> <search_K>");
    let nn_graph_path = std::env::args().nth(4).expect("usage: test_nsg <data.fvecs> <query.fvecs> <nsg_path> <nn_graph> <search_L> <search_K>");
    let search_l: usize = std::env::args().nth(5).expect("missing search_L").parse().expect("search_L must be a number");
    let search_k: usize = std::env::args().nth(6).expect("missing search_K").parse().expect("search_K must be a number");

    assert!(search_l >= search_k, "search_L cannot be smaller than search_K");

    println!("Loading features from {fvecs_path}...");
    let features: Vec<Vec<f32>> = Nsg::<L2Distance, Vec<f32>, 200, 500, 40>::load_fvecs(&fvecs_path)?;
    println!("Loaded {} features", features.len());

    let mut nsg = Nsg::<L2Distance, Vec<f32>, 200, 500, 40>::new();

    if std::path::Path::new(&nsg_path).exists() {
        println!("NSG file found, loading from {nsg_path}...");
        nsg.load(&nsg_path)?;
        nsg.set_features(features);
        println!("Loaded!");
    } else {
        println!("NSG file not found, building...");
        nsg.build(&nn_graph_path, &fvecs_path)?;
        println!("Built! Saving to {nsg_path}...");
        nsg.save(&nsg_path)?;
        println!("Saved!");
    }

    println!("Loading queries from {query_path}...");
    let queries = Nsg::<L2Distance, Vec<f32>, 200, 500, 40>::load_fvecs(&query_path)?;
    println!("Loaded {} queries", queries.len());

    println!("Searching top {search_k} neighbors with L={search_l}...");
    let start = Instant::now();
    let mut results: Vec<Vec<usize>> = Vec::with_capacity(queries.len());
    for query in &queries {
        let res = nsg.knn_search(query, search_k, search_l);
        results.push(res.into_iter().map(|(_, idx)| idx).collect());
    }
    let elapsed = start.elapsed();
    println!("search time: {:.3}s", elapsed.as_secs_f64());

    println!("\n--- Search Results (Top {search_k} Neighbors) ---");
    let print_limit = queries.len().min(20);
    for (i, res) in results.iter().take(print_limit).enumerate() {
        let indices: Vec<String> = res.iter().map(|idx| idx.to_string()).collect();
        println!("Query {i} nearest indices: [{}]", indices.join(", "));
    }
    if queries.len() > print_limit {
        println!("... and {} more queries.", queries.len() - print_limit);
    }
    println!("----------------------------------------------");

    Ok(())
}
