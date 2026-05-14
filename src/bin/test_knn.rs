use apotheosis2::datalayer::algorithms::L2Distance;
use apotheosis2::controllers::nndescent::NNDescent;
use apotheosis2::controllers::nsg::Nsg;
use std::io::Write;

fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt::init();

    let fvecs_path = std::env::args().nth(1).expect("usage: test_knn <data.fvecs> <nn_graph> <iter>");
    let nn_graph_path = std::env::args().nth(2).expect("usage: test_knn <data.fvecs> <nn_graph> <iter>");
    let iter: usize = std::env::args().nth(3).expect("missing iter").parse().expect("iter must be a number");

    // Load features
    println!("Loading features from {fvecs_path}...");
    let features: Vec<Vec<f32>> = Nsg::<L2Distance, Vec<f32>, 200, 500, 40>::load_fvecs(&fvecs_path)?;
    println!("Loaded {} features", features.len());

    // // Load reference KNN graph from file
    // println!("Loading reference KNN graph from {nn_graph_path}...");
    // let reference = Nsg::<L2Distance, Vec<f32>, 200, 500, 40>::load_nn_graph(&nn_graph_path)?;

    // // Write reference graph to text file
    // println!("Writing reference graph to reference.txt...");
    // let mut ref_file = std::fs::File::create("reference.txt")?;
    // for (node, nsg_node) in reference.iter().enumerate() {
    //     let neighbors: Vec<String> = nsg_node.active_neighbors()
    //         .iter().map(|nb| nb.to_string()).collect();
    //     writeln!(ref_file, "{node}: [{}]", neighbors.join(", "))?;
    // }

    // Build KNN graph with NNDescent
    println!("Building KNN graph with NNDescent (iter={iter})...");
    let start = std::time::Instant::now();
    let mut nnd = NNDescent::<L2Distance, Vec<f32>, 200, 200, 10, 100>::new(features);
    let built = nnd.build(iter);
    println!("Built in {:.3}s", start.elapsed().as_secs_f64());

    // Write built graph to text file
    println!("Writing built graph to built.txt...");
    let mut built_file = std::fs::File::create("built.txt")?;
    for (node, neighbors) in built.iter().enumerate() {
        let neighbors_str: Vec<String> = neighbors.iter().map(|&(idx, _dist): &(usize, u32)| idx.to_string()).collect();
        writeln!(built_file, "{node}: [{}]", neighbors_str.join(", "))?;
    }

    println!("Done!");
    Ok(())
}
