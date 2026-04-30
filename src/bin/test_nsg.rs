// src/bin/test_nsg.rs
use apotheosis2::datalayer::algorithms::DistanceAlgorithm;
use apotheosis2::controllers::nsg::Nsg;

// Provisional, I need to make sure NSG works properly
#[derive(Default)]
struct L2Distance;
impl DistanceAlgorithm<Vec<f32>> for L2Distance {
    fn calculate_distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> u32 {
        let dist: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        dist.sqrt() as u32
    }
}

fn main() -> std::io::Result<()> {
    let nsg_path   = std::env::args().nth(1).expect("usage: test_nsg <graph.nsg> <data.fvecs>");
    let fvecs_path = std::env::args().nth(2).expect("usage: test_nsg <graph.nsg> <data.fvecs>");

    println!("Loading...");
    let nsg = Nsg::<L2Distance, Vec<f32>, 100, 100>::from_knn_graph_features(
        &nsg_path,
        &fvecs_path,
        L2Distance,
    )?;

    println!("Loaded!");
    Ok(())
}
