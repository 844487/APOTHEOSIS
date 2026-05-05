// src/bin/test_nsg.rs
use apotheosis2::datalayer::algorithms::L2Distance;
use apotheosis2::controllers::nsg::Nsg;

fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt::init();

    let nn_graph_path = std::env::args().nth(1).expect("usage: test_nsg <nn_graph> <data.fvecs>");
    let fvecs_path    = std::env::args().nth(2).expect("usage: test_nsg <nn_graph> <data.fvecs>");

    println!("Building NSG...");
    let mut nsg = Nsg::<L2Distance, Vec<f32>, 500, 500>::new();
    nsg.build(&nn_graph_path, &fvecs_path)?;
    println!("Done!");

    Ok(())
}
