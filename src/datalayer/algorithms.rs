use tlsh2::TlshDefault;

pub trait DistanceAlgorithm<ID> {
    fn calculate_distance(&self, a: &ID, b: &ID) -> u32;
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct NormalDistance;
impl DistanceAlgorithm<u32> for NormalDistance {
    fn calculate_distance(&self, a: &u32, b: &u32) -> u32 {
        a.abs_diff(*b)
    }
}

#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct TlshDistance;
impl DistanceAlgorithm<TlshDefault> for TlshDistance {
    fn calculate_distance(&self, a: &TlshDefault, b: &TlshDefault) -> u32 {
        let diff = a.diff(&b, true);
        diff as u32
    }
}

pub trait Centroid: Sized + Clone {
    fn centroid(features: &[Self]) -> Self;
}

// We compute the medoid, a data point within the cluster that is the most 
// representative and centrally located point, minimizing the average distance 
// to all other points in the cluster. 
impl Centroid for TlshDefault {
    fn centroid(features: &[Self]) -> Self {
        let step = (features.len() / 256).max(1);
        let sample: Vec<&Self> = features.iter().step_by(step).collect();
        features.iter().step_by(step)
            .min_by_key(|f| {
                sample.iter().map(|s| f.diff(s, true) as u64).sum::<u64>()
            })
            .unwrap()
            .clone()
    }
}

// Again, we compute the medoid
impl Centroid for u32 {
    fn centroid(features: &[Self]) -> Self {
        let step = (features.len() / 256).max(1);
        let sample: Vec<&Self> = features.iter().step_by(step).collect();
        features.iter().step_by(step)
            .min_by_key(|&&f| {
                sample.iter().map(|&&s| f.abs_diff(s) as u64).sum::<u64>()
            })
            .copied()
            .unwrap()
    }
}
