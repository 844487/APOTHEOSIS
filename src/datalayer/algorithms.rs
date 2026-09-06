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

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SsdeepHash(pub String);

impl SsdeepHash {
    pub fn is_valid(&self) -> bool {
        !self.0.bytes().any(|b| b == 0) && ssdeep::compare(self.0.as_str(), self.0.as_str()).is_ok()
    }
}

#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct SsdeepDistance;
impl DistanceAlgorithm<SsdeepHash> for SsdeepDistance {
    fn calculate_distance(&self, a: &SsdeepHash, b: &SsdeepHash) -> u32 {
        100 - u32::from(ssdeep::compare(a.0.as_str(), b.0.as_str()).unwrap_or(0))
    }
}

pub trait Medoid: Sized + Clone {
    fn medoid(features: &[Self]) -> Self;
}

impl Medoid for TlshDefault {
    // Computes an approximate medoid over Tlsh features
    fn medoid(features: &[Self]) -> Self {
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

impl Medoid for u32 {
    // Computes an approximate medoid over u32 features
    fn medoid(features: &[Self]) -> Self {
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

impl Medoid for SsdeepHash {
    fn medoid(features: &[Self]) -> Self {
        let d = SsdeepDistance;
        let step = (features.len() / 256).max(1);
        let sample: Vec<&Self> = features.iter().step_by(step).collect();
        features.iter().step_by(step)
            .min_by_key(|f| {
                sample.iter().map(|s| d.calculate_distance(f, s) as u64).sum::<u64>()
            })
            .unwrap()
            .clone()
    }
}
