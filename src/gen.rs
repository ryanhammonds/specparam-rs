// Simulations
use ndarray::Array1;
use std::f64::consts::E;
use rand;
use rand_distr::{Normal, Distribution};

// Lorentzian (log10)
pub fn lorentzian(f: &Array1<f64>, mut fk: f64, mut x: f64, mut b: f64) -> Array1<f64> {
    if fk <= 0.0 {
        fk = 1e-6;
    }
    if x <= 0.0 {
        x = 1e-6;
    }
    if b <= 0.0 {
        b = 1e-6;
    }
    let base : f64 = (x * fk.log10()) + b.log10();
    let fkx : f64 = fk.powf(x);
    base - f.map(|f| (fkx + f.powf(x)).log10())
}

// Linear, 1/f (log10)
pub fn linear(f: &Array1<f64>, x: f64, b: f64) -> Array1<f64> {
    f.map(|f| b - (x * (f).log10()))
}

// Gaussian peaks (log10)
pub fn peak(f: &Array1<f64>, ctr : f64, hgt : f64, wid : f64) -> Array1<f64> {
    hgt * (f).map(|f| E.powf(-(f-ctr).powi(2) / wid.powi(2)))
}

pub fn noise(f: &Array1<f64>, scale: f64) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, scale);
    let mut powers: Array1<f64> = Array1::zeros(f.len());
    for i in 0..f.len() {
        powers[i] = normal.expect("REASON").sample(&mut rng);
    }
    powers
}
