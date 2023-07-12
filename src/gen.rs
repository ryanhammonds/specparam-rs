// Simulations
use ndarray::Array1;
use std::f64::consts::E;
use rand;
use rand_distr::{Normal, Distribution};

// Lorentzian (log10)
pub fn lorentzian(f: &Array1<f64>, mut fk: f64, mut x: f64, mut b: f64) -> Array1<f64> {
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

// Bounded functions
fn apply_bound(mut param : f64, lower : f64, upper : f64) -> f64{
    if param < lower {
        param = lower;
    } else if param > upper {
        param = upper;
    }
    param
}

pub fn lorentzian_bounded(
        f: &Array1<f64>,
        mut fk: f64,
        mut x: f64,
        mut b: f64,
        lower: &Array1<f64>,
        upper: &Array1<f64>
    ) -> Array1<f64> {
    // SpecParam python leaves aperiodic unbounded from -inf to inf
    //fk = apply_bound(fk, lower[0], upper[0]);
    //x = apply_bound(x, lower[1], upper[1]);
    //b = apply_bound(b, lower[2], upper[2]);
    lorentzian(&f, fk, x, b)
}

pub fn linear_bounded(
        f: &Array1<f64>,
        mut x: f64,
        mut b: f64,
        lower: &Array1<f64>,
        upper: &Array1<f64>
    ) -> Array1<f64> {
        // SpecParam python leaves aperiodic unbounded from -inf to inf
        //x = apply_bound(x, lower[0], upper[0]);
        //b = apply_bound(b, lower[1], upper[1]);
        linear(&f, x, b)
}

// Gaussian peaks (log10)
pub fn peak_bounded(
        f: &Array1<f64>,
        mut ctr : f64,
        mut hgt : f64,
        mut wid : f64,
        lower: &Vec<f64>,
        upper: &Vec<f64>
    ) -> Array1<f64> {
    ctr = apply_bound(ctr, lower[0], upper[0]);
    hgt = apply_bound(hgt, lower[1], upper[1]);
    wid = apply_bound(wid, lower[2], upper[2]);
    peak(&f, ctr, hgt, wid)
}
