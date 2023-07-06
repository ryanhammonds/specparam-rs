
use std::time::Instant;
use ndarray::{array, Array1};

mod curves;
mod optimization;
mod specparam;

use curves::{lorentzian, peak, noise};
use specparam::SpecParam;

fn run() {

    // Create frequencies
    let freqs : Array1<f64> = Array1::range(1., 101., 1.0);

    // Create powers
    let mut powers : Array1<f64> = lorentzian(&freqs, 10.0, 2.0, 100.0);
    let noise : Array1<f64> = noise(&freqs, 0.05);

    powers = (powers + noise).mapv(|p| (10.0_f64).powf(p));
    powers = powers + peak(&freqs, 20.0, 20.0, 2.0);

    // Time fitting procedure
    let now = Instant::now();

    {
        //let mut sp = SpecParam::new();
        let mut sp = SpecParam{max_n_peaks: 1, ..Default::default()};
        let results = sp.fit(freqs, powers);
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
}


fn main() {
    run()
}