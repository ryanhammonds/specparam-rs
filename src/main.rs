
use std::time::Instant;
use ndarray::{Array1};

mod gen;
mod optimizers;
mod specparam;

use gen::{lorentzian, linear, peak, noise};
use specparam::SpecParam;

fn main() {

    // Lorentzian Model
    // Create frequencies
    let freqs : Array1<f64> = Array1::range(1., 101., 1.0);

    // Create powers
    let mut powers : Array1<f64> = lorentzian(&freqs, 10.0, 2.0, 100.0);
    let sig_noise : Array1<f64> = noise(&freqs, 0.05);

    powers = (powers + sig_noise).mapv(|p| (10.0_f64).powf(p));
    powers = powers + peak(&freqs, 20.0, 20.0, 2.0);

    // Time fitting procedure
    println!("Lorentzian/Knee Results:");
    let now = Instant::now();
    {
        let mut sp = SpecParam{
            max_n_peaks: 1,
            aperiodic_mode: "lorentzian".to_string(),
            ..Default::default()
        };
        let _results = sp.fit(&freqs, &powers);
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    // Compute MSE and R^2
    let mut sp = SpecParam{
        max_n_peaks: 1,
        aperiodic_mode: "lorentzian".to_string(),
        ..Default::default()
    };
    let results = sp.fit(&freqs, &powers);
    println!("MSE: {:?}", sp.compute_error(&results.powers_log, &results.powers_log_fit));
    println!("R^2: {:?}", sp.compute_rsq(&results.powers_log, &results.powers_log_fit));


    // Linear Model
    println!("Linear/(1/f) Results:");
    let mut powers : Array1<f64> = linear(&freqs, 2.0, 2.0);
    let sig_noise : Array1<f64> = noise(&freqs, 0.05);
    powers = (powers + sig_noise).mapv(|p| (10.0_f64).powf(p));
    powers = powers + peak(&freqs, 20.0, 20.0, 2.0);

    let mut sp = SpecParam{
        max_n_peaks: 1,
        aperiodic_mode: "linear".to_string(),
        ..Default::default()
    };

    let now = Instant::now();
    {
        let mut sp = SpecParam{
            max_n_peaks: 1,
            aperiodic_mode: "linear".to_string(),
            ..Default::default()
        };
        let _results = sp.fit(&freqs, &powers);
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    // Compute MSE and R^2
    let mut sp = SpecParam{
        max_n_peaks: 1,
        aperiodic_mode: "linear".to_string(),
        ..Default::default()
    };
    let results = sp.fit(&freqs, &powers);
    println!("MSE: {:?}", sp.compute_error(&results.powers_log, &results.powers_log_fit));
    println!("R^2: {:?}", sp.compute_rsq(&results.powers_log, &results.powers_log_fit));
}

