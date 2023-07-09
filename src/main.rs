
use std::time::Instant;
use ndarray::{array, Array1};

mod gen;
mod optimizers;
mod specparam;

use gen::{lorentzian, linear, peak, noise};
use specparam::SpecParam;

fn main() {

    // Lorentzian Model
    // Create frequencies

    // let freqs : Array1<f64> = Array1::range(1., 101., 1.0);

    // // Create powers
    // let mut powers : Array1<f64> = lorentzian(&freqs, 10.0, 2.0, 100.0);
    // let sig_noise : Array1<f64> = noise(&freqs, 0.05);

    // powers = (powers + sig_noise).mapv(|p| (10.0_f64).powf(p));
    // powers = powers + peak(&freqs, 20.0, 20.0, 2.0);

    let freqs : Array1<f64> = array![  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,
        12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,  22.,
        23.,  24.,  25.,  26.,  27.,  28.,  29.,  30.,  31.,  32.,  33.,
        34.,  35.,  36.,  37.,  38.,  39.,  40.,  41.,  42.,  43.,  44.,
        45.,  46.,  47.,  48.,  49.,  50.,  51.,  52.,  53.,  54.,  55.,
        56.,  57.,  58.,  59.,  60.,  61.,  62.,  63.,  64.,  65.,  66.,
        67.,  68.,  69.,  70.,  71.,  72.,  73.,  74.,  75.,  76.,  77.,
        78.,  79.,  80.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,  88.,
        89.,  90.,  91.,  92.,  93.,  94.,  95.,  96.,  97.,  98.,  99.,
        100.];

    let powers : Array1<f64> = array![1.22530070e+01, 2.61989166e+00, 1.24991903e+00, 8.29915114e-01,
        5.48750368e-01, 3.38975952e-01, 4.80795959e-01, 6.20558660e-01,
        9.30788933e-01, 1.04841603e+00, 6.41141333e-01, 3.32060567e-01,
        1.37089799e-01, 7.24889751e-02, 5.72643731e-02, 5.68706289e-02,
        8.72260225e-02, 1.21916700e-01, 2.19120571e-01, 2.26589066e-01,
        1.28946362e-01, 9.00256453e-02, 4.40979708e-02, 2.17672993e-02,
        2.29903005e-02, 1.28364229e-02, 1.38595104e-02, 1.24928121e-02,
        1.41867538e-02, 1.31591902e-02, 1.05931294e-02, 1.02001900e-02,
        8.29054472e-03, 6.88657076e-03, 7.84278243e-03, 7.85628829e-03,
        8.41635015e-03, 7.95395372e-03, 6.28895257e-03, 6.03854923e-03,
        5.27679885e-03, 4.82244660e-03, 4.45957977e-03, 6.51060621e-03,
        4.71679348e-03, 4.59728226e-03, 4.07556894e-03, 5.06360039e-03,
        3.83154346e-03, 4.56132082e-03, 4.35547644e-03, 5.32545952e-03,
        5.17086921e-03, 5.24272405e-03, 6.62416039e-03, 7.72885319e-03,
        8.11355633e-03, 8.90893481e-03, 8.25428023e-03, 8.42481632e-03,
        7.68805788e-03, 7.22432268e-03, 6.00200970e-03, 4.61743128e-03,
        4.85634938e-03, 3.83883292e-03, 2.84443508e-03, 3.14152079e-03,
        2.37621783e-03, 2.39921314e-03, 2.39000406e-03, 2.08858202e-03,
        2.22509235e-03, 1.62074920e-03, 1.88603198e-03, 1.61109193e-03,
        1.53116873e-03, 1.54040605e-03, 1.54715595e-03, 1.57324378e-03,
        1.33304971e-03, 1.64985159e-03, 1.53157986e-03, 1.18750161e-03,
        1.64276946e-03, 1.68189248e-03, 1.51321524e-03, 1.26484845e-03,
        1.11604673e-03, 1.39391912e-03, 1.15281169e-03, 1.36002408e-03,
        1.18426216e-03, 1.26641558e-03, 1.15443932e-03, 1.17702637e-03,
        1.06409777e-03, 1.27891409e-03, 1.03532146e-03, 1.04736840e-03];

    // Time fitting procedure
    println!("Lorentzian/Knee Results:");
    let now = Instant::now();
    {
        let mut sp = SpecParam{
            max_n_peaks: 3,
            aperiodic_mode: "linear".to_string(),
            ..Default::default()
        };
        let _results = sp.fit(&freqs, &powers);
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    // Compute MSE and R^2
    // let mut sp = SpecParam{
    //     max_n_peaks: 5,
    //     aperiodic_mode: "lorentzian".to_string(),
    //     ..Default::default()
    // };

    // let results = sp.fit(&freqs, &powers);




//     println!("MSE: {:?}", sp.compute_error(&results.powers_log, &results.powers_log_fit));
//     println!("R^2: {:?}", sp.compute_rsq(&results.powers_log, &results.powers_log_fit));


    // // Linear Model
    // println!("Linear/(1/f) Results:");
    // let mut powers : Array1<f64> = linear(&freqs, 2.0, 2.0);
    // let sig_noise : Array1<f64> = noise(&freqs, 0.05);
    // powers = (powers + sig_noise).mapv(|p| (10.0_f64).powf(p));
    // powers = powers + peak(&freqs, 20.0, 2.0, 2.0) + peak(&freqs, 60.0, 2.0, 1.0);


    // let now = Instant::now();
    // {
    //     let mut sp = SpecParam{
    //         max_n_peaks: 1,
    //         aperiodic_mode: "linear".to_string(),
    //         ..Default::default()
    //     };
    //     let _results = sp.fit(&freqs, &powers);
    // }
    // let elapsed = now.elapsed();
    // println!("Elapsed: {:.2?}", elapsed);

    // // Compute MSE and R^2
    // let mut sp = SpecParam{
    //     max_n_peaks: 5,
    //     aperiodic_mode: "linear".to_string(),
    //     ..Default::default()
    // };
    // let results = sp.fit(&freqs, &powers);
    // println!("{:?}", results.peak_params_);
//     println!("MSE: {:?}", sp.compute_error(&results.powers_log, &results.powers_log_fit));
//     println!("R^2: {:?}", sp.compute_rsq(&results.powers_log, &results.powers_log_fit));
}

