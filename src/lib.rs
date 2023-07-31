
mod gen;
mod optimizers;
mod specparam;
use specparam::SpecParam;
use ndarray::{Array1, Array2};

use wasm_bindgen::prelude::*;
use js_sys::Float64Array;


#[wasm_bindgen]
pub fn simulate(
    // Simulation settings
    ap_params: *mut f64,
    ap_size: usize,
    pe_params: *mut f64,
    pe_size: usize,
    noise_scale: f64,
    // Model settings
    peak_width_lower: f64,
    peak_width_upper: f64,
    max_n_peaks: f64,
    min_peak_height: f64,
    peak_threshold: f64,
    aperiodic_mode: bool,
    freq_res: f64
)
    -> js_sys::Array {

    let ap_params = unsafe { std::slice::from_raw_parts(ap_params, ap_size) };
    let pe_params = unsafe { std::slice::from_raw_parts(pe_params, pe_size) };

    let freqs = Array1::range(1.0, 101.0, freq_res);

    let powers_ap : Array1<f64> =
        if ap_params.len() == 2{
            gen::linear(&freqs, ap_params[0], ap_params[1])
        } else if ap_params.len() == 3{
            gen::lorentzian(&freqs, ap_params[0], ap_params[1], ap_params[2])
        } else {
            Array1::zeros(freqs.len())
        };

    let n_osc= (pe_params.len() / 3) as usize;
    let mut powers_pe : Array1<f64> = Array1::zeros(freqs.len());

    for i in 0..n_osc {

        let _peak = gen::peak(
            &freqs,
            pe_params[i*3],
            pe_params[(i*3)+1],
            pe_params[(i*3)+2]
        );

        powers_pe = powers_pe + _peak;
    };

    let powers_noise = gen::noise(&freqs, noise_scale);

    let mut powers : Array1<f64> = Array1::zeros((3*freqs.len()) as usize);
    let mut _powers : Array1<f64> = Array1::zeros(freqs.len());
    for i in 0..freqs.len(){
        let p = 10.0_f64.powf(powers_pe[i] + powers_ap[i] + powers_noise[i]);

        powers[i] = freqs[i];
        powers[freqs.len() + i] = p;

        _powers[i] = p;
    };

    let _ap_mode: String =
        if aperiodic_mode {
            "linear".to_string()
        } else {
            "lorentzian".to_string()
        };

    let _max_n_peaks = max_n_peaks as i64;

    let mut sp = SpecParam{
        peak_width_limits: (peak_width_lower, peak_width_upper),
        max_n_peaks: _max_n_peaks,
        min_peak_height: min_peak_height,
        peak_threshold: peak_threshold,
        aperiodic_mode: _ap_mode,
        ..Default::default()
    };

    let res = sp.fit(&freqs, &_powers);

    // Hack to return two arrays
    //   can only return one js_sys::Array at a time,
    //   so we concatenate the arrays and then split them in js

    for i in 0..freqs.len(){
        powers[(2*freqs.len()) + i] = 10.0_f64.powf(res.powers_log_fit[i]);
    }

    powers.clone().to_vec().into_iter().map(JsValue::from).collect()

}

#[wasm_bindgen]
pub fn fit(freqs_ptr: *mut f64, powers_ptr: *mut f64, size: usize) -> js_sys::Array {
    let freqs = unsafe { std::slice::from_raw_parts(freqs_ptr, size) };
    let powers = unsafe { std::slice::from_raw_parts(powers_ptr, size) };

    let freqs_nd = Array1::from_vec((&freqs).to_vec());
    let powers_nd = Array1::from_vec((&powers).to_vec());

    let mut sp = SpecParam{
        max_n_peaks: 1,
        aperiodic_mode: "linear".to_string(),
        ..Default::default()
    };

    let results = sp.fit(&freqs_nd, &powers_nd);
    results.powers_log_fit.map(|&p| 10.0_f64.powf(p)).to_vec().into_iter().map(JsValue::from).collect()
}

#[wasm_bindgen]
pub fn get_vec_pointer(count: usize) -> *mut f64 {
    let mut v = Vec::with_capacity(count);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr
}

#[wasm_bindgen]
pub fn get_array(ptr: *mut f64, count: usize) -> Float64Array {
    unsafe { Float64Array::view(std::slice::from_raw_parts(ptr, count)) }
}