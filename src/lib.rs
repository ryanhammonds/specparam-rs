
mod gen;
mod optimizers;
mod specparam;
use specparam::SpecParam;
use ndarray::{Array1};

use wasm_bindgen::prelude::*;
use js_sys::Float64Array;


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