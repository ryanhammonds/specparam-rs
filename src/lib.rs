//
mod gen;
mod optimizers;
mod specparam;

use gen::{lorentzian};
use specparam::{SpecParam as SpecParamRS, Results as ResultsRS};

use pyo3::prelude::*;

use numpy::{IntoPyArray, PyArray1, PyArray2};
use ndarray::{Array1, Array2};


// Importable module
#[pymodule]
#[pyo3(name = "specparam")]
fn models(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SpecParam>()?;
    Ok(())
}

// Store results to be accessed from python class
#[pyclass]
pub struct Results {
    pub powers_log : Array1<f64>,
    pub powers_log_fit : Array1<f64>,
    pub aperiodic_params_ : Array1<f64>,
    pub gaussian_params_ : Array2<f64>,
    pub peak_params_ : Array2<f64>,
    pub _spectrum_flat : Array1<f64>,
    pub _spectrum_peak_rm : Array1<f64>,
    pub _ap_fit : Array1<f64>,
    pub _peak_fit : Array1<f64>,
}

// Access results as python class attributes
#[pymethods]
impl Results {
    #[getter]
    fn powers_log<'a>(&self, py: Python<'a>) -> &'a PyArray1<f64> {
        self.powers_log.clone().into_pyarray(py)
    }
    #[getter]
    fn powers_log_fit<'a>(&self, py: Python<'a>) -> &'a PyArray1<f64> {
        self.powers_log_fit.clone().into_pyarray(py)
    }
    #[getter]
    fn aperiodic_params_<'a>(&self, py: Python<'a>) -> &'a PyArray1<f64> {
        self.aperiodic_params_.clone().into_pyarray(py)
    }
    #[getter]
    fn peak_params_<'a>(&self, py: Python<'a>) -> &'a PyArray2<f64> {
        self.peak_params_.clone().into_pyarray(py)
    }
    #[getter]
    fn _spectrum_flat<'a>(&self, py: Python<'a>) -> &'a PyArray1<f64> {
        self._spectrum_flat.clone().into_pyarray(py)
    }
    #[getter]
    fn _spectrum_peak_rm<'a>(&self, py: Python<'a>) -> &'a PyArray1<f64> {
        self._spectrum_peak_rm.clone().into_pyarray(py)
    }
    #[getter]
    fn _ap_fit<'a>(&self, py: Python<'a>) -> &'a PyArray1<f64> {
        self._ap_fit.clone().into_pyarray(py)
    }
    #[getter]
    fn _peak_fit<'a>(&self, py: Python<'a>) -> &'a PyArray1<f64> {
        self._peak_fit.clone().into_pyarray(py)
    }
}

// Model class
#[pyclass]
struct SpecParam {
    pub specparam_rs : SpecParamRS
}

#[pymethods]
impl SpecParam {
    #[new]
    #[pyo3(signature = (
        peak_width_limits=(0.5, 12.0),
        max_n_peaks=100,
        min_peak_height=0.0,
        peak_threshold=2.0,
        aperiodic_mode="linear",
        verbose=true
    ))]
    fn new(
        peak_width_limits : (f64, f64),
        max_n_peaks : i64,
        min_peak_height : f64,
        peak_threshold : f64,
        aperiodic_mode : &str,
        verbose : bool
    ) -> Self {
        let mut sp = SpecParamRS{
            peak_width_limits : peak_width_limits,
            max_n_peaks : max_n_peaks,
            min_peak_height : min_peak_height,
            peak_threshold : peak_threshold,
            aperiodic_mode : aperiodic_mode.to_string(),
            verbose: verbose,
            ..Default::default()
        };

        SpecParam{
            specparam_rs : sp
        }
    }

    #[pyo3(text_signature = "(freqs, powers)")]
    fn fit<'a>(
        &mut self,
        freqs: &'a PyArray1<f64>,
        powers: &'a PyArray1<f64>
    )  -> Results {
        let f = unsafe { freqs.as_array() };
        let p = unsafe { powers.as_array() };

        let f = &f.to_owned();
        let p = &p.to_owned();

        let _results = self.specparam_rs.fit(&f, &p);

        let results = Results{
            powers_log : _results.powers_log,
            powers_log_fit : _results.powers_log_fit,
            aperiodic_params_ : _results.aperiodic_params_,
            gaussian_params_ : _results.gaussian_params_,
            peak_params_ : _results.peak_params_ ,
            _spectrum_flat : _results._spectrum_flat,
            _spectrum_peak_rm : _results._spectrum_peak_rm ,
            _ap_fit : _results._ap_fit,
            _peak_fit : _results._peak_fit,
        };
        results
    }
}