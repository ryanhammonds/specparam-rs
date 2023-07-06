use ndarray::{array, Array1, Array2, Axis, stack, s};
use ndarray_stats::{QuantileExt, Sort1dExt};


use crate::curves::{lorentzian, peak};
use crate::optimization::fit_lorentzian;

pub struct SpecParam {
    pub peak_width_limits : (f64, f64),
    pub max_n_peaks : i64,
    pub min_peak_height : f64,
    pub peak_threshold : f64,
    pub aperiodic_mode : String,
    pub verbose : bool,
    pub _internal_settings : InternalSettings
}

pub struct InternalSettings {
    _ap_percentile_thresh : f64,
    _ap_guess : (f64, f64, f64),
    _ap_bounds : ((f64, f64, f64), (f64, f64, f64)),
    _bw_std_edge : f64,
    _gauss_overlap_thresh : f64,
    _cf_bound : f64,
    _gauss_std_limits : (f64, f64)
}

pub struct Results {
    pub freqs : Array1<f64>,
    pub power_spectrum : Array1<f64>,
    // pub freqs : Array1<f64>,
    // pub freq_range : (f64, f64),
    // pub freq_res : f64,
    // pub power_spectrum : Array1<f64>,
    pub aperiodic_params_ : Array1<f64>,
    pub gaussian_params_ : Array2<f64>,
    // pub peak_params_ : Array1<f64>,
    // pub fooofed_spectrum_ : Array1<f64>,
    // pub _spectrum_flat : Array1<f64>,
    // pub _spectrum_peak_rm : Array1<f64>,
    // pub _ap_fit : Array1<f64>,
    // pub _peak_fit : Array1<f64>,
}

impl Default for SpecParam {
    fn default() -> SpecParam {
        SpecParam {
            // Public
            peak_width_limits : (0.0, 12.0),
            max_n_peaks : i64::MAX,
            min_peak_height : 0.0,
            peak_threshold : 2.0,
            aperiodic_mode : String::from("fixed"),
            verbose : true,
            // Private
            _internal_settings : InternalSettings {
                _ap_percentile_thresh : 0.025,
                _ap_guess : (0.0, 10.0, 2.0),
                _ap_bounds : (
                    (f64::MIN, f64::MIN, f64::MIN),
                    (f64::MAX, f64::MAX, f64::MAX)
                ),
                _bw_std_edge : 1.0,
                _gauss_overlap_thresh : 0.75,
                _cf_bound : 1.5,
                _gauss_std_limits : (0.0, 6.0)
            }
        }
    }
}

impl SpecParam {
    pub fn new() -> SpecParam {
        // Initalize with default parameters
        Default::default()
    }

    pub fn fit(&mut self, freqs : Array1<f64>, powers : Array1<f64>) -> Results {
        self._reset_internal_settings();
        let powers_log : Array1<f64> = powers.mapv(|p| p.log10());

        // Fit inital aperiodic
        let aperiodic_params_ = self._robust_ap_fit(&freqs, &powers, &powers_log);
        let _ap_fit = lorentzian(&freqs, aperiodic_params_[0], aperiodic_params_[1], aperiodic_params_[2]);
        let _spectrum_flat = &powers_log - _ap_fit;

        // Fit peaks
        let (gaussian_params_, _peak_fit) = self._fit_peaks(&freqs, &_spectrum_flat);
        let _spectrum_peak_rm_log = &powers_log - &_peak_fit;
        let _spectrum_peak_rm = _spectrum_peak_rm_log.mapv(|p| p.powf(10.0));

        // todo: fix this
        // Final aperiodic fit
        // let aperiodic_params_ = self._simple_ap_fit(&freqs, &_spectrum_peak_rm, &_spectrum_peak_rm_log);
        // let _ap_fit = lorentzian(&freqs, aperiodic_params_[0], aperiodic_params_[1], aperiodic_params_[2]);
        // let _spectrum_flat = &powers_log - &_ap_fit;
        // let fooofed_spectrum_ = &_peak_fit + &_ap_fit;

        //println!("{:?}", powers_log);
        //println!("{:?}", fooofed_spectrum_);

        Results {
            freqs : freqs,
            power_spectrum : powers,
            aperiodic_params_ : aperiodic_params_,
            gaussian_params_ : gaussian_params_
        }
    }

    fn _reset_internal_settings(&mut self) {
        self._internal_settings._gauss_std_limits =
            (self.peak_width_limits.0 / 2.0, self.peak_width_limits.1 / 2.0);
    }

    fn _robust_ap_fit(&mut self, freqs : &Array1<f64>, powers : &Array1<f64>, powers_log : &Array1<f64>)-> Array1<f64> {

        // Initial fit
        let popt = self._simple_ap_fit(freqs, powers, powers_log);
        let inital_fit = lorentzian(&freqs, popt[0], popt[1], popt[2]);

        // Flatten spectrum
        let mut flat_spec : Array1<f64> = powers_log - inital_fit;
        for i in 0..flat_spec.len() {
            if flat_spec[i] < 0.0 {
                flat_spec[i] = 0.0;
            }
        }

        let mut flat_spec_v : Vec<f64> = flat_spec.to_vec();

        // Get percentile threshold
        flat_spec_v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ind : usize = (self._internal_settings._ap_percentile_thresh * flat_spec_v.len() as f64) as usize;
        let perc_thresh : f64 = flat_spec_v[ind];

        // Mask frequency and power arrays
        let perc_mask = flat_spec.mapv(|p| p <= perc_thresh);

        let mask_size : usize = perc_mask.iter().filter(|&x| *x).count();

        let mut freqs_ignore : Array1<f64> = Array1::zeros(mask_size);
        let mut powers_ignore : Array1<f64> = Array1::zeros(mask_size);
        let mut powers_ignore_log : Array1<f64> = Array1::zeros(mask_size);

        let mut j : i64 = 0;
        for i in 0..powers_log.len() {
            if perc_mask[i] {
                freqs_ignore[j as usize] = freqs[i];
                powers_ignore[j as usize] = powers[i];
                powers_ignore_log[j as usize] = powers_log[i];
                j += 1;
            }
        }

        // Fit flattened spectrum
        let aperiodic_params = self._simple_ap_fit(&freqs_ignore, &powers_ignore, &powers_ignore_log);
        aperiodic_params
    }
    fn _simple_ap_fit(&mut self, freqs : &Array1<f64>, powers: &Array1<f64>, powers_log: &Array1<f64>) -> Array1<f64> {

        // todo: added fixed mode
        // if self.aperiodic_mode == "fixed" {
        //
        //     let exp_guess : f64 = (powers[0] - powers[1]).abs() /
        //         (freqs[freqs.len()-1].log10() - freqs[0].log10());
        //     // fit_linear
        // } else {
        // }

        // Offset
        let off_guess : f64 = powers[0];

        // Knee
        let half_max : f64 = powers[0] / 2.0;
        let mut knee_guess : f64 = 0.0;
        for i in 0..powers.len() {
            if powers[i] <= half_max {
                knee_guess = freqs[i];
                break;
            }
        }

        // Exponent
        let exp_guess : f64 = 2.0;

        // Fit
        let init_params : Array1<f64> = array![knee_guess, exp_guess, off_guess];
        let aperiodic_params_ : Array1<f64> = fit_lorentzian(&freqs, &powers_log, &init_params).unwrap();
        aperiodic_params_
    }

    fn _fit_peaks(&mut self, freqs : &Array1<f64>, flat_iter : &Array1<f64>) -> (Array2<f64>, Array1<f64>) {

        // Acutal max peaks is 100 since stackings ndarray's is a pain
        let mut guess : Array2<f64> = Array2::zeros((100, 3));
        let mut guess_len : i64 = 0;
        let mut i_peak : i64 = 0;
        let mut flat_peaks : Array1<f64> = Array1::zeros(flat_iter.len());

        while guess_len < self.max_n_peaks {

            // Argmax of flattened powers
            let _flat_iter : Array1<f64> = flat_iter - &flat_peaks;
            let mut max_val : f64 = f64::MIN;
            let mut max_ind : usize = 0;

            for i in 0.._flat_iter.len() {
                if _flat_iter[i] > max_val {
                    max_val = flat_iter[i];
                    max_ind = i;
                }
            }
            let max_height : f64 = _flat_iter[max_ind];

            // Stoping criteria
            if max_height <= self.peak_threshold * _flat_iter.std(0.0) {
                break;
            }

            // Set the guess parameters
            let guess_freq : f64 = freqs[max_ind];
            let guess_height : f64 = max_height;

            // Halt fitting process if candidate peak drops below minimum height
            if guess_height < self.min_peak_height {
                break;
            }

            // Data-driven first guess at standard deviation
            //   Find half height index on each side of the center frequency
            let half_height : f64 = 0.5 * max_height;

            let mut le_ind : i64 = -1;
            let mut ri_ind : i64 = -1;
            for i in (1..max_ind as usize).rev() {
                if flat_iter[i] <= half_height {
                    le_ind = i as i64;
                    break;
                }
            }
            for i in max_ind+1..flat_iter.len() {
                if flat_iter[i] <= half_height {
                    ri_ind = i as i64;
                    break;
                }
            }

            let le_len : i64 =
                if le_ind != -1 {
                    (le_ind - max_ind as i64).abs()
                } else {
                    i64::MAX
                };

            let ri_len : i64 =
                if ri_ind != -1 {
                    (ri_ind - max_ind as i64).abs()
                } else {
                    i64::MAX
                };

            let mut guess_std : f64 =
                if le_ind == -1 && ri_ind == -1 {
                    (self.peak_width_limits.0 + self.peak_width_limits.1) / 2.0
                } else {
                    let short_side : i64 = le_len.min(ri_len);
                    let fwhm : f64 = short_side as f64 * 2.0 * (freqs[1]-freqs[0]);
                    fwhm / (2.0 * (2.0 * (2.0_f64.ln()))).sqrt()
                };

            if guess_std < self._internal_settings._gauss_std_limits.0 {
                guess_std = self._internal_settings._gauss_std_limits.0;
            } else if guess_std > self._internal_settings._gauss_std_limits.1 {
                guess_std = self._internal_settings._gauss_std_limits.1;
            };

            // Collect guess parameters and subtract this guess gaussian from the data
            let _guess : Array1<f64> = array![guess_freq, guess_height, guess_std];
            for i in 0..3{
                guess[[i_peak as usize, i]] = _guess[i];
            }

            let peak_gauss : Array1<f64> = peak(&freqs, guess_freq, guess_height, guess_std);

            flat_peaks = flat_peaks + peak_gauss;
            i_peak = i_peak + 1;
        }
        (guess.slice(s![..(i_peak) as usize, 0..3]).to_owned(), flat_peaks)
    }
}
