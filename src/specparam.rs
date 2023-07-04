
pub struct SpecParam {
    pub peak_width_limits : Vec<f64>,
    pub max_n_peaks : i64,
    pub min_peak_height : f64,
    pub peak_threshold : f64,
    pub aperiodic_mode : String,
    pub verbose : bool
}

impl Default for SpecParam {
    fn default() -> SpecParam {
        SpecParam {
            peak_width_limits : vec![0.0, 12.0],
            max_n_peaks : i64::MAX,
            min_peak_height : 0.0,
            peak_threshold : 2.0,
            aperiodic_mode : String::from("fixed"),
            verbose : true
        }
    }
}
