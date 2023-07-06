============
specparam-rs
============

A Rust implementation of the `specparam <https://github.com/fooof-tools/fooof>`_ (fooof).
Procedures are identical to specparam in Python, with the exception that
`argmin <https://argmin-rs.org/>`_ is used for optimization, rather than scipy's
curve_fit. Performance tests show that this implementation is 30-50x faster than
the python implementation, depending on simulation parameters.


Quickstart
----------
In ``main.rs``:

.. code-block:: rust

    use ndarray::{Array1};

    mod gen;
    mod optimization;
    mod specparam;

    use gen::{lorentzian, linear, peak, noise};
    use specparam::SpecParam;

    fn main() {
        // Frequencies
        let freqs : Array1<f64> = Array1::range(1., 101., 1.0);

        // Powers : lorentzian + oscillation + noise
        let mut powers : Array1<f64> = lorentzian(&freqs, 10.0, 2.0, 100.0);
        let sig_noise : Array1<f64> = noise(&freqs, 0.05);

        powers = (powers + sig_noise).mapv(|p| (10.0_f64).powf(p));
        powers = powers + peak(&freqs, 20.0, 20.0, 2.0);

        // Initialize & fit
        let mut sp = SpecParam{
            max_n_peaks: 1,
            aperiodic_mode: "lorentzian".to_string(),
            ..Default::default()
        };

        let results = sp.fit(&freqs, &powers);
    }

And then to excute from a shell:

.. code-block:: bash

    cargo run --release


Dependencies
------------

- ndarray
- ndarray-stats
- blas-src
- openblas-src
- finitediff
- argmin
- argmin-math
- rand
- rand_distr
