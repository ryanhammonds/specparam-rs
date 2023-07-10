// Parameter optimization
extern crate blas_src;
use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::{condition::ArmijoCondition, BacktrackingLineSearch};
use argmin::solver::quasinewton::LBFGS;
use argmin::core::State;
use ndarray::{Array1, Array2};
use finitediff::FiniteDiff;

use crate::gen::{lorentzian, linear, peak};


// Lorentzian fitting
pub fn fit_lorentzian(
        freqs : &Array1<f64>,
        powers : &Array1<f64>,
        init_param : &Array1<f64>,
        ctol : f64
    ) -> Result<Array1<f64>, Error> {

    let cost = Lorentzian{freqs: freqs.clone(), powers_true: powers.clone()};
    let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001)?).rho(0.9)?;

    let solver = LBFGS::new(linesearch, 10)
        .with_tolerance_grad(1e-3)?
        .with_tolerance_cost(ctol)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param.clone()).max_iters(100))
        //.add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;

    let param = res.state.get_best_param().unwrap();
    Ok(param.clone())
}

struct Lorentzian {
    freqs: Array1<f64>,
    powers_true: Array1<f64>
}

pub fn lorentzian_loss(freqs: &Array1<f64>, powers: &Array1<f64>, fk: f64, x: f64, b: f64) -> f64 {
    //(lorentzian(&freqs, fk, x, b) - powers).map(|p| p.powi(2)).mean().unwrap()
    // Pseudo-Huber loss
    let delta : f64 = 1.0;
    let delta_pow2 : f64 = delta.powi(2);
    (lorentzian(&freqs, fk, x, b) - powers).map(|p| delta_pow2 * ((1.0 + (p/delta).powi(2)).sqrt() - 1.0)).mean().unwrap()
}

impl CostFunction for Lorentzian {
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(lorentzian_loss(&self.freqs, &self.powers_true, param[0], param[1], param[2]))
    }
}

impl Gradient for Lorentzian {
    type Param =Array1<f64>;
    type Gradient = Array1<f64>;
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok((*param).forward_diff(&|p| lorentzian_loss(&self.freqs, &self.powers_true, p[0], p[1], p[2])))
    }
}

// Linear fitting
pub fn fit_linear(freqs : &Array1<f64>, powers : &Array1<f64>, init_param : &Array1<f64>) -> Result<Array1<f64>, Error> {

    let cost = Linear{freqs: freqs.clone(), powers_true: powers.clone()};
    let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001)?).rho(0.9)?;

    let solver = LBFGS::new(linesearch, 5)
        .with_tolerance_grad(1e-3)?
        .with_tolerance_cost(1e-9)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param.clone()).max_iters(100))
        //.add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;

    let param = res.state.get_best_param().unwrap();
    Ok(param.clone())
}

struct Linear {
    freqs: Array1<f64>,
    powers_true: Array1<f64>
}

pub fn linear_loss(freqs: &Array1<f64>, powers: &Array1<f64>, x: f64, b: f64) -> f64 {
    //(linear(&freqs, x, b) - powers).map(|p| p.powi(2)).mean().unwrap()
    // Pseudo-Huber loss
    let delta : f64 = 1.0;
    let delta_pow2 : f64 = delta.powi(2);
    (linear(&freqs, x, b) - powers).map(|p| delta_pow2 * ((1.0 + (p/delta).powi(2)).sqrt() - 1.0)).mean().unwrap()
}

impl CostFunction for Linear {
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(linear_loss(&self.freqs, &self.powers_true, param[0], param[1]))
    }
}

impl Gradient for Linear {
    type Param =Array1<f64>;
    type Gradient = Array1<f64>;
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok((*param).central_diff(&|p| linear_loss(&self.freqs, &self.powers_true, p[0], p[1])))
    }
}

// Gaussian fitting
pub fn fit_gaussian(freqs : &Array1<f64>, powers : &Array1<f64>, init_param : Array1<f64>) -> Result<Array1<f64>, Error> {

    let cost = Gaussian{freqs: freqs.clone(), powers_true: powers.clone()};
    let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001)?).rho(0.9)?;
    let solver = LBFGS::new(linesearch, 10)
        .with_tolerance_grad(1e-3)?
        .with_tolerance_cost(1e-9)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(100))
        //.add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;


    let param = res.state.get_best_param().unwrap();
    Ok(param.clone())
}

struct Gaussian {
    freqs: Array1<f64>,
    powers_true: Array1<f64>
}

pub fn gaussian_loss(
    freqs: &Array1<f64>,
    powers: &Array1<f64>,
    param: &Array1<f64>,
) -> f64 {
    let n_gaussian : i64 = param.len() as i64 / 3;
    let mut y_pred : Array1<f64> = peak(&freqs, param[0], param[1], param[2]);
    for i in 1..n_gaussian{
        let j : usize = (i * 3) as usize;
        y_pred = y_pred + peak(&freqs, param[j], param[j+1], param[j+2]);
    }
    (y_pred - powers).map(|p| p.powi(2)).mean().unwrap()
}

impl CostFunction for Gaussian {
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(gaussian_loss(&self.freqs, &self.powers_true, param))
    }
}

impl Gradient for Gaussian {
    type Param =Array1<f64>;
    type Gradient = Array1<f64>;
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok((*param).forward_diff(&|p| gaussian_loss(&self.freqs, &self.powers_true, p)))
    }
}
