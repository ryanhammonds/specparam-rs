// Parameter optimization
extern crate blas_src;
use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use argmin::core::State;
use ndarray::{Array1, Array2, Axis, s};
use finitediff::FiniteDiff;

use crate::gen::{lorentzian, linear, peak};


// Lorentzian fitting
pub fn fit_lorentzian(freqs : &Array1<f64>, powers : &Array1<f64>, init_param : &Array1<f64>) -> Result<Array1<f64>, Error> {

    let cost = Lorentzian{freqs: freqs.clone(), powers_true: powers.clone()};
    let linesearch = MoreThuenteLineSearch::new();//.with_c(1e-4, 0.9)?;

    let solver = LBFGS::new(linesearch, 7)
        .with_tolerance_grad(1e-6)?
        .with_tolerance_cost(1e-6)?;

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
    let y_pred = lorentzian(&freqs, fk, x, b);
    (y_pred - powers).mapv(|p| p.powf(2.0)).sum() / freqs.len() as f64
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
    let linesearch = MoreThuenteLineSearch::new();//.with_c(1e-4, 0.9)?;

    let solver = LBFGS::new(linesearch, 7)
        .with_tolerance_grad(1e-4)?
        .with_tolerance_cost(1e-6)?;

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
    let y_pred = linear(&freqs, x, b);
    (y_pred - powers).mapv(|p| p.powf(2.0)).sum() / freqs.len() as f64
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
        Ok((*param).forward_diff(&|p| linear_loss(&self.freqs, &self.powers_true, p[0], p[1])))
    }
}

// Gaussian fitting
pub fn fit_gaussian(freqs : &Array1<f64>, powers : &Array1<f64>, init_param : &Array1<f64>) -> Result<Array1<f64>, Error> {

    let cost = Gaussian{freqs: freqs.clone(), powers_true: powers.clone()};
    let linesearch = MoreThuenteLineSearch::new();//.with_c(1e-4, 0.9)?;

    let solver = LBFGS::new(linesearch, 7)
        .with_tolerance_grad(1e-4)?
        .with_tolerance_cost(1e-6)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param.clone()).max_iters(100))
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
    param: Array1<f64>,
) -> f64 {
    let n_gaussian : i64 = param.len() as i64 / 3;
    let mut param2 = Array2::from_shape_vec((n_gaussian as usize, 3), param.to_vec()).unwrap();
    let mut y_pred : Array1<f64> = Array1::zeros(freqs.len());
    for i in 0..n_gaussian{
        let ctr : f64 = param2[[i as usize, 0]];
        let hgt : f64 = param2[[i as usize, 1]];
        let wid : f64 = param2[[i as usize, 2]];
        y_pred = y_pred + peak(&freqs, ctr, hgt, wid);
    }
    let loss = (y_pred - powers).mapv(|p| p.powf(2.0)).sum() / freqs.len() as f64;
    loss
}

impl CostFunction for Gaussian {
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let loss = gaussian_loss(&self.freqs, &self.powers_true, param.clone());
        Ok(loss)
    }
}

impl Gradient for Gaussian {
    type Param =Array1<f64>;
    type Gradient = Array1<f64>;
    fn gradient(&self, mut param: &Self::Param) -> Result<Self::Gradient, Error> {
        let grad = (*param).forward_diff(&|p| gaussian_loss(&self.freqs, &self.powers_true, p.clone()));
        Ok(grad)
    }
}
