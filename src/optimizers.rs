// Parameter optimization
use argmin::core::State;
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::{condition::ArmijoCondition, BacktrackingLineSearch};
use argmin::solver::quasinewton::LBFGS;
use finitediff::FiniteDiff;
use ndarray::Array1;
use crate::gen::{linear_bounded, lorentzian_bounded, peak_bounded, double_lorentzian};

// Loss functions
fn mse(y_pred: &Array1<f64>, y_true: &Array1<f64>) -> f64{
    // Mean-squared error
    (y_pred - y_true)
        .map(|p| p.powi(2))
        .mean()
        .unwrap()
}

fn huber(y_pred: &Array1<f64>, y_true: &Array1<f64>, delta: f64) -> f64{
    // Psuedo-Huber loss
    let delta_pow2: f64 = delta.powi(2);
        (y_pred-y_true)
            .map(|p| delta_pow2 * ((1.0 + (p / delta).powi(2)).sqrt() - 1.0))
            .mean()
            .unwrap()
}

// Lorentzian fitting
struct Lorentzian {
    freqs: Array1<f64>,
    powers_true: Array1<f64>,
    lower: Array1<f64>,
    upper: Array1<f64>,
    delta: f64,
}

pub fn fit_lorentzian(
    freqs: &Array1<f64>,
    powers: &Array1<f64>,
    init_param: &Array1<f64>,
    ctol: f64,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    delta: f64,
) -> Result<Array1<f64>, Error> {
    let cost = Lorentzian {
        freqs: freqs.clone(),
        powers_true: powers.clone(),
        lower: lower.clone(),
        upper: upper.clone(),
        delta: delta,
    };
    let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001)?).rho(0.9)?;

    let solver = LBFGS::new(linesearch, 20)
        .with_tolerance_grad(1e-6)?
        .with_tolerance_cost(ctol)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param.clone()).max_iters(100))
        .run()?;

    let param = res.state.get_best_param().unwrap();
    Ok(param.clone())
}

pub fn lorentzian_loss(
    freqs: &Array1<f64>,
    powers: &Array1<f64>,
    fk: f64,
    x: f64,
    b: f64,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    delta: f64,
) -> f64 {
    let powers_pred: Array1<f64> = lorentzian_bounded(&freqs, fk, x, b, &lower, &upper);
    if delta <= 0.0 {
        // MSE loss
        mse(&powers_pred, &powers)
    } else {
        // Huber loss
        huber(&powers_pred, &powers, delta)
    }
}

impl CostFunction for Lorentzian {
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(lorentzian_loss(
            &self.freqs,
            &self.powers_true,
            param[0],
            param[1],
            param[2],
            &self.lower,
            &self.upper,
            self.delta,
        ))
    }
}

impl Gradient for Lorentzian {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok((*param).forward_diff(&|p| {
            lorentzian_loss(
                &self.freqs,
                &self.powers_true,
                p[0],
                p[1],
                p[2],
                &self.lower,
                &self.upper,
                self.delta,
            )
        }))
    }
}

// Double Lorentzian
struct DoubleLorentzian {
    freqs: Array1<f64>,
    powers_true: Array1<f64>,
    delta: f64,
}

pub fn fit_double_lorentzian(
    freqs: &Array1<f64>,
    powers: &Array1<f64>,
    init_param: &Array1<f64>,
    ctol: f64,
    delta: f64
) -> Result<Array1<f64>, Error> {
    let cost = DoubleLorentzian {
        freqs: freqs.clone(),
        powers_true: powers.clone(),
        delta: delta,
    };

    let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001)?).rho(0.9)?;

    let solver = LBFGS::new(linesearch, 20)
        .with_tolerance_grad(1e-6)?
        .with_tolerance_cost(ctol)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param.clone()).max_iters(100))
        .run()?;

    let param = res.state.get_best_param().unwrap();
    Ok(param.clone())
}

pub fn double_lorentzian_loss(
    freqs: &Array1<f64>,
    powers: &Array1<f64>,
    fk0: f64,
    x0: f64,
    b0: f64,
    fk1: f64,
    x1: f64,
    b1: f64,
    delta: f64
) -> f64 {
    let powers_pred : Array1<f64> = double_lorentzian(&freqs, fk0, x0, b0, fk1, x1, b1);
    if delta <= 0.0 {
        // MSE loss
        mse(&powers_pred, &powers)
    } else {
        // Huber loss
        huber(&powers_pred, &powers, delta)
    }
}


impl CostFunction for DoubleLorentzian {
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(double_lorentzian_loss(
            &self.freqs,
            &self.powers_true,
            param[0], param[1], param[2],
            param[3], param[4], param[5],
            self.delta,
        ))
    }
}

impl Gradient for DoubleLorentzian {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok((*param).forward_diff(&|p| {
            double_lorentzian_loss(
                &self.freqs,
                &self.powers_true,
                p[0], p[1], p[2],
                p[3], p[4], p[5],
                self.delta,
            )
        }))
    }
}

// Linear fitting
struct Linear {
    freqs: Array1<f64>,
    powers_true: Array1<f64>,
    lower: Array1<f64>,
    upper: Array1<f64>,
    delta: f64,
}

pub fn fit_linear(
    freqs: &Array1<f64>,
    powers: &Array1<f64>,
    init_param: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    delta: f64,
) -> Result<Array1<f64>, Error> {
    let cost = Linear {
        freqs: freqs.clone(),
        powers_true: powers.clone(),
        lower: lower.clone(),
        upper: upper.clone(),
        delta: delta,
    };
    let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001)?).rho(0.9)?;

    let solver = LBFGS::new(linesearch, 20)
        .with_tolerance_grad(1e-6)?
        .with_tolerance_cost(1e-9)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param.clone()).max_iters(100))
        .run()?;

    let param = res.state.get_best_param().unwrap();
    Ok(param.clone())
}

pub fn linear_loss(
    freqs: &Array1<f64>,
    powers: &Array1<f64>,
    x: f64,
    b: f64,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    delta: f64,
) -> f64 {
    let powers_pred: Array1<f64> = linear_bounded(&freqs, x, b, lower, upper);
    if delta <= 0.0 {
        // MSE loss
        mse(&powers_pred, &powers)
    } else {
        // Huber loss
        huber(&powers_pred, &powers, delta)
    }
}

impl CostFunction for Linear {
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(linear_loss(
            &self.freqs,
            &self.powers_true,
            param[0],
            param[1],
            &self.lower,
            &self.upper,
            self.delta,
        ))
    }
}

impl Gradient for Linear {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok((*param).forward_diff(&|p| {
            linear_loss(
                &self.freqs,
                &self.powers_true,
                p[0],
                p[1],
                &self.lower,
                &self.upper,
                self.delta,
            )
        }))
    }
}

// Gaussian fitting
pub fn fit_gaussian(
    freqs: &Array1<f64>,
    powers: &Array1<f64>,
    init_param: Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
) -> Result<Array1<f64>, Error> {
    let cost = Gaussian {
        freqs: freqs.clone(),
        powers_true: powers.clone(),
        lower: lower.clone(),
        upper: upper.clone(),
    };
    let linesearch = BacktrackingLineSearch::new(ArmijoCondition::new(0.0001)?).rho(0.9)?;
    let solver = LBFGS::new(linesearch, 20)
        .with_tolerance_grad(1e-9)?
        .with_tolerance_cost(1e-9)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(100))
        .run()?;

    let param = res.state.get_best_param().unwrap();
    Ok(param.clone())
}

struct Gaussian {
    freqs: Array1<f64>,
    powers_true: Array1<f64>,
    lower: Array1<f64>,
    upper: Array1<f64>,
}

pub fn gaussian_loss(
    freqs: &Array1<f64>,
    powers: &Array1<f64>,
    param: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
) -> f64 {
    let n_gaussian: i64 = param.len() as i64 / 3;

    let (mut y_pred, out_of_bounds): (Array1<f64>, bool) = peak_bounded(
        &freqs,
        param[0],
        param[1],
        param[2],
        &vec![lower[0], lower[1], lower[2]],
        &vec![upper[0], upper[1], upper[2]],
    );

    if out_of_bounds {
        return 0.0;
    }

    for i in 1..n_gaussian {
        let j: usize = (i * 3) as usize;
        let (_y, out_of_bounds) = peak_bounded(
            &freqs,
            param[j],
            param[j + 1],
            param[j + 2],
            &vec![lower[j], lower[j + 1], lower[j + 2]],
            &vec![upper[j], upper[j + 1], upper[j + 2]],
        );
        y_pred = y_pred + _y;
        if out_of_bounds {
            return 0.0;
        }
    }
    // MSE loss
    mse(&y_pred, &powers)
}

impl CostFunction for Gaussian {
    type Param = Array1<f64>;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(gaussian_loss(
            &self.freqs,
            &self.powers_true,
            param,
            &self.lower,
            &self.upper,
        ))
    }
}

impl Gradient for Gaussian {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok((*param).forward_diff(&|p| {
            gaussian_loss(&self.freqs, &self.powers_true, p, &self.lower, &self.upper)
        }))
    }
}
