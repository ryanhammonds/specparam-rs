
// extern crate blas_src;
// // use argmin::core::observers::{ObserverMode, SlogLogger};
// use argmin::core::observers::{ObserverMode, SlogLogger};
// use argmin::core::{CostFunction, Error, Executor, Gradient, Hessian};
// use argmin::solver::linesearch::{MoreThuenteLineSearch, BacktrackingLineSearch, condition::ArmijoCondition};
// use argmin::solver::quasinewton::LBFGS;
// use argmin::solver::quasinewton::SR1TrustRegion;
// use argmin::solver::trustregion::{CauchyPoint, Dogleg, Steihaug, TrustRegion};
// use ndarray::{array, Array1, Array2};
// use finitediff::FiniteDiff;
// use argmin::solver::newton::NewtonCG;
// use argmin::solver::gradientdescent::SteepestDescent;



//pub use self::specparam::SpecParam;
mod specparam;
use specparam::SpecParam;

fn main() {
    //let sp = SpecParam{} ;//{Default::default()};
    let sp = SpecParam{..Default::default()};
}





// // Lorentzian (log10) function
// pub fn lorentzian( f: &Array1<f64>, fk: f64, x: f64, b: f64) -> Array1<f64> {
//     let mut p : Array1<f64> = Array1::from_vec(vec![1.0; f.len()]);
//     let base : f64 = (x * fk.log10()) + b.log10();
//     let fkx : f64 = fk.powf(x);
//     for i in 0..f.len() {
//         p[i] = base - (fkx + f[i].powf(x)).log10();
//     }
//    p
// }

// // Lorentzian (log10) loss function
// pub fn lorentzian_loss(freqs: &Array1<f64>, powers: &Array1<f64>, fk: f64, x: f64, b: f64) -> f64 {
//     let mut y_diff : f64 = 0.0;
//     let y_pred = lorentzian(&freqs, fk, x, b);
//     for i in 0..freqs.len() {
//         y_diff = y_diff + (y_pred[i] - powers[i]).powf(2.0);
//     }
//     y_diff / freqs.len() as f64
// }

// // Lorentzian optimizer
// struct Lorentzian {
//     freqs: Array1<f64>,
//     y_true: Array1<f64>
// }

// impl CostFunction for Lorentzian {
//     type Param = Array1<f64>;
//     type Output = f64;
//     fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
//         Ok(lorentzian_loss(&self.freqs, &self.y_true, param[0], param[1], param[2]))
//     }
// }

// impl Gradient for Lorentzian {
//     type Param =Array1<f64>;
//     type Gradient = Array1<f64>;

//     fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
//         Ok((*param).forward_diff(&|p| lorentzian_loss(&self.freqs, &self.y_true, p[0], p[1], p[2])))
//     }
// }

// impl Hessian for Lorentzian {
//     type Param = Array1<f64>;
//     type Hessian = Array2<f64>;

//     fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
//         Ok((*param).forward_hessian(&|p| self.gradient(p).unwrap()))
//     }
// }


// fn run() -> Result<(), Error> {
//     // Create frequencies
//     let mut freqs : Array1<f64> = Array1::from_vec(vec![1.0; 100]);

//     for i in 1..101 {
//         freqs[i-1] = i as f64;
//     }

//     // Define powers as lorentzian function
//     let y_true = lorentzian(&freqs, 10.0, 2.0, 100.0);

//     // Initalize cost and params
//     let cost = Lorentzian { freqs: freqs, y_true: y_true };

//     let init_param : Array1<f64> = array![10.0, 1.0, 100.0];

//     // set up a line search
//     let linesearch = MoreThuenteLineSearch::new();//.with_c(1e-4, 0.9)?;

//     // Set up solver
//     let solver = LBFGS::new(linesearch, 7)
//         .with_tolerance_grad(1e-4)?
//         .with_tolerance_cost(1e-6)?;

//     // Run solver
//     let res = Executor::new(cost, solver)
//         .configure(|state| state.param(init_param).max_iters(100))
//         .add_observer(SlogLogger::term(), ObserverMode::Always)
//         .run()?;

//     // Wait a second (lets the logger flush everything before printing again)
//     std::thread::sleep(std::time::Duration::from_secs(1));

//     // Print result
//     println!("{res}");
//     Ok(())
// }

// fn main() {
//     if let Err(ref e) = run() {
//         println!("{e}");
//         std::process::exit(1);
//     }
// }