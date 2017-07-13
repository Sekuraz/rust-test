#![cfg_attr(test, feature(test))]
#![feature(step_by)]

#![feature(associated_consts)]
#![feature(drop_types_in_const)]
#![feature(const_fn)]

#![feature(inclusive_range_syntax)]

#[macro_use]
extern crate lazy_static;
extern crate hwloc;
extern crate cpuprofiler;
extern crate rayon;

use cpuprofiler::PROFILER;

use hwloc::Topology;

#[allow(unused)]
mod triades;

#[allow(unused)]
mod matmul;

use triades::*;
use matmul::*;


lazy_static ! {
    static ref TOPOLOGY: Topology = Topology::new();
}

extern crate rand;
#[allow(unused)]
fn random_array() -> Vec<NumType>
{
    let array_length = match std::env::var("ARRAY_SIZE") {
        Ok(len) => len.parse::<usize>().expect("ARRAY_SIZE env variable must be the array length"),
        Err(_)  => 4000000
    };
    (0..array_length).map(|_| rand::random::<NumType>()).collect()
}

#[allow(unused)]
fn main() {

    let n = 128 * 8;

    let a = Matrix::random(n, n);
    let b = Matrix::random(n, n);
    let mut c = Matrix::zero(n, n);

    rayon::initialize(rayon::Configuration::new());

    PROFILER.lock().unwrap().start("/tmp/profile");
    for _ in 0..10 {
        matmul::naive::mult(&a, &b, &mut c);

        matmul::naive_unchecked::mult(&a, &b, &mut c);
        matmul::naive_reordered::mult(&a, &b, &mut c);
        matmul::naive_simd::mult(&a, &b, &mut c);
        matmul::naive_rayon::mult(&a, &b, &mut c);

    }
    PROFILER.lock().unwrap().stop();

    println!("{}", c[0]);

    //let mut res = vec![0 as NumType; ARRAY_SIZE];
    //let a = random_array();
    //let b = random_array();
    //let c = random_array();

    /*
    copy(&a, &mut res);
    add(&mut res, &a, &b);
    striad(&mut res, &a, &b, S);
    vtriad(&mut res, &a, &b, &c);
    */
    /*
    vtriad_itertools(&mut res, &a, &b, &c);
    for i in 0..ARRAY_SIZE {
        println!("{}: {} * {} + {} = {}", i, a[i], c[i], b[i], res[i]);
    }
    */
    //vtriad_rayon(&mut res, &a, &b, &c);

    //vtriad_rayon(&mut res, &a, &b, &c);
    //vtriad_threads(&mut res, &a, &b, &c);
    //vtriad_itertools(&mut res, &a, &b, &c);
    //vtriad_simd(&mut res, &a, &b, &c);
    //for i in 0..ARRAY_SIZE {
     //   println!("{}: {} * {} + {} = {}", i, a[i], c[i], b[i], res[i]);
    //}
}
