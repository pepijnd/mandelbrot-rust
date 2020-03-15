#![feature(test)]

extern crate test;

#[macro_use]
extern crate imgui;
#[macro_use]
extern crate glium;
extern crate imgui_glium_renderer;
extern crate imgui_winit_support;
extern crate packed_simd;
extern crate palette;
extern crate rug;
extern crate threadpool;
extern crate time;

mod mandelbrot;
mod ui;

use ui::app::{App, AppSettings};

fn main() {
    let mut args = std::env::args();
    match args.nth(1).unwrap_or_else(|| String::from("")).as_str() {
        "perf_test" => {
            use rug::Float;

            use mandelbrot::{
                bounded::BoundsSettings,
                compute::{Compute, ComputeEngine, ComputeSettings},
            };

            let start = std::time::Instant::now();

            let size = (1600 / 2, 900 / 2);

            let precision = 53;
            let settings = ComputeSettings::new(
                Float::with_val(precision, -0.5),
                Float::with_val(precision, 0.0),
                Float::with_val(precision, 1.75),
                size.0,
                size.1,
                ComputeEngine::Precision,
                BoundsSettings::new(250, precision),
            );

            Compute::compute_set(None, None, &settings);

            let duration = std::time::Instant::now() - start;
            println!("{}", duration.as_secs_f64());
        }
        _ => {
            let app = App::new(AppSettings::new());
            app.run();
        }
    }
}
