use std::sync::mpsc::{channel, Sender};
use threadpool::ThreadPool;

use packed_simd::f64x4;
use rug::{Complex, Float};

use crate::mandelbrot::bounded::{Bound, BoundsChecker, BoundsSettings};
use crate::ui::events::ComputeEvent;

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum ComputeEngine {
    Single,
    Double,
    MPC,
    SimdF64x4,
}

impl ComputeEngine {
    pub fn to_int(self) -> i32 {
        match self {
            Self::Single => 0,
            Self::Double => 1,
            Self::SimdF64x4 => 2,
            Self::MPC => 3,
        }
    }

    pub fn from_int(value: i32) -> Self {
        match value {
            0 => Self::Single,
            1 => Self::Double,
            2 => Self::SimdF64x4,
            3 => Self::MPC,
            _ => Self::Double,
        }
    }
}

pub struct ComputeSettings {
    x: Float,
    y: Float,
    scale: Float,
    width: u32,
    height: u32,
    engine: ComputeEngine,
    bounds: BoundsSettings,
}

impl Clone for ComputeSettings {
    fn clone(&self) -> Self {
        ComputeSettings::new(
            self.x.clone(),
            self.y.clone(),
            self.scale.clone(),
            self.width,
            self.height,
            self.engine,
            self.bounds,
        )
    }
}

impl ComputeSettings {
    pub fn new(
        x: Float,
        y: Float,
        scale: Float,
        width: u32,
        height: u32,
        engine: ComputeEngine,
        bounds: BoundsSettings,
    ) -> ComputeSettings {
        ComputeSettings {
            x,
            y,
            scale,
            width,
            height,
            engine,
            bounds,
        }
    }
}

pub struct ComputedSet {
    width: u32,
    height: u32,
    data: Option<Vec<Bound>>,
}

impl ComputedSet {
    pub fn new(width: u32, height: u32, data: Vec<Bound>) -> ComputedSet {
        ComputedSet {
            width,
            height,
            data: Some(data),
        }
    }

    pub fn empty(width: u32, height: u32) -> ComputedSet {
        ComputedSet {
            width,
            height,
            data: None,
        }
    }

    pub fn get_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn iter(&self) -> Option<std::slice::Iter<Bound>> {
        match &self.data {
            Some(data) => Some(data.iter()),
            None => None,
        }
    }
}

pub struct Compute {}

impl Compute {
    pub fn compute_set(
        thread_pool: Option<&mut ThreadPool>,
        message: Option<Sender<ComputeEvent>>,
        settings: &ComputeSettings,
    ) -> ComputedSet {
        match settings.engine {
            ComputeEngine::Single => {
                Self::compute_set_with_engine::<f32>(thread_pool, message, &settings)
            }
            ComputeEngine::Double => {
                Self::compute_set_with_engine::<f64>(thread_pool, message, &settings)
            }
            ComputeEngine::MPC => {
                Self::compute_set_with_engine::<Complex>(thread_pool, message, &settings)
            }
            ComputeEngine::SimdF64x4 => {
                Self::compute_set_with_engine::<f64x4>(thread_pool, message, &settings)
            }
        }
    }

    fn compute_set_with_engine<T: BoundsChecker + 'static>(
        thread_pool: Option<&mut ThreadPool>,
        message: Option<Sender<ComputeEvent>>,
        settings: &ComputeSettings,
    ) -> ComputedSet {
        let precision = settings.bounds.precision;

        let w = Float::with_val(precision, settings.width);
        let h = Float::with_val(precision, settings.height);
        let ratio = Float::with_val(precision, &w / &h);

        let x_start = Float::with_val(
            precision,
            &settings.x - (Float::with_val(precision, &settings.scale * &ratio) / 2.0),
        );
        let x_step = Float::with_val(precision, &settings.scale * &ratio) / &w;
        let y_start = Float::with_val(
            precision,
            &settings.y - (Float::with_val(precision, &settings.scale / 2.0)),
        );
        let y_step = Float::with_val(precision, &settings.scale / &h);

        if let Some(sender) = &message {
            sender.send(ComputeEvent::Start).unwrap();
        }

        let mut output = vec![Bound::Bounded; settings.width as usize * settings.height as usize];
        match thread_pool {
            None => {
                for y in 0..settings.height {
                    let out = &mut output
                        [(y * settings.width) as usize..((y + 1) * settings.width) as usize];
                    Self::compute_row::<T>(
                        y,
                        [&x_start, &y_start],
                        [&x_step, &y_step],
                        out,
                        settings.clone(),
                    );
                    if let Some(sender) = &message {
                        sender
                            .send(ComputeEvent::Progress((y, settings.height)))
                            .unwrap();
                    }
                }
            }
            Some(thread_pool) => {
                let (tx, rx) = channel();
                for y in 0..settings.height {
                    let tx = tx.clone();
                    let settings = settings.clone();
                    let x_start = x_start.clone();
                    let y_start = y_start.clone();
                    let x_step = x_step.clone();
                    let y_step = y_step.clone();
                    thread_pool.execute(move || {
                        let mut out = vec![Bound::Bounded; settings.width as usize];
                        Self::compute_row::<T>(
                            y,
                            [&x_start, &y_start],
                            [&x_step, &y_step],
                            &mut out,
                            settings,
                        );
                        tx.send((y, out)).unwrap();
                    });
                }
                for n in 0..settings.height {
                    let (y, row) = rx.recv().unwrap();
                    for (input, output) in row
                        .iter()
                        .zip(output.iter_mut().skip((y * settings.width) as usize))
                    {
                        *output = *input;
                    }
                    if let Some(sender) = &message {
                        sender
                            .send(ComputeEvent::Progress((n, settings.height)))
                            .unwrap();
                    }
                }
            }
        }
        if let Some(sender) = &message {
            sender.send(ComputeEvent::End).unwrap();
        }
        ComputedSet::new(settings.width, settings.height, output)
    }

    fn compute_row<T: BoundsChecker + 'static>(
        y: u32,
        start: [&Float; 2],
        step: [&Float; 2],
        out: &mut [Bound],
        settings: ComputeSettings,
    ) {
        let step_by = T::mask().len();
        let precision = settings.bounds.precision;
        let yy = Float::with_val(
            precision,
            start[1] + Float::with_val(precision, step[1] * y),
        );
        for x in (0..settings.width).step_by(step_by) {
            let mut xx: Vec<Float> = Vec::with_capacity(step_by);
            for i in 0..step_by {
                xx.push(start[0] + step[0] * Float::with_val(precision, x + i as u32))
            }
            let yy = vec![Float::with_val(precision, &yy); step_by];

            let out = &mut out[x as usize..x as usize + step_by];
            T::check_bounded(&xx, &yy, settings.bounds, out);
        }
    }
}
