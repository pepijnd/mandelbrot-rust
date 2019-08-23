use packed_simd::{f64x4, u64x4};
use rug::{Assign, Complex, Float};

#[derive(Debug, Copy, Clone)]
pub enum Bound {
    Bounded,
    Unbounded(u64),
}

pub trait BoundsChecker: Send {
    fn check_bounded(x: &[Float], y: &[Float], settings: BoundsSettings, out: &mut [Bound]);
    fn mask() -> Vec<usize>;
}

#[derive(Copy, Clone)]
pub struct BoundsSettings {
    pub limit: u64,
    pub precision: u32,
}

impl BoundsSettings {
    pub fn new(limit: u64, precision: u32) -> BoundsSettings {
        BoundsSettings { limit, precision }
    }
}

macro_rules! float_to {
    ($e:expr, f32) => {
        $e.to_f32()
    };
    ($e:expr, f64) => {
        $e.to_f64()
    };
}

macro_rules! impl_boundscheck_primitive {
    ($type:tt) => {
        impl BoundsChecker for $type {
            fn check_bounded(
                x: &[Float],
                y: &[Float],
                settings: BoundsSettings,
                out: &mut [Bound],
            ) {
                let x = float_to!(&x[0], $type);
                let y = float_to!(&y[0], $type);
                let c = (x, y);
                let mut z = (0.0, 0.0);
                let mut iter = 0;
                while iter < settings.limit {
                    z = (z.0 * z.0 - z.1 * z.1 + c.0, 2.0 * z.0 * z.1 + c.1);
                    if z.0 * z.0 + z.1 * z.1 < 4.0 {
                        iter += 1;
                    } else {
                        out[0] = Bound::Unbounded(iter);
                        return;
                    }
                }
                out[0] = Bound::Bounded;
            }

            fn mask() -> Vec<usize> {
                vec![0]
            }
        }
    };
}

impl_boundscheck_primitive!(f64);
impl_boundscheck_primitive!(f32);

impl BoundsChecker for Complex {
    fn check_bounded(x: &[Float], y: &[Float], settings: BoundsSettings, out: &mut [Bound]) {
        let mut buffer = Complex::new(settings.precision);
        let c = Complex::with_val(settings.precision, (&x[0], &y[0]));
        let mut z = Complex::with_val(settings.precision, (0.0, 0.0));
        let mut iter = 0;
        while iter < settings.limit {
            let z_temp = Complex::with_val(settings.precision, z.square_ref());
            z.assign(z_temp + &c);
            buffer.assign(z.norm_ref());
            if buffer.real() < &4 {
                iter += 1;
            } else {
                out[0] = Bound::Unbounded(iter);
                return;
            }
        }
        out[0] = Bound::Bounded;
    }

    fn mask() -> Vec<usize> {
        vec![0]
    }
}

impl BoundsChecker for f64x4 {
    fn check_bounded(x: &[Float], y: &[Float], settings: BoundsSettings, out: &mut [Bound]) {
        let x = f64x4::from_slice_unaligned(
            x.iter().map(|x| x.to_f64()).collect::<Vec<_>>().as_slice(),
        );
        let y = f64x4::from_slice_unaligned(
            y.iter().map(|y| y.to_f64()).collect::<Vec<_>>().as_slice(),
        );
        let c = (x, y);
        let mut z = (f64x4::splat(0.0), f64x4::splat(0.0));
        let mut iter = u64x4::splat(0);

        let check = f64x4::splat(4.0);
        for _ in 0..settings.limit {
            z = (
                z.0 * z.0 - z.1 * z.1 + c.0,
                f64x4::splat(2.0) * z.0 * z.1 + c.1,
            );
            let mask = (z.0 * z.0 + z.1 * z.1).lt(check);
            if mask.none() {
                break;
            }
            iter = mask.select(iter + u64x4::splat(1), iter);
        }
        let mut checks = vec![0; 4];
        iter.write_to_slice_unaligned(&mut checks);
        for (o, n) in out.iter_mut().zip(checks.iter()) {
            *o = if *n < settings.limit {
                Bound::Unbounded(*n)
            } else {
                Bound::Bounded
            }
        }
    }

    fn mask() -> Vec<usize> {
        vec![0, 1, 2, 3]
    }
}
