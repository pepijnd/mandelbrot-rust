use packed_simd::{f32x8, f64x4, u32x8, u64x4};
use rug::{Assign, Complex, Float};

#[derive(Debug, Copy, Clone)]
pub enum Bound {
    Bounded,
    Unbounded(u64),
}

pub trait BoundsChecker<F>: Send {
    fn check_bounded(x: &[F], y: &[F], settings: &BoundsSettings, out: &mut [Bound]);
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

macro_rules! impl_boundscheck_primitive {
    ($type:tt) => {
        impl BoundsChecker<f64> for $type {
            fn check_bounded(x: &[f64], y: &[f64], settings: &BoundsSettings, out: &mut [Bound]) {
                let x = x[0];
                let y = y[0];
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

impl BoundsChecker<Float> for Complex {
    fn check_bounded(x: &[Float], y: &[Float], settings: &BoundsSettings, out: &mut [Bound]) {
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

impl BoundsChecker<f64> for f32x8 {
    fn check_bounded(x: &[f64], y: &[f64], settings: &BoundsSettings, out: &mut [Bound]) {
        let mut t = [0f32; 8];
        t.iter_mut()
            .zip(x.iter())
            .map(|(t, s)| *t = *s as f32)
            .for_each(drop);
        let x = f32x8::from_slice_aligned(&t);
        t.iter_mut()
            .zip(y.iter())
            .map(|(t, s)| *t = *s as f32)
            .for_each(drop);
        let y = f32x8::from_slice_aligned(&t);
        let c = (x, y);
        let mut z = (f32x8::splat(0.0), f32x8::splat(0.0));
        let mut iter = u32x8::splat(0);

        let check = f32x8::splat(4.0);
        for _ in 0..settings.limit {
            z = (
                z.0 * z.0 - z.1 * z.1 + c.0,
                f32x8::splat(2.0) * z.0 * z.1 + c.1,
            );
            let mask = (z.0 * z.0 + z.1 * z.1).lt(check);
            if mask.none() {
                break;
            }
            iter = mask.select(iter + u32x8::splat(1), iter);
        }
        let mut checks = [0; 8];
        iter.write_to_slice_aligned(&mut checks);
        out.iter_mut()
            .zip(checks.iter())
            .map(|(o, n)| {
                *o = if *n < settings.limit as u32 {
                    Bound::Unbounded(*n as u64)
                } else {
                    Bound::Bounded
                }
            })
            .for_each(drop);
    }

    fn mask() -> Vec<usize> {
        vec![0, 1, 2, 3, 4, 5, 6, 7]
    }
}

impl BoundsChecker<f64> for f64x4 {
    fn check_bounded(x: &[f64], y: &[f64], settings: &BoundsSettings, out: &mut [Bound]) {
        let mut t = [0f64; 4];
        t.iter_mut()
            .zip(x.iter())
            .map(|(t, s)| *t = *s)
            .for_each(drop);
        let x = f64x4::from_slice_aligned(&t);
        t.iter_mut()
            .zip(y.iter())
            .map(|(t, s)| *t = *s)
            .for_each(drop);
        let y = f64x4::from_slice_aligned(&t);
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
        let mut checks = [0; 4];
        iter.write_to_slice_aligned(&mut checks);
        out.iter_mut()
            .zip(checks.iter())
            .map(|(o, n)| {
                *o = if *n < settings.limit {
                    Bound::Unbounded(*n)
                } else {
                    Bound::Bounded
                }
            })
            .for_each(drop);
    }

    fn mask() -> Vec<usize> {
        vec![0, 1, 2, 3]
    }
}
