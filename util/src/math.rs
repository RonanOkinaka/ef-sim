//! This file has a very limited set of math utility features.
//! This is intentional, as this project needs very little mathematical
//! functionality to operate; less code has fewer bugs!

use bytemuck::{Pod, Zeroable};
use std::ops::{Add, Mul, Sub};

pub const RELAXED_EPSILON: f32 = 0.001;

pub fn almost_zero(value: f32) -> bool {
    value.abs() < RELAXED_EPSILON
}

/// Represents one point in RxR.
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Point(pub f32, pub f32);

impl Add for Point {
    type Output = Point;

    fn add(self, rhs: Point) -> Self::Output {
        Point(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Sub for Point {
    type Output = Point;

    fn sub(self, rhs: Point) -> Self::Output {
        Point(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl Mul<f32> for Point {
    type Output = Point;

    fn mul(self, rhs: f32) -> Self::Output {
        Point(self.0 * rhs, self.1 * rhs)
    }
}

impl Point {
    /// Returns the dot product with another vector.
    pub fn dot(self, rhs: Self) -> f32 {
        (self.0 * rhs.0) + (self.1 * rhs.1)
    }

    /// Returns the normalized vector.
    pub fn normalize(self) -> Self {
        let (a, b) = {
            let a = self.0.abs();
            let b = self.1.abs();

            if a < b {
                (a, b)
            } else {
                (b, a)
            }
        };

        let recip_b = 1.0 / b;
        let frac_a = a * recip_b;
        let inv_magnitude = recip_b * (frac_a * frac_a + 1.0).powf(-0.5);

        Self(self.0, self.1) * inv_magnitude
    }

    /// Returns the vector rotated 90 degrees.
    pub fn orthogonalize(self) -> Self {
        Self(-self.1, self.0)
    }

    /// Returns the vector rotated -90 degrees.
    pub fn reverse_orthogonalize(self) -> Self {
        Self(self.1, -self.0)
    }

    /// Returns whether this vector is very close to the origin.
    pub fn almost_zero(self) -> bool {
        almost_zero(self.0) && almost_zero(self.1)
    }
}

#[cfg(test)]
mod test_math {
    use super::*;

    #[test]
    fn test_almost_zero() -> Result<(), &'static str> {
        let p = Point(0.0, 0.00000001);

        if !p.almost_zero() {
            return Err("almost_zero() should accept very small values");
        }

        if almost_zero(5.0) {
            return Err("almost_zero() should not accept large positive values");
        }

        if almost_zero(-2.0) {
            return Err("almost_zero() should not accept large negative values");
        }

        Ok(())
    }

    #[test]
    fn test_dot() -> Result<(), &'static str> {
        let mut dot = Point(1.0, 2.0).dot(Point(3.0, -4.0));
        if !almost_zero(dot + 5.0) {
            return Err("Dot product is calculated improperly");
        }

        dot = Point(5.0, 0.0).dot(Point(0.0, 6.0));
        if !almost_zero(dot) {
            return Err("Dot product is calculated improperly");
        }

        Ok(())
    }

    #[test]
    fn test_normalization() -> Result<(), &'static str> {
        let p = Point(-3.0, 4.0);
        let p_norm = p.normalize();

        let expected = Point(-0.6, 0.8);
        if !(p_norm - expected).almost_zero() {
            return Err("Normalization is calculated improperly");
        }

        Ok(())
    }

    #[test]
    fn test_orthogonalization() -> Result<(), &'static str> {
        let p = Point(-2.0, 1.0);
        let p2 = p.orthogonalize();

        let p2_expected = Point(-1.0, -2.0);
        if !(p2 - p2_expected).almost_zero() {
            return Err("orthogonalize() should rotate vector 90 degrees");
        }

        let p3 = p2.reverse_orthogonalize();
        if !(p3 - p).almost_zero() {
            return Err("reverse_orthogonalize() should rotate vector -90 degrees");
        }

        Ok(())
    }
}
