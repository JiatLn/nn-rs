use std::fmt::{self, Debug, Display};

use crate::Matrix;

impl Display for Matrix<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (h, _w) = self.shape();
        let mut s = String::new();
        s += r#"
            "#;
        for i in 0..h {
            s += &format!("{:?}", self.0[i]);
            s += r#"
            "#
        }
        write!(f, "{}", s)
    }
}

impl Debug for Matrix<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (h, w) = self.shape();
        let mut s = String::new();
        s += r#"
            "#;
        for i in 0..h {
            s += "|";
            for j in 0..w {
                s += &format!(" {:?} ", self.0[i][j]);
            }
            s += "|";
            s += r#"
            "#
        }
        write!(f, "{}", s)
    }
}
