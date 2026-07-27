//! Tiny dense linear algebra, no external dependencies.
//!
//! Only what the learned-projection (PCA) path needs: a symmetric
//! eigendecomposition via the cyclic Jacobi method. Computation is carried out
//! in `f64` for numerical stability; inputs/outputs at the boundary are `f32`.

/// Cyclic Jacobi eigendecomposition of a symmetric `n×n` matrix (row-major).
///
/// Returns `(eigenvalues, eigenvectors)`. Eigenvector `j` is stored as **column
/// `j`** of the row-major `n×n` matrix `vecs`, i.e. component `i` is
/// `vecs[i*n + j]`. Eigenvalues are returned in the matrix's natural (diagonal)
/// order — callers that want them sorted should sort indices themselves.
pub fn jacobi_eigh(a_in: &[f32], n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a_in.len(), n * n, "matrix must be n*n");
    let mut a: Vec<f64> = a_in.iter().map(|&x| x as f64).collect();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    const MAX_SWEEPS: usize = 100;
    for _ in 0..MAX_SWEEPS {
        // Off-diagonal Frobenius norm (squared). Only upper triangle as A is symmetric.
        let mut off = 0.0f64;
        for p in 0..n {
            let row = &a[p * n + (p + 1)..p * n + n];
            for &val in row {
                off += val * val;
            }
        }
        if off <= 1e-20 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // A <- Jᵀ A J : first the column rotation (A J)...
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                // ...then the row rotation (Jᵀ ·).
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                // Accumulate eigenvectors: V <- V J.
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let eigvals: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigvals, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn jacobi_handles_equal_diagonal_with_nonzero_off_diagonal() {
        // Cette matrice a des diagonales égales et un terme hors diagonale
        // non nul. Ses valeurs propres exactes sont 0 et 2.
        let a = [1.0f32, 1.0, 1.0, 1.0];

        let (vals, vecs) = jacobi_eigh(&a, 2);

        let mut sorted = vals.clone();
        sorted.sort_by(f64::total_cmp);

        assert!(
            sorted[0].abs() < 1e-12,
            "première valeur propre inattendue : {}",
            sorted[0]
        );
        assert!(
            (sorted[1] - 2.0).abs() < 1e-12,
            "seconde valeur propre inattendue : {}",
            sorted[1]
        );

        // Vérification indépendante par reconstruction A = V Λ Vᵀ.
        for i in 0..2 {
            for j in 0..2 {
                let mut reconstructed = 0.0f64;

                for k in 0..2 {
                    reconstructed += vecs[i * 2 + k] * vals[k] * vecs[j * 2 + k];
                }

                let expected = a[i * 2 + j] as f64;

                assert!(
                    (reconstructed - expected).abs() < 1e-12,
                    "reconstruction[{i},{j}] = {reconstructed}, attendu {expected}"
                );
            }
        }
    }

    #[test]
    fn jacobi_reconstructs_and_is_orthonormal() {
        let n = 24;
        let mut rng = Rng::new(42);

        // Build a random *symmetric* matrix.
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            for j in i..n {
                let x = rng.next_gaussian();
                a[i * n + j] = x;
                a[j * n + i] = x;
            }
        }

        let (vals, vecs) = jacobi_eigh(&a, n);

        // 1) Eigenvectors orthonormal: Vᵀ V ≈ I.
        for i in 0..n {
            for j in 0..n {
                let mut dotv = 0.0f64;
                for k in 0..n {
                    dotv += vecs[k * n + i] * vecs[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dotv - expected).abs() < 1e-6, "VᵀV[{i},{j}] = {dotv}");
            }
        }

        // 2) Reconstruction: A ≈ V Λ Vᵀ.
        let mut max_err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0f64;
                for k in 0..n {
                    acc += vecs[i * n + k] * vals[k] * vecs[j * n + k];
                }
                max_err = max_err.max((acc - a[i * n + j] as f64).abs());
            }
        }
        assert!(max_err < 1e-4, "reconstruction error {max_err}");

        // 3) Trace is preserved (sum of eigenvalues == trace).
        let trace: f64 = (0..n).map(|i| a[i * n + i] as f64).sum();
        let sum_vals: f64 = vals.iter().sum();
        assert!((trace - sum_vals).abs() < 1e-4);
    }
}
