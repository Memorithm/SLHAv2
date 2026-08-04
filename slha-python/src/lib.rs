use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use scirust::attention::slha_v2::{
    quantize_latent, quantize_latent_grouped, quantize_latent_mix3, quantize_latent_mixed,
    quantize_latent_nf4, quantize_latent_tq3, LatentCodec, SciRustSlhaTile, D_C, D_S, FLAG_MIX3,
    FLAG_MIXED, FLAG_NF4, FLAG_TQ3, FLAG_TQ3_NOCORR, FLAG_WARM, LATENT_BYTES, N_GROUPS,
    RESIDUAL_WORDS,
};
use scirust::audit;
use std::panic::{catch_unwind, AssertUnwindSafe};

const CODEC_FLAGS: u16 = FLAG_NF4 | FLAG_MIXED | FLAG_TQ3 | FLAG_MIX3;
const KNOWN_FLAGS: u16 = FLAG_WARM | FLAG_NF4 | FLAG_MIXED | FLAG_TQ3 | FLAG_TQ3_NOCORR | FLAG_MIX3;

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn runtime_error(message: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(message.into())
}

fn validate_finite(values: &[f32], name: &str) -> PyResult<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(value_error(format!("{name}[{index}] must be finite")));
    }

    Ok(())
}

fn validate_tile(tile: &SciRustSlhaTile) -> PyResult<()> {
    let unknown = tile.flags & !KNOWN_FLAGS;

    if unknown != 0 {
        return Err(value_error(format!(
            "tile contains unknown flag bits: 0x{unknown:04x}"
        )));
    }

    let codecs = tile.flags & CODEC_FLAGS;

    if codecs.count_ones() > 1 {
        return Err(value_error("tile cannot select more than one latent codec"));
    }

    if tile.flags & FLAG_TQ3_NOCORR != 0 && tile.flags & (FLAG_TQ3 | FLAG_MIX3) == 0 {
        return Err(value_error(
            "FLAG_TQ3_NOCORR requires the TQ3 or MIX3 codec",
        ));
    }

    for (name, value) in [
        ("scale", tile.scale),
        ("dynamic_lambda", tile.dynamic_lambda),
        ("residual_sigma", tile.residual_sigma),
    ] {
        if !value.is_finite() {
            return Err(value_error(format!("{name} must be finite")));
        }

        if value < 0.0 {
            return Err(value_error(format!("{name} must be non-negative")));
        }
    }

    Ok(())
}

fn codec_from_name(name: &str) -> PyResult<LatentCodec> {
    match name {
        "int4" => Ok(LatentCodec::Int4Single),
        "grouped" => Ok(LatentCodec::Int4Grouped),
        "nf4" => Ok(LatentCodec::Nf4),
        "mixed" => Ok(LatentCodec::Mixed),
        "tq3" => Ok(LatentCodec::Tq3),
        "mix3" => Ok(LatentCodec::Mix3),
        _ => Err(value_error(format!(
            "unknown codec {name:?}; expected one of: \
             int4, grouped, nf4, mixed, tq3, mix3"
        ))),
    }
}

fn codec_name(tile: &SciRustSlhaTile) -> &'static str {
    if tile.flags & FLAG_MIX3 != 0 {
        "mix3"
    } else if tile.flags & FLAG_MIXED != 0 {
        "mixed"
    } else if tile.flags & FLAG_TQ3 != 0 {
        "tq3"
    } else if tile.flags & FLAG_NF4 != 0 {
        "nf4"
    } else {
        // INT4 simple and grouped deliberately share flag 0 in the core
        // layout. group_scales still preserves the actual micro-scales.
        "int4"
    }
}

fn read_f32_buffer<const N: usize>(
    py: Python<'_>,
    buffer: &PyBuffer<f32>,
    name: &str,
) -> PyResult<[f32; N]> {
    if buffer.dimensions() != 1 || buffer.shape()[0] != N {
        return Err(value_error(format!(
            "{name} must be a one-dimensional float32 buffer of length {N}"
        )));
    }

    // PyO3 validates the element format and performs a safe copy. This also
    // supports strided and potentially unaligned buffer exporters without
    // constructing a Rust slice from their raw pointer.
    let values = buffer.to_vec(py)?;

    validate_finite(&values, name)?;

    values.try_into().map_err(|values: Vec<f32>| {
        value_error(format!(
            "{name} contains {} elements; expected {N}",
            values.len()
        ))
    })
}

fn read_u64_buffer<const N: usize>(
    py: Python<'_>,
    buffer: &PyBuffer<u64>,
    name: &str,
) -> PyResult<[u64; N]> {
    if buffer.dimensions() != 1 || buffer.shape()[0] != N {
        return Err(value_error(format!(
            "{name} must be a one-dimensional uint64 buffer of length {N}"
        )));
    }

    let values = buffer.to_vec(py)?;

    values.try_into().map_err(|values: Vec<u64>| {
        value_error(format!(
            "{name} contains {} elements; expected {N}",
            values.len()
        ))
    })
}

fn encode_latent(
    latent: &[f32; D_C],
    codec: LatentCodec,
    token_id: u32,
    position: u32,
    head_id: u16,
) -> PyResult<PySlhaTile> {
    validate_finite(latent, "latent")?;

    let encoded = catch_unwind(AssertUnwindSafe(|| match codec {
        LatentCodec::Int4Single => {
            let (bytes, scale) = quantize_latent(latent);
            (bytes, scale, [255u8; N_GROUPS], 0)
        }
        LatentCodec::Int4Grouped => {
            let (bytes, scale, groups) = quantize_latent_grouped(latent);
            (bytes, scale, groups, 0)
        }
        LatentCodec::Nf4 => {
            let (bytes, scale, groups) = quantize_latent_nf4(latent);
            (bytes, scale, groups, FLAG_NF4)
        }
        LatentCodec::Mixed => {
            let (bytes, scale, groups) = quantize_latent_mixed(latent);
            (bytes, scale, groups, FLAG_MIXED)
        }
        LatentCodec::Tq3 => {
            let (bytes, scale, groups) = quantize_latent_tq3(latent);
            (bytes, scale, groups, FLAG_TQ3)
        }
        LatentCodec::Mix3 => {
            let (bytes, scale, groups) = quantize_latent_mix3(latent);
            (bytes, scale, groups, FLAG_MIX3)
        }
    }))
    .map_err(|_| runtime_error("panic while quantizing the latent vector"))?;

    let tile = SciRustSlhaTile {
        latent_kv: encoded.0,
        residual_bitmap: [0; RESIDUAL_WORDS],
        scale: encoded.1,
        dynamic_lambda: 0.0,
        residual_sigma: 0.0,
        token_id,
        position,
        head_id,
        flags: encoded.3,
        group_scales: encoded.2,
    };

    validate_tile(&tile)?;

    Ok(PySlhaTile { inner: tile })
}

fn checked_score(
    tile: &SciRustSlhaTile,
    q_coarse: &[f32; D_C],
    q_sign: &[u64; RESIDUAL_WORDS],
) -> PyResult<f32> {
    validate_tile(tile)?;
    validate_finite(q_coarse, "q_coarse")?;

    let score = catch_unwind(AssertUnwindSafe(|| tile.compute_score(q_coarse, q_sign)))
        .map_err(|_| runtime_error("panic while computing the SLHA score"))?;

    if !score.is_finite() {
        return Err(value_error(
            "SLHA score is non-finite; check the tile and query magnitudes",
        ));
    }

    Ok(score)
}

/// Python wrapper for `SciRustSlhaTile`.
///
/// `skip_from_py_object`: nothing extracts `PySlhaTile` by value (all methods
/// take `&self`), so the automatic `FromPyObject` impl pyo3 is deprecating was
/// dead code — skipping it matches the upcoming pyo3 default.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct PySlhaTile {
    pub inner: SciRustSlhaTile,
}

#[pymethods]
impl PySlhaTile {
    #[new]
    #[pyo3(signature = (
        latent_kv,
        residual_bitmap,
        scale,
        dynamic_lambda,
        residual_sigma,
        token_id,
        position,
        head_id,
        flags,
        group_scales
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        latent_kv: [u8; LATENT_BYTES],
        residual_bitmap: [u64; RESIDUAL_WORDS],
        scale: f32,
        dynamic_lambda: f32,
        residual_sigma: f32,
        token_id: u32,
        position: u32,
        head_id: u16,
        flags: u16,
        group_scales: [u8; N_GROUPS],
    ) -> PyResult<Self> {
        let tile = SciRustSlhaTile {
            latent_kv,
            residual_bitmap,
            scale,
            dynamic_lambda,
            residual_sigma,
            token_id,
            position,
            head_id,
            flags,
            group_scales,
        };

        validate_tile(&tile)?;

        Ok(Self { inner: tile })
    }

    #[getter]
    fn latent_kv<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.latent_kv)
    }

    #[getter]
    fn residual_bitmap(&self) -> Vec<u64> {
        self.inner.residual_bitmap.to_vec()
    }

    #[getter]
    fn scale(&self) -> f32 {
        self.inner.scale
    }

    #[getter]
    fn dynamic_lambda(&self) -> f32 {
        self.inner.dynamic_lambda
    }

    #[getter]
    fn residual_sigma(&self) -> f32 {
        self.inner.residual_sigma
    }

    #[getter]
    fn token_id(&self) -> u32 {
        self.inner.token_id
    }

    #[getter]
    fn position(&self) -> u32 {
        self.inner.position
    }

    #[getter]
    fn head_id(&self) -> u16 {
        self.inner.head_id
    }

    #[getter]
    fn flags(&self) -> u16 {
        self.inner.flags
    }

    #[getter]
    fn group_scales(&self) -> Vec<u8> {
        self.inner.group_scales.to_vec()
    }

    #[getter]
    fn codec(&self) -> &'static str {
        codec_name(&self.inner)
    }

    /// Compute the score for this tile.
    ///
    /// Inputs may be C-contiguous or strided buffers. They are copied through
    /// PyO3's checked buffer API before Rust performs the computation.
    fn compute_score(
        &self,
        py: Python<'_>,
        q_coarse: PyBuffer<f32>,
        q_sign: PyBuffer<u64>,
    ) -> PyResult<f32> {
        let q_coarse = read_f32_buffer::<D_C>(py, &q_coarse, "q_coarse")?;
        let q_sign = read_u64_buffer::<RESIDUAL_WORDS>(py, &q_sign, "q_sign")?;

        checked_score(&self.inner, &q_coarse, &q_sign)
    }

    /// Materialize the canonical dequantized 128-dimensional latent.
    fn dequantize(&self) -> PyResult<Vec<f32>> {
        validate_tile(&self.inner)?;

        let values = catch_unwind(AssertUnwindSafe(|| self.inner.dequant_latent()))
            .map_err(|_| runtime_error("panic while dequantizing the tile"))?;

        validate_finite(&values, "dequantized_latent")?;

        Ok(values.to_vec())
    }

    fn __repr__(&self) -> String {
        format!(
            "PySlhaTile(codec='{}', position={}, token_id={}, flags=0x{:04x})",
            codec_name(&self.inner),
            self.inner.position,
            self.inner.token_id,
            self.inner.flags,
        )
    }
}

/// Compress a 128-dimensional latent vector into a 128-byte SLHA tile.
#[pyfunction]
#[pyo3(signature = (
    latent,
    codec = "grouped",
    token_id = 0,
    position = 0,
    head_id = 0
))]
fn compress_latent(
    py: Python<'_>,
    latent: PyBuffer<f32>,
    codec: &str,
    token_id: u32,
    position: u32,
    head_id: u16,
) -> PyResult<PySlhaTile> {
    let latent = read_f32_buffer::<D_C>(py, &latent, "latent")?;
    let codec = codec_from_name(codec)?;

    encode_latent(&latent, codec, token_id, position, head_id)
}

/// Run the SLHA v2 self-audit and return a JSON string.
///
/// The audit performs no Python work, so the interpreter is released while it
/// runs. Any Rust panic is converted into `RuntimeError`.
#[pyfunction]
fn run_audit(py: Python<'_>) -> PyResult<String> {
    // pyo3 0.24 renamed `allow_threads` to `detach`; 0.29 removed the old name.
    let result = py.detach(|| catch_unwind(AssertUnwindSafe(|| audit::run().to_compact())));

    result.map_err(|_| runtime_error("panic while running the SLHA self-audit"))
}

/// Native `slha_core` Python module.
#[pymodule]
fn slha_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySlhaTile>()?;
    m.add_function(wrap_pyfunction!(compress_latent, m)?)?;
    m.add_function(wrap_pyfunction!(run_audit, m)?)?;

    m.add("D_C", D_C)?;
    m.add("D_S", D_S)?;
    m.add("LATENT_BYTES", LATENT_BYTES)?;
    m.add("RESIDUAL_WORDS", RESIDUAL_WORDS)?;
    m.add("N_GROUPS", N_GROUPS)?;

    m.add("FLAG_WARM", FLAG_WARM)?;
    m.add("FLAG_NF4", FLAG_NF4)?;
    m.add("FLAG_MIXED", FLAG_MIXED)?;
    m.add("FLAG_TQ3", FLAG_TQ3)?;
    m.add("FLAG_TQ3_NOCORR", FLAG_TQ3_NOCORR)?;
    m.add("FLAG_MIX3", FLAG_MIX3)?;

    m.add("CODEC_INT4", "int4")?;
    m.add("CODEC_GROUPED", "grouped")?;
    m.add("CODEC_NF4", "nf4")?;
    m.add("CODEC_MIXED", "mixed")?;
    m.add("CODEC_TQ3", "tq3")?;
    m.add("CODEC_MIX3", "mix3")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_latent() -> [f32; D_C] {
        core::array::from_fn(|index| {
            let base = ((index * 17 + 11) % 43) as f32 - 21.0;
            base / (2.0 + (index % 7) as f32)
        })
    }

    #[test]
    fn all_codecs_encode_and_dequantize_to_finite_values() {
        let latent = sample_latent();

        for (codec, expected_flag) in [
            (LatentCodec::Int4Single, 0),
            (LatentCodec::Int4Grouped, 0),
            (LatentCodec::Nf4, FLAG_NF4),
            (LatentCodec::Mixed, FLAG_MIXED),
            (LatentCodec::Tq3, FLAG_TQ3),
            (LatentCodec::Mix3, FLAG_MIX3),
        ] {
            let tile = encode_latent(&latent, codec, 7, 9, 2).expect("encode");

            assert_eq!(tile.inner.flags & CODEC_FLAGS, expected_flag);
            assert_eq!(tile.inner.token_id, 7);
            assert_eq!(tile.inner.position, 9);
            assert_eq!(tile.inner.head_id, 2);

            let decoded = tile.inner.dequant_latent();
            assert!(decoded.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn invalid_tile_metadata_and_flag_combinations_are_rejected() {
        let mut tile = encode_latent(&sample_latent(), LatentCodec::Int4Grouped, 0, 0, 0)
            .expect("encode")
            .inner;

        tile.scale = f32::NAN;
        assert!(validate_tile(&tile).is_err());

        tile.scale = 1.0;
        tile.dynamic_lambda = -1.0;
        assert!(validate_tile(&tile).is_err());

        tile.dynamic_lambda = 0.0;
        tile.flags = FLAG_NF4 | FLAG_TQ3;
        assert!(validate_tile(&tile).is_err());

        tile.flags = FLAG_TQ3_NOCORR;
        assert!(validate_tile(&tile).is_err());

        tile.flags = 1 << 15;
        assert!(validate_tile(&tile).is_err());
    }

    #[test]
    fn checked_score_rejects_non_finite_inputs_and_outputs() {
        let tile = encode_latent(&sample_latent(), LatentCodec::Int4Grouped, 0, 0, 0)
            .expect("encode")
            .inner;

        let mut query = [0.0f32; D_C];
        let signs = [0u64; RESIDUAL_WORDS];

        query[4] = f32::NAN;
        assert!(checked_score(&tile, &query, &signs).is_err());

        let mut huge_tile = tile;
        huge_tile.scale = f32::MAX;
        huge_tile.group_scales = [255; N_GROUPS];
        huge_tile.latent_kv = [0xFF; LATENT_BYTES];

        let huge_query = [f32::MAX; D_C];
        assert!(checked_score(&huge_tile, &huge_query, &signs).is_err());
    }

    #[test]
    fn codec_names_are_strict_and_complete() {
        for name in ["int4", "grouped", "nf4", "mixed", "tq3", "mix3"] {
            assert!(codec_from_name(name).is_ok(), "{name}");
        }

        for invalid in ["", "INT4", "group", "tq4", "mix-3"] {
            assert!(codec_from_name(invalid).is_err(), "{invalid}");
        }
    }
}
