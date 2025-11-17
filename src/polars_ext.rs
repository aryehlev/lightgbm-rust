use crate::error::{LightGBMError, LightGBMResult};
use crate::Booster;
use polars::prelude::*;

/// Extension trait for LightGBM Booster to support Polars DataFrames
pub trait BoosterPolarsExt {
    /// Predict using a Polars DataFrame as input
    ///
    /// This method efficiently converts the DataFrame to the format LightGBM expects
    /// and runs prediction. All numeric columns will be used as features.
    ///
    /// # Arguments
    /// * `df` - Input DataFrame with numeric features
    /// * `predict_type` - Prediction type (see `predict_type` module constants)
    ///
    /// # Returns
    /// A vector of prediction values
    ///
    /// # Example
    /// ```no_run
    /// # use lightgbm_rust::{Booster, BoosterPolarsExt, predict_type};
    /// # use polars::prelude::*;
    /// let booster = Booster::load("model.txt").unwrap();
    ///
    /// let df = df! {
    ///     "feature1" => [1.0f32, 2.0, 3.0],
    ///     "feature2" => [4.0f32, 5.0, 6.0],
    /// }.unwrap();
    ///
    /// let predictions = booster.predict_dataframe(&df, predict_type::NORMAL).unwrap();
    /// ```
    fn predict_dataframe(&self, df: &DataFrame, predict_type: i32) -> LightGBMResult<Vec<f64>>;

    /// Predict using specific columns from a Polars DataFrame
    ///
    /// # Arguments
    /// * `df` - Input DataFrame
    /// * `columns` - Column names to use as features (in order)
    /// * `predict_type` - Prediction type
    fn predict_dataframe_with_columns(
        &self,
        df: &DataFrame,
        columns: &[&str],
        predict_type: i32,
    ) -> LightGBMResult<Vec<f64>>;
}

impl BoosterPolarsExt for Booster {
    fn predict_dataframe(&self, df: &DataFrame, predict_type: i32) -> LightGBMResult<Vec<f64>> {
        let (data, num_rows, num_cols) = dataframe_to_dense(df)?;
        self.predict_f32(&data, num_rows, num_cols, predict_type)
    }

    fn predict_dataframe_with_columns(
        &self,
        df: &DataFrame,
        columns: &[&str],
        predict_type: i32,
    ) -> LightGBMResult<Vec<f64>> {
        let column_names: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        let selected = df.select(column_names).map_err(|e| LightGBMError {
            description: format!("Failed to select columns: {}", e),
        })?;

        let (data, num_rows, num_cols) = dataframe_to_dense(&selected)?;
        self.predict_f32(&data, num_rows, num_cols, predict_type)
    }
}

/// Convert a Polars DataFrame to dense f32 data in row-major format
///
/// This is optimized for efficient memory layout matching LightGBM's expectations.
fn dataframe_to_dense(df: &DataFrame) -> LightGBMResult<(Vec<f32>, i32, i32)> {
    let num_rows = df.height();
    let num_cols = df.width();

    if num_rows == 0 || num_cols == 0 {
        return Err(LightGBMError {
            description: "DataFrame has zero rows or columns".to_string(),
        });
    }

    // Pre-allocate output buffer
    let mut data = Vec::with_capacity(num_rows * num_cols);

    // Convert row by row for better cache locality
    for row_idx in 0..num_rows {
        for col in df.get_columns() {
            let series = col.as_materialized_series();
            let value = extract_f32_value(series, row_idx)?;
            data.push(value);
        }
    }

    Ok((data, num_rows as i32, num_cols as i32))
}

/// Extract an f32 value from a Series at the given index
///
/// Supports conversion from all numeric types and booleans.
fn extract_f32_value(series: &Series, idx: usize) -> LightGBMResult<f32> {
    use DataType::*;

    match series.dtype() {
        Float32 => {
            let ca = series.f32().map_err(|e| LightGBMError {
                description: format!("Failed to cast to f32: {}", e),
            })?;
            ca.get(idx).ok_or_else(|| LightGBMError {
                description: format!("Null value at index {}", idx),
            })
        }
        Float64 => {
            let ca = series.f64().map_err(|e| LightGBMError {
                description: format!("Failed to cast to f64: {}", e),
            })?;
            Ok(ca.get(idx).ok_or_else(|| LightGBMError {
                description: format!("Null value at index {}", idx),
            })? as f32)
        }
        Int8 => {
            let ca = series.i8().map_err(|e| LightGBMError {
                description: format!("Failed to cast to i8: {}", e),
            })?;
            Ok(ca.get(idx).ok_or_else(|| LightGBMError {
                description: format!("Null value at index {}", idx),
            })? as f32)
        }
        Int16 => {
            let ca = series.i16().map_err(|e| LightGBMError {
                description: format!("Failed to cast to i16: {}", e),
            })?;
            Ok(ca.get(idx).ok_or_else(|| LightGBMError {
                description: format!("Null value at index {}", idx),
            })? as f32)
        }
        Int32 => {
            let ca = series.i32().map_err(|e| LightGBMError {
                description: format!("Failed to cast to i32: {}", e),
            })?;
            Ok(ca.get(idx).ok_or_else(|| LightGBMError {
                description: format!("Null value at index {}", idx),
            })? as f32)
        }
        Int64 => {
            let ca = series.i64().map_err(|e| LightGBMError {
                description: format!("Failed to cast to i64: {}", e),
            })?;
            Ok(ca.get(idx).ok_or_else(|| LightGBMError {
                description: format!("Null value at index {}", idx),
            })? as f32)
        }
        UInt8 => {
            let ca = series.u8().map_err(|e| LightGBMError {
                description: format!("Failed to cast to u8: {}", e),
            })?;
            Ok(ca.get(idx).ok_or_else(|| LightGBMError {
                description: format!("Null value at index {}", idx),
            })? as f32)
        }
        UInt16 => {
            let ca = series.u16().map_err(|e| LightGBMError {
                description: format!("Failed to cast to u16: {}", e),
            })?;
            Ok(ca.get(idx).ok_or_else(|| LightGBMError {
                description: format!("Null value at index {}", idx),
            })? as f32)
        }
        UInt32 => {
            let ca = series.u32().map_err(|e| LightGBMError {
                description: format!("Failed to cast to u32: {}", e),
            })?;
            Ok(ca.get(idx).ok_or_else(|| LightGBMError {
                description: format!("Null value at index {}", idx),
            })? as f32)
        }
        UInt64 => {
            let ca = series.u64().map_err(|e| LightGBMError {
                description: format!("Failed to cast to u64: {}", e),
            })?;
            Ok(ca.get(idx).ok_or_else(|| LightGBMError {
                description: format!("Null value at index {}", idx),
            })? as f32)
        }
        Boolean => {
            let ca = series.bool().map_err(|e| LightGBMError {
                description: format!("Failed to cast to bool: {}", e),
            })?;
            Ok(
                if ca.get(idx).ok_or_else(|| LightGBMError {
                    description: format!("Null value at index {}", idx),
                })? {
                    1.0
                } else {
                    0.0
                },
            )
        }
        dt => Err(LightGBMError {
            description: format!("Unsupported data type for conversion to f32: {}", dt),
        }),
    }
}
