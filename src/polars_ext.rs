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
    /// * `predict_type` - Type of prediction (see `predict_type` module)
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
    /// * `predict_type` - Type of prediction
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
        self.predict(&data, num_rows, num_cols, predict_type)
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
        self.predict(&data, num_rows, num_cols, predict_type)
    }
}

/// Convert a Polars DataFrame to dense f64 data in row-major format
///
/// Optimized column-by-column conversion using Polars' cast for simplicity and speed.
fn dataframe_to_dense(df: &DataFrame) -> LightGBMResult<(Vec<f64>, i32, i32)> {
    let num_rows = df.height();
    let num_cols = df.width();

    if num_rows == 0 || num_cols == 0 {
        return Err(LightGBMError {
            description: "DataFrame has zero rows or columns".to_string(),
        });
    }

    // Pre-allocate with exact size
    let total_elements = num_rows * num_cols;
    let mut data = vec![0.0f64; total_elements];

    // Process column by column - cast to Float64 for simplicity and speed
    for (col_idx, column) in df.get_columns().iter().enumerate() {
        let series = column.as_materialized_series();

        // Cast to Float64 - Polars handles all type conversions efficiently
        let f64_series = series.cast(&DataType::Float64).map_err(|e| LightGBMError {
            description: format!("Failed to cast column to f64: {}", e),
        })?;

        let ca = f64_series.f64().map_err(|e| LightGBMError {
            description: format!("Failed to get f64 array: {}", e),
        })?;

        for (row_idx, opt_val) in ca.iter().enumerate() {
            let val = opt_val.ok_or_else(|| LightGBMError {
                description: format!("Null value at row {}, col {}", row_idx, col_idx),
            })?;
            data[row_idx * num_cols + col_idx] = val;
        }
    }

    Ok((data, num_rows as i32, num_cols as i32))
}

