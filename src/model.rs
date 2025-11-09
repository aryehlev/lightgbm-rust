use crate::error::{LightGBMError, LightGBMResult};
use crate::sys;
use std::ffi::CString;
use std::path::Path;
use std::ptr;

pub struct Booster {
    handle: sys::BoosterHandle,
}

unsafe impl Send for Booster {}
unsafe impl Sync for Booster {}

impl Booster {
    /// Load a model from a file
    pub fn load<P: AsRef<Path>>(path: P) -> LightGBMResult<Self> {
        let path_c_str = CString::new(path.as_ref().to_str().unwrap()).unwrap();
        let mut handle: sys::BoosterHandle = ptr::null_mut();
        let mut num_iterations = 0i32;

        LightGBMError::check_return_value(unsafe {
            sys::LGBM_BoosterCreateFromModelfile(
                path_c_str.as_ptr(),
                &mut num_iterations,
                &mut handle,
            )
        })?;

        Ok(Booster { handle })
    }

    /// Load a model from a string buffer
    ///
    /// # Arguments
    /// * `model_str` - Model content as a string (text format)
    ///
    /// # Example
    /// ```no_run
    /// use lightgbm_rust::Booster;
    /// use std::fs;
    ///
    /// let model_string = fs::read_to_string("model.txt").unwrap();
    /// let booster = Booster::load_from_string(&model_string).unwrap();
    /// ```
    pub fn load_from_string(model_str: &str) -> LightGBMResult<Self> {
        let model_c_str = CString::new(model_str).unwrap();
        let mut handle: sys::BoosterHandle = ptr::null_mut();
        let mut num_iterations = 0i32;

        LightGBMError::check_return_value(unsafe {
            sys::LGBM_BoosterLoadModelFromString(
                model_c_str.as_ptr(),
                &mut num_iterations,
                &mut handle,
            )
        })?;

        Ok(Booster { handle })
    }

    /// Load a model from a byte buffer
    ///
    /// # Arguments
    /// * `buffer` - Model content as bytes
    ///
    /// # Example
    /// ```no_run
    /// use lightgbm_rust::Booster;
    /// use std::fs;
    ///
    /// let model_bytes = fs::read("model.txt").unwrap();
    /// let booster = Booster::load_from_buffer(&model_bytes).unwrap();
    /// ```
    pub fn load_from_buffer(buffer: &[u8]) -> LightGBMResult<Self> {
        // Convert bytes to string (LightGBM models are text-based)
        let model_str = std::str::from_utf8(buffer)
            .map_err(|e| LightGBMError {
                description: format!("Invalid UTF-8 in model buffer: {}", e),
            })?;
        Self::load_from_string(model_str)
    }

    /// Get the number of features
    pub fn num_features(&self) -> LightGBMResult<i32> {
        let mut num_features = 0i32;
        LightGBMError::check_return_value(unsafe {
            sys::LGBM_BoosterGetNumFeature(self.handle, &mut num_features)
        })?;
        Ok(num_features)
    }

    /// Get the number of classes (for classification models)
    pub fn num_classes(&self) -> LightGBMResult<i32> {
        let mut num_classes = 0i32;
        LightGBMError::check_return_value(unsafe {
            sys::LGBM_BoosterGetNumClasses(self.handle, &mut num_classes)
        })?;
        Ok(num_classes)
    }

    /// Predict for a dense matrix
    ///
    /// # Arguments
    /// * `data` - Input data in row-major format (flattened 2D array)
    /// * `num_rows` - Number of rows (samples)
    /// * `num_cols` - Number of columns (features)
    /// * `predict_type` - Prediction type (0 for normal, 1 for raw score, 2 for leaf index)
    ///
    /// # Returns
    /// Vector of predictions
    pub fn predict(
        &self,
        data: &[f64],
        num_rows: i32,
        num_cols: i32,
        predict_type: i32,
    ) -> LightGBMResult<Vec<f64>> {
        let mut out_len = 0i64;

        // First call to get the output length
        LightGBMError::check_return_value(unsafe {
            sys::LGBM_BoosterPredictForMat(
                self.handle,
                data.as_ptr() as *const std::os::raw::c_void,
                sys::C_API_DTYPE_FLOAT64 as i32,
                num_rows,
                num_cols,
                1, // is_row_major
                predict_type,
                0,  // start_iteration (0 means from the first)
                -1, // num_iteration (-1 means use all)
                ptr::null(),
                &mut out_len,
                ptr::null_mut(),
            )
        })?;

        // Allocate output buffer
        let mut out_result = vec![0.0f64; out_len as usize];

        // Second call to get the actual predictions
        LightGBMError::check_return_value(unsafe {
            sys::LGBM_BoosterPredictForMat(
                self.handle,
                data.as_ptr() as *const std::os::raw::c_void,
                sys::C_API_DTYPE_FLOAT64 as i32,
                num_rows,
                num_cols,
                1, // is_row_major
                predict_type,
                0,  // start_iteration
                -1, // num_iteration
                ptr::null(),
                &mut out_len,
                out_result.as_mut_ptr(),
            )
        })?;

        Ok(out_result)
    }

    /// Predict for f32 data
    pub fn predict_f32(
        &self,
        data: &[f32],
        num_rows: i32,
        num_cols: i32,
        predict_type: i32,
    ) -> LightGBMResult<Vec<f64>> {
        let mut out_len = 0i64;

        // First call to get the output length
        LightGBMError::check_return_value(unsafe {
            sys::LGBM_BoosterPredictForMat(
                self.handle,
                data.as_ptr() as *const std::os::raw::c_void,
                sys::C_API_DTYPE_FLOAT32 as i32,
                num_rows,
                num_cols,
                1, // is_row_major
                predict_type,
                0,  // start_iteration (0 means from the first)
                -1, // num_iteration (-1 means use all)
                ptr::null(),
                &mut out_len,
                ptr::null_mut(),
            )
        })?;

        // Allocate output buffer
        let mut out_result = vec![0.0f64; out_len as usize];

        // Second call to get the actual predictions
        LightGBMError::check_return_value(unsafe {
            sys::LGBM_BoosterPredictForMat(
                self.handle,
                data.as_ptr() as *const std::os::raw::c_void,
                sys::C_API_DTYPE_FLOAT32 as i32,
                num_rows,
                num_cols,
                1, // is_row_major
                predict_type,
                0,  // start_iteration
                -1, // num_iteration
                ptr::null(),
                &mut out_len,
                out_result.as_mut_ptr(),
            )
        })?;

        Ok(out_result)
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        unsafe {
            sys::LGBM_BoosterFree(self.handle);
        }
    }
}
