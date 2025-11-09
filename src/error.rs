use crate::sys;
use std::ffi::CStr;
use std::fmt;

pub type LightGBMResult<T> = std::result::Result<T, LightGBMError>;

#[derive(Debug, Eq, PartialEq)]
pub struct LightGBMError {
    pub description: String,
}

impl LightGBMError {
    /// Check the return value from a LightGBM FFI call, and return the last error message on error.
    /// Return values of 0 are treated as success, non-zero values are treated as errors.
    pub fn check_return_value(ret_val: i32) -> LightGBMResult<()> {
        if ret_val == 0 {
            Ok(())
        } else {
            Err(LightGBMError::fetch_lightgbm_error())
        }
    }

    /// Fetch current error message from LightGBM.
    fn fetch_lightgbm_error() -> Self {
        let c_str = unsafe { CStr::from_ptr(sys::LGBM_GetLastError()) };
        let str_slice = c_str.to_str().unwrap_or("Unknown error");
        LightGBMError {
            description: str_slice.to_owned(),
        }
    }
}

impl fmt::Display for LightGBMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

impl std::error::Error for LightGBMError {}
