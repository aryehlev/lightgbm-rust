// Include the LightGBM C API bindings
mod sys;

mod error;
pub use crate::error::{LightGBMError, LightGBMResult};

mod model;
pub use crate::model::Booster;

// Re-export prediction type constants for convenience
pub mod predict_type {
    /// Normal prediction
    pub const NORMAL: i32 = 0;
    /// Raw score prediction
    pub const RAW_SCORE: i32 = 1;
    /// Leaf index prediction
    pub const LEAF_INDEX: i32 = 2;
    /// Feature contribution (SHAP values)
    pub const CONTRIB: i32 = 3;
}
