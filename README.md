# LightGBM Rust Bindings

Rust bindings for [LightGBM](https://github.com/microsoft/LightGBM), a fast, distributed, high-performance gradient boosting framework. This crate provides a safe and ergonomic Rust interface to LightGBM's C API.

## Features

- **Cross-platform**: Works on Linux, macOS, and Windows
- **Self-contained**: Downloads LightGBM binaries at build time - no system dependencies required
- **Version control**: Specify different LightGBM versions via environment variable
- **Safe Rust API**: Memory-safe wrapper around LightGBM's C API
- **Multiple prediction types**: Support for normal predictions, raw scores, leaf indices, and SHAP values
- **Debugger-friendly**: Proper rpath configuration for IDE debugging
- **GPU support**: Optional GPU acceleration (requires `gpu` feature)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
lightgbm-rust = "0.1.0"
```

For GPU support:

```toml
[dependencies]
lightgbm-rust = { version = "0.1.0", features = ["gpu"] }
```

## Quick Start

```rust
use lightgbm_rust::{Booster, predict_type};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a trained LightGBM model
    let booster = Booster::load("model.txt")?;

    // Get model information
    println!("Features: {}", booster.num_features()?);
    println!("Classes: {}", booster.num_classes()?);

    // Make predictions with numeric features
    let data = vec![1.0, 2.0, 3.0, 4.0];  // Single sample with 4 features
    let predictions = booster.predict(&data, 1, 4, predict_type::NORMAL)?;

    println!("Predictions: {:?}", predictions);

    Ok(())
}
```

## Usage Examples

### Basic Usage

```rust
use lightgbm_rust::{Booster, predict_type};

// Load model from file
let booster = Booster::load("model.txt")?;

// Single prediction
let data = vec![1.0, 2.0, 3.0, 4.0];
let predictions = booster.predict(&data, 1, 4, predict_type::NORMAL)?;

// Batch prediction (3 samples with 4 features each)
let batch_data = vec![
    1.0, 2.0, 3.0, 4.0,  // Sample 1
    2.0, 3.0, 4.0, 5.0,  // Sample 2
    3.0, 4.0, 5.0, 6.0,  // Sample 3
];
let batch_predictions = booster.predict(&batch_data, 3, 4, predict_type::NORMAL)?;
```

### Loading from Buffer

```rust
use lightgbm_rust::Booster;
use std::fs;

// Load from string
let model_string = fs::read_to_string("model.txt")?;
let booster = Booster::load_from_string(&model_string)?;

// Load from byte buffer
let model_bytes = fs::read("model.txt")?;
let booster = Booster::load_from_buffer(&model_bytes)?;
```

### Using f32 for Memory Efficiency

```rust
use lightgbm_rust::{Booster, predict_type};

let booster = Booster::load("model.txt")?;

// Use f32 instead of f64 for large datasets (predict accepts both)
let data_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
let predictions = booster.predict(&data_f32, 1, 4, predict_type::NORMAL)?;
```

### Different Prediction Types

```rust
use lightgbm_rust::{Booster, predict_type};

let booster = Booster::load("model.txt")?;
let data = vec![1.0, 2.0, 3.0, 4.0];

// Normal prediction (default)
let normal = booster.predict(&data, 1, 4, predict_type::NORMAL)?;

// Raw scores (before sigmoid/softmax)
let raw = booster.predict(&data, 1, 4, predict_type::RAW_SCORE)?;

// Leaf indices (which leaf each tree predicts)
let leaves = booster.predict(&data, 1, 4, predict_type::LEAF_INDEX)?;

// SHAP feature contributions
let shap = booster.predict(&data, 1, 4, predict_type::CONTRIB)?;
```

### Thread Safety

**Important:** `Booster` is **NOT thread-safe** by default. The underlying LightGBM C API does not guarantee thread-safety for concurrent predictions.

For multi-threaded use cases, choose one of these approaches:

**Option 1: One Booster per thread (recommended)**
```rust
use std::thread;

let model_bytes = std::fs::read("model.txt")?;

let handles: Vec<_> = (0..4).map(|_| {
    let model_bytes = model_bytes.clone();
    thread::spawn(move || {
        let booster = Booster::load_from_buffer(&model_bytes).unwrap();
        booster.predict(&[1.0, 2.0, 3.0], 1, 3, predict_type::NORMAL)
    })
}).collect();

for handle in handles {
    let result = handle.join().unwrap()?;
    println!("Result: {:?}", result);
}
```

**Option 2: Shared access with Arc<Mutex<Booster>>**
```rust
use std::sync::{Arc, Mutex};
use std::thread;

let booster = Arc::new(Mutex::new(Booster::load("model.txt")?));

let handles: Vec<_> = (0..4).map(|_| {
    let booster = booster.clone();
    thread::spawn(move || {
        let booster = booster.lock().unwrap();
        booster.predict(&[1.0, 2.0, 3.0], 1, 3, predict_type::NORMAL)
    })
}).collect();

for handle in handles {
    let result = handle.join().unwrap()?;
    println!("Result: {:?}", result);
}
```

## Configuration

### LightGBM Version

You can specify which version of LightGBM to use by setting the `LIGHTGBM_VERSION` environment variable:

```bash
export LIGHTGBM_VERSION=4.6.0
cargo build
```

The default version is `4.6.0`.

**Supported versions:** Any version from `3.0.0` onwards that has pre-built binaries available on the [LightGBM releases page](https://github.com/microsoft/LightGBM/releases).

The library automatically:
- Downloads the correct binary for your platform and specified version
- Fetches the matching C API headers for that version
- Generates version-specific Rust bindings
- Handles API differences across versions
