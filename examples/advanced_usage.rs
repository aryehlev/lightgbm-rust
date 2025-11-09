use lightgbm_rust::{Booster, predict_type};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a trained LightGBM model
    let model_path = "model.txt";

    println!("Loading model from: {}", model_path);
    let booster = Booster::load(model_path)?;

    // Get model information
    let num_features = booster.num_features()?;
    let num_classes = booster.num_classes()?;

    println!("Model Information:");
    println!("  Features: {}", num_features);
    println!("  Classes: {}", num_classes);

    // Example data with f32 (more memory efficient for large datasets)
    let data_f32: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0,
        2.0, 3.0, 4.0, 5.0, 6.0,
    ];
    let num_rows = 2;
    let num_cols = 5;

    println!("\n--- Normal Prediction ---");
    let predictions = booster.predict_f32(&data_f32, num_rows, num_cols, predict_type::NORMAL)?;
    println!("Normal predictions: {:?}", predictions);

    println!("\n--- Raw Score Prediction ---");
    let raw_scores = booster.predict_f32(&data_f32, num_rows, num_cols, predict_type::RAW_SCORE)?;
    println!("Raw scores: {:?}", raw_scores);

    println!("\n--- Leaf Index Prediction ---");
    let leaf_indices = booster.predict_f32(&data_f32, num_rows, num_cols, predict_type::LEAF_INDEX)?;
    println!("Leaf indices: {:?}", leaf_indices);

    println!("\n--- Feature Contribution (SHAP) ---");
    let contributions = booster.predict_f32(&data_f32, num_rows, num_cols, predict_type::CONTRIB)?;
    println!("Feature contributions: {:?}", contributions);

    Ok(())
}
