use lightgbm_rust::{Booster, predict_type};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a trained LightGBM model
    // Note: You'll need a trained model file to run this example
    let model_path = "model.txt";

    println!("Loading model from: {}", model_path);
    let booster = Booster::load(model_path)?;

    // Get model information
    let num_features = booster.num_features()?;
    let num_classes = booster.num_classes()?;

    println!("Model loaded successfully!");
    println!("Number of features: {}", num_features);
    println!("Number of classes: {}", num_classes);

    // Example: Predict for a single sample with 4 features
    // Data in row-major format: [feature1, feature2, feature3, feature4]
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let num_rows = 1;
    let num_cols = 4;

    println!("\nMaking prediction for data: {:?}", data);
    let predictions = booster.predict(&data, num_rows, num_cols, predict_type::NORMAL)?;

    println!("Predictions: {:?}", predictions);

    // Example: Predict for multiple samples (batch prediction)
    let batch_data = vec![
        1.0, 2.0, 3.0, 4.0,  // Sample 1
        2.0, 3.0, 4.0, 5.0,  // Sample 2
        3.0, 4.0, 5.0, 6.0,  // Sample 3
    ];
    let num_rows = 3;
    let num_cols = 4;

    println!("\nMaking batch prediction...");
    let batch_predictions = booster.predict(&batch_data, num_rows, num_cols, predict_type::NORMAL)?;

    println!("Batch predictions: {:?}", batch_predictions);

    Ok(())
}
