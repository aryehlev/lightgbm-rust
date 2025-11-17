extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};
use std::fs;
use std::io;

fn get_lightgbm_version() -> String {
    env::var("LIGHTGBM_VERSION").unwrap_or_else(|_| "4.6.0".to_string())
}

fn get_platform_info() -> (String, String) {
    let target = env::var("TARGET").unwrap();

    // Determine OS
    let os = if target.contains("apple-darwin") {
        "darwin"
    } else if target.contains("linux") {
        "linux"
    } else if target.contains("windows") {
        "windows"
    } else {
        panic!("Unsupported target: {}", target);
    };

    // Determine architecture
    let arch = if target.contains("x86_64") {
        "x86_64"
    } else if target.contains("aarch64") || target.contains("arm64") {
        "aarch64"
    } else if target.contains("i686") || target.contains("i586") {
        "i686"
    } else {
        panic!("Unsupported architecture for target: {}", target);
    };

    (os.to_string(), arch.to_string())
}

fn download_lightgbm_headers(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let version = get_lightgbm_version();

    // Create the include/LightGBM directory
    let include_dir = out_dir.join("include/LightGBM");
    fs::create_dir_all(&include_dir)?;

    // Download the c_api.h file
    let c_api_url = format!(
        "https://raw.githubusercontent.com/microsoft/LightGBM/v{}/include/LightGBM/c_api.h",
        version
    );

    println!("cargo:warning=Downloading c_api.h from: {}", c_api_url);

    let response = ureq::get(&c_api_url).call()?;
    let status = response.status();
    if status < 200 || status >= 300 {
        return Err(format!("Failed to download c_api.h: HTTP {}", status).into());
    }

    let c_api_path = include_dir.join("c_api.h");
    let mut file = fs::File::create(&c_api_path)?;
    io::copy(&mut response.into_reader(), &mut file)?;

    // Also download export.h which is referenced by c_api.h
    let export_url = format!(
        "https://raw.githubusercontent.com/microsoft/LightGBM/v{}/include/LightGBM/export.h",
        version
    );

    println!("cargo:warning=Downloading export.h from: {}", export_url);

    let response = ureq::get(&export_url).call()?;
    let status = response.status();
    if status < 200 || status >= 300 {
        return Err(format!("Failed to download export.h: HTTP {}", status).into());
    }

    let export_path = include_dir.join("export.h");
    let mut file = fs::File::create(&export_path)?;
    io::copy(&mut response.into_reader(), &mut file)?;

    // Try to download arrow.h which is referenced by c_api.h (added in v4.2.0)
    // For older versions, this file doesn't exist, so we skip it
    let arrow_url = format!(
        "https://raw.githubusercontent.com/microsoft/LightGBM/v{}/include/LightGBM/arrow.h",
        version
    );

    println!("cargo:warning=Attempting to download arrow.h from: {}", arrow_url);

    match ureq::get(&arrow_url).call() {
        Ok(response) if response.status() >= 200 && response.status() < 300 => {
            let arrow_path = include_dir.join("arrow.h");
            let mut file = fs::File::create(&arrow_path)?;
            io::copy(&mut response.into_reader(), &mut file)?;
            println!("cargo:warning=Successfully downloaded arrow.h");

            // Also try to download arrow.tpp which is referenced by arrow.h
            let arrow_tpp_url = format!(
                "https://raw.githubusercontent.com/microsoft/LightGBM/v{}/include/LightGBM/arrow.tpp",
                version
            );

            println!("cargo:warning=Attempting to download arrow.tpp from: {}", arrow_tpp_url);

            match ureq::get(&arrow_tpp_url).call() {
                Ok(resp) if resp.status() >= 200 && resp.status() < 300 => {
                    let arrow_tpp_path = include_dir.join("arrow.tpp");
                    let mut file = fs::File::create(&arrow_tpp_path)?;
                    io::copy(&mut resp.into_reader(), &mut file)?;
                    println!("cargo:warning=Successfully downloaded arrow.tpp");
                }
                _ => {
                    println!("cargo:warning=arrow.tpp not available for this version (optional)");
                }
            }
        }
        _ => {
            println!("cargo:warning=arrow.h not available for this version (optional, only in v4.2.0+)");
        }
    }

    Ok(())
}

fn download_compiled_library(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (os, arch) = get_platform_info();
    let version = get_lightgbm_version();

    // Create the library directory
    let lib_dir = out_dir.join("libs");
    fs::create_dir_all(&lib_dir)?;

    // For macOS and Linux, we need to download Python wheels and extract the library
    match os.as_str() {
        "darwin" => {
            // Determine the wheel filename based on architecture
            let wheel_name = match arch.as_str() {
                "aarch64" => format!("lightgbm-{}-py3-none-macosx_12_0_arm64.whl", version),
                "x86_64" => format!("lightgbm-{}-py3-none-macosx_10_15_x86_64.whl", version),
                _ => return Err(format!("Unsupported macOS architecture: {}", arch).into()),
            };

            let download_url = format!(
                "https://files.pythonhosted.org/packages/py3/l/lightgbm/{}",
                wheel_name
            );

            println!(
                "cargo:warning=Downloading LightGBM wheel from: {}",
                download_url
            );

            // Download the wheel
            let response = ureq::get(&download_url).call()?;
            let status = response.status();
            if status < 200 || status >= 300 {
                return Err(format!("Failed to download wheel: HTTP {}", status).into());
            }

            // Save wheel to temp file
            let wheel_path = out_dir.join(&wheel_name);
            let mut wheel_file = fs::File::create(&wheel_path)?;
            io::copy(&mut response.into_reader(), &mut wheel_file)?;
            drop(wheel_file);

            println!("cargo:warning=✓ Downloaded and cached wheel");

            // Extract the dylib from the wheel
            println!("cargo:warning=Extracting library from wheel");

            let file = fs::File::open(&wheel_path)?;
            let mut archive = zip::ZipArchive::new(file)?;

            // Look for the dylib in the wheel
            let mut found = false;
            for i in 0..archive.len() {
                let mut file = archive.by_index(i)?;
                let name = file.name().to_string();

                if name.contains("lib_lightgbm") && name.ends_with(".dylib") {
                    println!("cargo:warning=Found library at: {}", name);

                    let lib_path = lib_dir.join("lib_lightgbm.dylib");
                    let mut outfile = fs::File::create(&lib_path)?;
                    io::copy(&mut file, &mut outfile)?;

                    println!(
                        "cargo:warning=✓ Successfully extracted LightGBM library to: {}",
                        lib_dir.display()
                    );
                    found = true;
                    break;
                }
            }

            if !found {
                return Err("Could not find lib_lightgbm.dylib in wheel".into());
            }
        },
        "linux" => {
            // Determine the wheel filename based on architecture
            let wheel_name = match arch.as_str() {
                "aarch64" => format!("lightgbm-{}-py3-none-manylinux2014_aarch64.whl", version),
                "x86_64" => format!("lightgbm-{}-py3-none-manylinux_2_28_x86_64.whl", version),
                _ => return Err(format!("Unsupported Linux architecture: {}", arch).into()),
            };

            let download_url = format!(
                "https://files.pythonhosted.org/packages/py3/l/lightgbm/{}",
                wheel_name
            );

            println!(
                "cargo:warning=Downloading LightGBM wheel from: {}",
                download_url
            );

            let response = ureq::get(&download_url).call()?;
            let status = response.status();
            if status < 200 || status >= 300 {
                return Err(format!("Failed to download wheel: HTTP {}", status).into());
            }

            let wheel_path = out_dir.join(&wheel_name);
            let mut wheel_file = fs::File::create(&wheel_path)?;
            io::copy(&mut response.into_reader(), &mut wheel_file)?;
            drop(wheel_file);

            println!("cargo:warning=✓ Downloaded and cached wheel");

            // Extract the .so from the wheel
            println!("cargo:warning=Extracting library from wheel");

            let file = fs::File::open(&wheel_path)?;
            let mut archive = zip::ZipArchive::new(file)?;

            let mut found = false;
            for i in 0..archive.len() {
                let mut file = archive.by_index(i)?;
                let name = file.name().to_string();

                if name.contains("lib_lightgbm") && name.ends_with(".so") {
                    println!("cargo:warning=Found library at: {}", name);

                    let lib_path = lib_dir.join("lib_lightgbm.so");
                    let mut outfile = fs::File::create(&lib_path)?;
                    io::copy(&mut file, &mut outfile)?;

                    println!(
                        "cargo:warning=✓ Successfully extracted LightGBM library to: {}",
                        lib_dir.display()
                    );
                    found = true;
                    break;
                }
            }

            if !found {
                return Err("Could not find lib_lightgbm.so in wheel".into());
            }
        },
        "windows" => {
            let wheel_name = format!("lightgbm-{}-py3-none-win_amd64.whl", version);
            let download_url = format!(
                "https://files.pythonhosted.org/packages/py3/l/lightgbm/{}",
                wheel_name
            );

            println!(
                "cargo:warning=Downloading LightGBM wheel from: {}",
                download_url
            );

            let response = ureq::get(&download_url).call()?;
            let status = response.status();
            if status < 200 || status >= 300 {
                return Err(format!("Failed to download wheel: HTTP {}", status).into());
            }

            let wheel_path = out_dir.join(&wheel_name);
            let mut wheel_file = fs::File::create(&wheel_path)?;
            io::copy(&mut response.into_reader(), &mut wheel_file)?;
            drop(wheel_file);

            // Extract the .dll from the wheel
            let file = fs::File::open(&wheel_path)?;
            let mut archive = zip::ZipArchive::new(file)?;

            let mut found = false;
            for i in 0..archive.len() {
                let mut file = archive.by_index(i)?;
                let name = file.name().to_string();

                if name.contains("lib_lightgbm") && name.ends_with(".dll") {
                    let lib_path = lib_dir.join("lib_lightgbm.dll");
                    let mut outfile = fs::File::create(&lib_path)?;
                    io::copy(&mut file, &mut outfile)?;

                    println!(
                        "cargo:warning=✓ Successfully extracted LightGBM library to: {}",
                        lib_dir.display()
                    );
                    found = true;
                    break;
                }
            }

            if !found {
                return Err("Could not find lib_lightgbm.dll in wheel".into());
            }
        },
        _ => return Err(format!("Unsupported platform: {}", os).into()),
    }

    Ok(())
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let lgbm_include_root = out_dir.join("include");

    // Download the headers
    if let Err(e) = download_lightgbm_headers(&out_dir) {
        eprintln!("Failed to download LightGBM headers: {}", e);
        panic!("Cannot proceed without headers");
    }

    // Download the compiled library
    if let Err(e) = download_compiled_library(&out_dir) {
        eprintln!("Failed to download compiled library: {}", e);
        panic!("Cannot proceed without compiled library");
    }

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", lgbm_include_root.display()))
        .clang_arg("-xc++")
        .clang_arg("-std=c++11")
        // Only generate bindings for functions starting with LGBM_
        .allowlist_function("LGBM_.*")
        // Allowlist the main types we need
        .allowlist_type("BoosterHandle")
        .allowlist_type("DatasetHandle")
        .allowlist_type("FastConfigHandle")
        .allowlist_type("ArrowArray")
        .allowlist_type("ArrowSchema")
        // Allowlist constants
        .allowlist_var("C_API_DTYPE_.*")
        // Treat Arrow types as opaque
        .opaque_type("ArrowArray")
        .opaque_type("ArrowSchema")
        // Block problematic C++ code from arrow.h
        .blocklist_type("std::.*")
        .blocklist_type("ArrowTable")
        .blocklist_type("ArrowChunkedArray")
        .blocklist_type(".*_Tp.*")
        .blocklist_type(".*_Pred.*")
        .size_t_is_usize(true)
        .rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate bindings.");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    // Get platform info using your existing function
    let (os, _arch) = get_platform_info();

    // Determine the library filename based on the OS
    let lib_filename = match os.as_str() {
        "windows" => "lib_lightgbm.dll",
        "darwin" => "lib_lightgbm.dylib",
        _ => "lib_lightgbm.so", // Default to Linux/Unix
    };

    // Copy the library from OUT_DIR/libs to the final target directory
    let lib_source_path = out_dir.join("libs").join(lib_filename);

    // Find the final output directory (e.g., target/release)
    let target_dir = out_dir
        .ancestors()
        .find(|p| p.ends_with("target"))
        .unwrap()
        .join(env::var("PROFILE").unwrap());

    let lib_dest_path = target_dir.join(lib_filename);
    fs::copy(&lib_source_path, &lib_dest_path)
        .expect("Failed to copy library to target directory");

    // Set the library search path for the build-time linker
    let lib_search_path = out_dir.join("libs");
    println!(
        "cargo:rustc-link-search=native={}",
        lib_search_path.display()
    );

    // Set the rpath for the run-time linker based on the OS
    match os.as_str() {
        "darwin" => {
            // For macOS, add multiple rpath entries for IDE compatibility
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../..");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_search_path.display());
            // Add the target directory to rpath as well
            if let Some(target_root) = out_dir.ancestors().find(|p| p.ends_with("target")) {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}/debug", target_root.display());
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}/release", target_root.display());
            }
        },
        "linux" => {
            // For Linux, use $ORIGIN
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../..");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_search_path.display());
        },
        _ => {} // No rpath needed for Windows
    }

    println!("cargo:rustc-link-lib=dylib=_lightgbm");
}
