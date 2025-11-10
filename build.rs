extern crate bindgen;

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

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
    if !(200..300).contains(&status) {
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
    if !(200..300).contains(&status) {
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

    println!(
        "cargo:warning=Attempting to download arrow.h from: {}",
        arrow_url
    );

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

            println!(
                "cargo:warning=Attempting to download arrow.tpp from: {}",
                arrow_tpp_url
            );

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
            println!(
                "cargo:warning=arrow.h not available for this version (optional, only in v4.2.0+)"
            );
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

    // For macOS and Linux, extract from Python wheel to get architecture-specific binaries
    match (os.as_str(), arch.as_str()) {
        // macOS - both x86_64 and ARM64 available
        ("darwin", "aarch64") | ("darwin", "x86_64") => {
            let wheel_arch = if arch == "aarch64" { "arm64" } else { "x86_64" };
            let macos_version = if arch == "aarch64" { "12_0" } else { "10_15" };
            let wheel_url = format!(
                "https://github.com/microsoft/LightGBM/releases/download/v{}/lightgbm-{}-py3-none-macosx_{}_{}.whl",
                version, version, macos_version, wheel_arch
            );

            println!(
                "cargo:warning=Downloading LightGBM v{} macOS {} wheel from: {}",
                version, wheel_arch, wheel_url
            );

            download_and_extract_from_wheel(&wheel_url, out_dir, &lib_dir, "lib_lightgbm.dylib")?;
        }

        // Linux - both x86_64 and ARM64 available
        ("linux", "aarch64") | ("linux", "x86_64") => {
            let (wheel_platform, lib_pattern) = if arch == "aarch64" {
                ("manylinux2014_aarch64", "lib_lightgbm.so")
            } else {
                ("manylinux_2_28_x86_64", "lib_lightgbm.so")
            };

            let wheel_url = format!(
                "https://github.com/microsoft/LightGBM/releases/download/v{}/lightgbm-{}-py3-none-{}.whl",
                version, version, wheel_platform
            );

            println!(
                "cargo:warning=Downloading LightGBM v{} Linux {} wheel from: {}",
                version, arch, wheel_url
            );

            download_and_extract_from_wheel(&wheel_url, out_dir, &lib_dir, lib_pattern)?;
        }

        // Windows - only x86_64 available
        ("windows", "x86_64") => {
            // For Windows, extract from wheel - need both DLL and import library
            let wheel_url = format!(
                "https://github.com/microsoft/LightGBM/releases/download/v{}/lightgbm-{}-py3-none-win_amd64.whl",
                version, version
            );

            println!(
                "cargo:warning=Downloading LightGBM v{} Windows x86_64 wheel from: {}",
                version, wheel_url
            );

            download_and_extract_windows_libs(&wheel_url, out_dir, &lib_dir)?;
        }

        ("windows", "i686") => {
            return Err("Windows 32-bit (i686) is not supported by LightGBM releases. Please use x86_64 Windows or compile LightGBM from source.".into());
        }

        ("windows", "aarch64") => {
            return Err("Windows ARM64 is not currently supported by LightGBM releases. Please use x86_64 Windows or compile LightGBM from source.".into());
        }

        _ => {
            return Err(format!(
                "Unsupported platform/architecture combination: {} / {}",
                os, arch
            )
            .into());
        }
    }

    Ok(())
}

fn download_and_extract_from_wheel(
    wheel_url: &str,
    out_dir: &Path,
    lib_dir: &Path,
    lib_filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Download the wheel to a temp file
    let wheel_path = out_dir.join("lightgbm.whl");
    let mut dest = fs::File::create(&wheel_path)?;

    let response = ureq::get(wheel_url).call()?;
    let status = response.status();
    if !(200..300).contains(&status) {
        return Err(format!("Failed to download wheel: HTTP {}", status).into());
    }

    io::copy(&mut response.into_reader(), &mut dest)?;
    drop(dest); // Close file before reading

    // Extract the library from the wheel
    // Wheels are just zip files
    let wheel_file = fs::File::open(&wheel_path)?;
    let mut archive = zip::ZipArchive::new(wheel_file)?;

    // Find and extract the library
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if file.name().ends_with(lib_filename) {
            let lib_path = lib_dir.join(lib_filename);
            let mut outfile = fs::File::create(&lib_path)?;
            io::copy(&mut file, &mut outfile)?;

            println!(
                "cargo:warning=Extracted LightGBM library to: {}",
                lib_path.display()
            );
            return Ok(());
        }
    }

    Err(format!("{} not found in wheel", lib_filename).into())
}

fn download_and_extract_windows_libs(
    wheel_url: &str,
    out_dir: &Path,
    lib_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Download the wheel to a temp file
    let wheel_path = out_dir.join("lightgbm.whl");
    let mut dest = fs::File::create(&wheel_path)?;

    let response = ureq::get(wheel_url).call()?;
    let status = response.status();
    if !(200..300).contains(&status) {
        return Err(format!("Failed to download wheel: HTTP {}", status).into());
    }

    io::copy(&mut response.into_reader(), &mut dest)?;
    drop(dest); // Close file before reading

    // Extract both the DLL and the import library from the wheel
    // Wheels are just zip files
    let wheel_file = fs::File::open(&wheel_path)?;
    let mut archive = zip::ZipArchive::new(wheel_file)?;

    let mut dll_found = false;
    let mut lib_found = false;

    // Find and extract both lib_lightgbm.dll and lib_lightgbm.lib
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let filename = file.name();

        if filename.ends_with("lib_lightgbm.dll") {
            let lib_path = lib_dir.join("lib_lightgbm.dll");
            let mut outfile = fs::File::create(&lib_path)?;
            io::copy(&mut file, &mut outfile)?;
            println!(
                "cargo:warning=Extracted LightGBM DLL to: {}",
                lib_path.display()
            );
            dll_found = true;
        } else if filename.ends_with("lib_lightgbm.lib") {
            let lib_path = lib_dir.join("lib_lightgbm.lib");
            let mut outfile = fs::File::create(&lib_path)?;
            io::copy(&mut file, &mut outfile)?;
            println!(
                "cargo:warning=Extracted LightGBM import library to: {}",
                lib_path.display()
            );
            lib_found = true;
        }

        if dll_found && lib_found {
            return Ok(());
        }
    }

    if !dll_found {
        return Err("lib_lightgbm.dll not found in wheel".into());
    }
    if !lib_found {
        return Err("lib_lightgbm.lib not found in wheel".into());
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
        .clang_arg("-std=c++14")
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
    fs::copy(&lib_source_path, &lib_dest_path).expect("Failed to copy library to target directory");

    // On Windows, also copy the import library (.lib) to the libs directory for linking
    if os == "windows" {
        let import_lib_source = out_dir.join("libs").join("lib_lightgbm.lib");
        if import_lib_source.exists() {
            // No need to copy the .lib to target dir, it's only used during linking
            println!(
                "cargo:warning=Found import library at: {}",
                import_lib_source.display()
            );
        }
    }

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
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                lib_search_path.display()
            );
            // Add the target directory to rpath as well
            if let Some(target_root) = out_dir.ancestors().find(|p| p.ends_with("target")) {
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath,{}/debug",
                    target_root.display()
                );
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath,{}/release",
                    target_root.display()
                );
            }
            println!("cargo:rustc-link-lib=dylib=_lightgbm");
        }
        "linux" => {
            // For Linux, use $ORIGIN
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../..");
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                lib_search_path.display()
            );
            println!("cargo:rustc-link-lib=dylib=_lightgbm");
        }
        "windows" => {
            // On Windows, we need to tell the linker where to find the DLL at runtime
            // Copy the DLL to the output directory (already done above)
            println!("cargo:rustc-link-lib=dylib=lib_lightgbm");
        }
        _ => {}
    }
}
