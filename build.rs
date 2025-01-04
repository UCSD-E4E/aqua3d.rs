use std::env;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use anyhow::Result;
use chrono::DateTime;
use chrono::Local;
use flate2::read::GzDecoder;
use futures::StreamExt;
use glob::glob;
use serde::{Deserialize, Serialize};
use tar::Archive;
use tokio::{fs, io};

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Asset {
    browser_download_url: String,
    name: String
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Release {
    name: String,
    draft: bool,
    prerelease: bool,
    published_at: DateTime<Local>,
    assets: Vec<Asset>
}

fn get_tar_path() -> PathBuf {
    let mut target_file_path = PathBuf::new();
    target_file_path.push(".");
    target_file_path.push("target");
    target_file_path.push("tools");
    target_file_path.push("slang");
    target_file_path.push("slang.tar.gz");

    target_file_path
}

fn get_slangc_path() -> PathBuf {
    let mut target_file_path = PathBuf::new();
    target_file_path.push(".");
    target_file_path.push("target");
    target_file_path.push("tools");
    target_file_path.push("slang");
    target_file_path.push("bin");
    target_file_path.push("slangc");

    target_file_path
}

async fn get_asset_url() -> Result<String> {
    const RELEASE_URL: &str = "https://api.github.com/repos/shader-slang/slang/releases";

    let client = reqwest::ClientBuilder::new()
        .user_agent("cargo")
        .build()?;
    
    let releases = client.get(RELEASE_URL)
        .send()
        .await?
        .json::<Vec<Release>>()
        .await?;

    let mut final_releases: Vec<&Release> = releases
        .iter()
        .filter(|r| !(r.draft || r.prerelease))
        .collect();
    final_releases.sort_by(|a, b| a.published_at.cmp(&b.published_at));

    let current_release = *final_releases.last().context("There should be more than one release")?;

    let operating_system = env::consts::OS;
    let arch = env::consts::ARCH;

    let platform_string = format!("{}-{}", operating_system, arch);
    let asset_name = format!("slang-{}-{}.tar.gz", &current_release.name[1..], platform_string);

    let platform_assets: Vec<&Asset> = current_release.assets.iter().filter(|a| a.name == asset_name).collect();
    let asset = *platform_assets.first().context("There should be one matching asset")?;

    Ok(asset.browser_download_url.clone())
}

async fn download_slang_tarball() -> Result<()> {
    let target_file_path = get_tar_path();

    if target_file_path.exists() { // Early escape if we have already downloaded the tools
        return Ok(());
    }

    fs::create_dir_all(target_file_path.parent().context("The target file path has a parent")?).await?;

    let download_url = get_asset_url().await?;
    let mut byte_stream = reqwest::get(download_url).await?.bytes_stream();
    let mut file = fs::File::create(target_file_path).await?;
    while let Some(item) = byte_stream.next().await {
        io::copy(&mut item?.as_ref(), &mut file).await?;
    }

    Ok(())
}

fn unpack_slang_tarball() -> Result<()> {
    let slangc_path = get_slangc_path();
    if slangc_path.exists() { // Early escape if we have already unpacked the tools
        return Ok(());
    }

    let target_file_path = get_tar_path();
    let tar_gz = std::fs::File::open(&target_file_path)?;
    let tar = GzDecoder::new(tar_gz);

    let mut archive = Archive::new(tar);
    archive.unpack(target_file_path.parent().context("Target path should have parent.")?)?;

    Ok(())
}

fn compile_files() -> Result<()> {
    let slangc_path = get_slangc_path();

    for entry in glob("./src/**/*.slang").context("Failed to read glob pattern")? {
        let source_path = entry?;
        let mut target_path = PathBuf::new();
        target_path.push(source_path.parent().context("Source path does not have parent")?);
        target_path.push(format!("{}.spv", source_path.file_stem().context("Should have file stem")?.to_string_lossy()));

        let mut compiler_command = Command::new(&slangc_path);
        compiler_command.arg("-fvk-use-entrypoint-name").arg(&source_path).arg("-o").arg(target_path);

        let output = compiler_command.output().expect("compiler failed to execute.");
        if !output.status.success() {
            panic!("slangc returned an error.");
        }

        println!("cargo::rerun-if-changed={}", source_path.as_os_str().to_string_lossy());
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    download_slang_tarball().await?;
    unpack_slang_tarball()?;
    compile_files()?;

    Ok(())
}