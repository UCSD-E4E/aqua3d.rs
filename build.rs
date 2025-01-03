use std::env;
use std::fs;

use chrono::DateTime;
use chrono::Local;
use glob::glob;
use serde::{Deserialize, Serialize};


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

#[tokio::main]
async fn main() {
    const RELEASE_URL: &str = "https://api.github.com/repos/shader-slang/slang/releases";

    let client = reqwest::ClientBuilder::new()
        .user_agent("cargo")
        .build()
        .unwrap();
    
    let releases = client.get(RELEASE_URL)
        .send()
        .await.unwrap()
        .json::<Vec<Release>>()
        .await.unwrap();

    let mut final_releases: Vec<&Release> = releases
        .iter()
        .filter(|r| !(r.draft || r.prerelease))
        .collect();
    final_releases.sort_by(|a, b| a.published_at.cmp(&b.published_at));

    let current_release = *final_releases.last().expect("There should be more than one release");

    let operating_system = env::consts::OS;
    let arch = env::consts::ARCH;

    let platform_string = format!("{}-{}", operating_system, arch);
    let asset_name = format!("slang-{}-{}.tar.gz", &current_release.name[1..], platform_string);

    let platform_assets: Vec<&Asset> = current_release.assets.iter().filter(|a| a.name == asset_name).collect();
    let asset = *platform_assets.first().expect("There should be one matching asset");

    println!("cargo::warning={}", asset.browser_download_url);

    for entry in glob("./src/**/*.slang").expect("Failed to read glob pattern") {
        let path = entry.unwrap();
    }
}