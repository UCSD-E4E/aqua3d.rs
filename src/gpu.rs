use wgpu::{Device, PowerPreference, Queue};

use crate::errors::Aqua3dError;

pub async fn get_device_and_queue() -> Result<(Device, Queue), Aqua3dError> {
    let instance = wgpu::Instance::default();

    let mut request_options = wgpu::RequestAdapterOptions::default();
    request_options.power_preference = PowerPreference::HighPerformance;

    let adapter = instance
        .request_adapter(&request_options)
        .await;

    let adapter = match adapter {
        Some(adapter) => Ok(adapter),
        None => Err(Aqua3dError::CannotAcquireGpu)
    }?;

    println!("adapter={}", adapter.get_info().name);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage
            },
            None
        )
        .await?;

    Ok((device, queue))
}