use wgpu::{Device, Queue};

use crate::errors::Aqua3dError;

pub async fn get_device_and_queue() -> Result<(Device, Queue), Aqua3dError> {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await;

    let adapter = match adapter {
        Some(adapter) => Ok(adapter),
        None => Err(Aqua3dError::CannotAcquireGpu)
    }?;

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