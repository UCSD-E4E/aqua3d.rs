use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use ndarray::Array2;
use wgpu::{util::DeviceExt, Buffer, CommandEncoder, Device, ShaderModule};

use crate::{clustering::{dbscan_main, dbscan_preprocessing, DbScanParameters}, errors::Aqua3dError, gpu::get_device_and_queue};

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct NeighborhoodmapParameters {
    pub height: u32,
    pub width: u32
}

fn seathru_estimate_neighborhoodmap_preprocessing(
    neighborhoodmap_parameters_buffer: &Buffer,
    dbscan_parameters_buffer: &Buffer,
    x_buffer: &Buffer,
    depth_buffer: &Buffer,
    width: usize,
    height: usize,
    shader_module: &ShaderModule,
    encoder: &mut CommandEncoder,
    device: &Device
) {
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader_module,
        entry_point: Some("seathru_estimate_neighborhoodmap_preprocessing"),
        compilation_options: Default::default(),
        cache: None
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dbscan_parameters_buffer.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buffer.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: neighborhoodmap_parameters_buffer.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: depth_buffer.as_entire_binding()
            }
        ]
    });

    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None
    });
    cpass.set_pipeline(&compute_pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    cpass.insert_debug_marker("seathru_estimate_neighborhoodmap_preprocessing");
    cpass.dispatch_workgroups(
        (width as f32 / 16f32).ceil() as u32,
        (height as f32 / 16f32).ceil() as u32,
        1);
}

fn seathru_estimate_neighborhoodmap_postprocessing(
    neighborhoodmap_parameters_buffer: &Buffer,
    y_pred_buffer: &Buffer,
    neighborhoodmap_buffer: &Buffer,
    width: usize,
    height: usize,
    shader_module: &ShaderModule,
    encoder: &mut CommandEncoder,
    device: &Device
) {
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader_module,
        entry_point: Some("seathru_estimate_neighborhoodmap_postprocessing"),
        compilation_options: Default::default(),
        cache: None
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y_pred_buffer.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: neighborhoodmap_parameters_buffer.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: neighborhoodmap_buffer.as_entire_binding()
            }
        ]
    });

    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None
    });
    cpass.set_pipeline(&compute_pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    cpass.insert_debug_marker("seathru_estimate_neighborhoodmap_postprocessing");
    cpass.dispatch_workgroups(
        (width as f32 / 16f32).ceil() as u32,
        (height as f32 / 16f32).ceil() as u32,
        1);
}

pub async fn estimate_neighborhood_map(depths: &Array2<f32>, epsilon: f32) -> Result<Array2<u32>, Aqua3dError> {
    let (height, width) = depths.dim();
    let count = height * width;
    let x_size = (count * 3 * size_of::<u32>()) as u64;
    let core_points_size = (count * size_of::<u32>()) as u64;
    let y_pred_size = (count * size_of::<u32>()) as u64;
    let neighborhoodmap_size = (count * size_of::<u32>()) as u64;

    let neighborhoodmap_parameters = NeighborhoodmapParameters {
        height: height as u32,
        width: width as u32
    };
    let neighborhoodmap_parameters_bytes = bytemuck::bytes_of(&neighborhoodmap_parameters);

    let dbscan_parameters = DbScanParameters {
        count: count as u32,
        dim: 3,
        epsilon,
        min_points: 2
    };
    let dbscan_parameters_bytes = bytemuck::bytes_of(&dbscan_parameters);

    let (device, queue) = get_device_and_queue().await?;
    let shader_module = device.create_shader_module(wgpu::include_wgsl!("seathru.wgsl"));

    let neighborhoodmap_parameters_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("neighborhoodmap_parameters_buffer"),
        contents: neighborhoodmap_parameters_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::UNIFORM
    });

    let dbscan_parameters_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dbscan_parameters_buffer"),
        contents: dbscan_parameters_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::UNIFORM
    });

    let x_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("x_buffer"),
        size: x_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false
    });

    let core_points_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("core_points_buffer"),
        size: core_points_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false
    });

    let y_pred_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("y_pred_buffer"),
        size: y_pred_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false
    });

    let depth_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("depth_buffer"),
        contents: bytemuck::cast_slice(depths.as_slice().context("Depths should have data.")?),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
    });

    let neighborhoodmap_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neighborhoodmap_buffer"),
        size: neighborhoodmap_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: None
    });

    seathru_estimate_neighborhoodmap_preprocessing(
        &neighborhoodmap_parameters_buffer,
        &dbscan_parameters_buffer,
        &x_buffer,
        &depth_buffer,
        width,
        height,
        &shader_module,
        &mut encoder,
        &device);

    dbscan_preprocessing(
        &dbscan_parameters_buffer, 
        &x_buffer, 
        &core_points_buffer, 
        &y_pred_buffer, 
        count as u32, 
        &shader_module, 
        &mut encoder,
        &device);

    dbscan_main(
        &dbscan_parameters_buffer,
        &x_buffer,
        &core_points_buffer,
        &y_pred_buffer,
        count as u32,
        &shader_module,
        &mut encoder,
        &device);

    seathru_estimate_neighborhoodmap_postprocessing(
        &neighborhoodmap_parameters_buffer,
        &y_pred_buffer,
        &neighborhoodmap_buffer,
        width,
        height,
        &shader_module,
        &mut encoder,
        &device);

    let neighborhoodmap_cpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neighborhoodmap_cpu_buffer"),
        size: neighborhoodmap_size,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false
    });

    encoder.copy_buffer_to_buffer(&neighborhoodmap_buffer, 0, &neighborhoodmap_cpu_buffer, 0, neighborhoodmap_size);

    queue.submit(Some(encoder.finish()));

    let neighborhoodmap_slice = neighborhoodmap_cpu_buffer.slice(..);

    let (sender, receiver) = flume::bounded(1);
    neighborhoodmap_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    if let Ok(Ok(())) = receiver.recv_async().await {
        let neighborhoodmap_bytes = neighborhoodmap_slice.get_mapped_range();
        let neighborhoodmap_array: Vec<u32> = bytemuck::cast_slice(&neighborhoodmap_bytes).to_vec();

        return Ok(Array2::from_shape_vec((height, width), neighborhoodmap_array)?);
    }

    Err(Aqua3dError::UnknownError)
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use ndarray::Array2;
    use ndarray_npy::{NpzReader, NpzWriter};

    use super::estimate_neighborhood_map;

    #[tokio::test]
    async fn estimate_neighborhood_map_test() {
        let mut npz = NpzReader::new(File::open("./data/seathru/D3/D3/depth/depthT_S04923.npz").unwrap()).unwrap();
        let depths: Array2<f32> = npz.by_name("depths").unwrap();

        let neighborhood_map = estimate_neighborhood_map(&depths, 0.21).await.unwrap();
        
        let mut npz_writer = NpzWriter::new(File::create("../pyAqua3dDev/neighborhood_map.npz").unwrap());
        npz_writer.add_array("neighborhood_map", &neighborhood_map).unwrap();
    }
}