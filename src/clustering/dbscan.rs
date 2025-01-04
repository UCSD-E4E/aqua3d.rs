use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use ndarray::{Array1, Array2};
use wgpu::{util::DeviceExt, Buffer, CommandEncoder, Device, Queue, ShaderModule};

use crate::errors::Aqua3dError;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct Parameters {
    pub count: u32,
    pub dim: u32,
    pub epsilon: f32,
    pub min_points: u32
}

async fn get_device_and_queue() -> Result<(Device, Queue), Aqua3dError> {
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

fn dbscan_preprocessing(
    parameters_buffer: &Buffer,
    x_buffer: &Buffer,
    core_points_buffer: &Buffer,
    tree_buffer: &Buffer,
    count: u32,
    shader_module: &ShaderModule,
    encoder: &mut CommandEncoder,
    device: &Device
) {
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader_module,
        entry_point: Some("dbscan_preprocessing"),
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
                resource: parameters_buffer.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buffer.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: core_points_buffer.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: tree_buffer.as_entire_binding()
            }
        ]
    });

    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None
    });
    cpass.set_pipeline(&compute_pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    cpass.insert_debug_marker("dbscan_preprocessing");
    cpass.dispatch_workgroups((count as f32 / 64f32).ceil() as u32, 1, 1);
}

fn dbscan_main(
    parameters_buffer: &Buffer,
    x_buffer: &Buffer,
    core_points_buffer: &Buffer,
    tree_buffer: &Buffer,
    count: u32,
    shader_module: &ShaderModule,
    encoder: &mut CommandEncoder,
    device: &Device
) {
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader_module,
        entry_point: Some("dbscan_main"),
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
                resource: parameters_buffer.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: x_buffer.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: core_points_buffer.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: tree_buffer.as_entire_binding()
            }
        ]
    });

    let workgroup_count = (count as f32 / 16f32).ceil() as u32;

    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None
    });
    cpass.set_pipeline(&compute_pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    cpass.insert_debug_marker("dbscan_main");
    cpass.dispatch_workgroups(
        workgroup_count, 
        workgroup_count, 
        1);
}

fn dbscan_postprocessing(
    tree_buffer: &Buffer,
    y_pred_buffer: &Buffer,
    count: u32,
    shader_module: &ShaderModule,
    encoder: &mut CommandEncoder,
    device: &Device
) {
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader_module,
        entry_point: Some("dbscan_postprocessing"),
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
                resource: tree_buffer.as_entire_binding()
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: y_pred_buffer.as_entire_binding()
            }
        ]
    });

    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None
    });
    cpass.set_pipeline(&compute_pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    cpass.insert_debug_marker("dbscan_main");
    cpass.dispatch_workgroups((count as f32 / 64f32).ceil() as u32, 1, 1);
}

pub async fn dbscan(x: &Array2<f32>, epsilon: f32, min_points: u32) -> Result<Array1<u32>, Aqua3dError> {
    let (count, dim) = x.dim();
    let core_points_size = (count * size_of::<u32>()) as u64;
    let y_pred_size = (count * size_of::<u32>()) as u64;

    let parameters = Parameters {
        count: count as u32,
        dim: dim as u32,
        epsilon,
        min_points
    };
    let parameters_bytes = bytemuck::bytes_of(&parameters);

    let (device, queue) = get_device_and_queue().await?;
    let shader_module = device.create_shader_module(wgpu::include_spirv!("dbscan.spv"));

    let parameters_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("parameters_buffer"),
        contents: parameters_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::UNIFORM
    });

    let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("X"),
        contents: bytemuck::cast_slice(x.as_slice().context("Depths should have data.")?),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
    });

    let core_points_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("core_points_buffer"),
        size: core_points_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false
    });

    let tree_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tree_buffer"),
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: None
    });

    dbscan_preprocessing(
        &parameters_buffer,
        &x_buffer,
        &core_points_buffer,
        &tree_buffer,
        count as u32,
        &shader_module,
        &mut encoder,
        &device);

    dbscan_main(
        &parameters_buffer,
        &x_buffer,
        &core_points_buffer,
        &tree_buffer,
        count as u32,
        &shader_module,
        &mut encoder,
        &device);

    dbscan_postprocessing(
        &tree_buffer,
        &y_pred_buffer,
        count as u32,
        &shader_module,
        &mut encoder,
        &device
    );

    let y_pred_cpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("y_pred_cpu_buffer"),
        size: y_pred_size,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false
    });
    encoder.copy_buffer_to_buffer(&y_pred_buffer, 0, &y_pred_cpu_buffer, 0, y_pred_size);

    queue.submit(Some(encoder.finish()));

    let y_pred_slice = y_pred_cpu_buffer.slice(..);

    let (sender, receiver) = flume::bounded(1);
    y_pred_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    if let Ok(Ok(())) = receiver.recv_async().await {
        let y_pred_bytes = y_pred_slice.get_mapped_range();
        let y_pred_array: Vec<u32> = bytemuck::cast_slice(&y_pred_bytes).to_vec();

        return Ok(Array1::from_shape_vec((count,), y_pred_array)?);
    }

    Err(Aqua3dError::UnknownError)?
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use ndarray::Array2;
    use ndarray_npy::{NpzReader, NpzWriter};

    use crate::clustering::dbscan::dbscan;

    #[tokio::test]
    async fn dbscan_test() {
        let epsilon = 0.3f32;
        let min_points: u32 = 5;

        let mut npz = NpzReader::new(File::open(format!("./data/clusters/2D/{}.npz", "noisy_circles")).unwrap()).unwrap();
        let x: Array2<f64> = npz.by_name("X").unwrap();
        let x_f32 = x.mapv(|x| x as f32);
        // let truth: Array1<i64> = npz.by_name("dbscan").unwrap();
        // let truth_u32 = truth.mapv(|x| x as u32);

        let y_pred = dbscan(&x_f32, epsilon, min_points).await.unwrap();

        let mut npz_writer = NpzWriter::new(File::create("./data/y_pred.npz").unwrap());
        npz_writer.add_array("y_pred", &y_pred).unwrap();

        //assert_eq!(y_pred == truth_u32, true);
    }
}