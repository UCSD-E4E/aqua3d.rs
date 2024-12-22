// use std::collections::{HashMap, HashSet};

// use anyhow::Context;
// use bytemuck::{Pod, Zeroable};
// use ndarray::Array2;
// use thiserror::Error;
// use wgpu::{util::DeviceExt, Buffer, CommandEncoder, Device, Queue, ShaderModule};

// #[derive(Error, Debug)]
// pub enum SeaThruError {
//     #[error("A GPU could not be aquired.")]
//     CannotAcquireGpu,
//     #[error(transparent)]
//     WgpuRequestDeviceError(#[from] wgpu::RequestDeviceError),
//     #[error(transparent)]
//     AnyhowError(#[from] anyhow::Error),
//     #[error(transparent)]
//     NdArrayShapeError(#[from] ndarray::ShapeError),
//     #[error("An unknown error occurred")]
//     UnknownError
// }

// #[derive(Copy, Clone, Pod, Zeroable)]
// #[repr(C)]
// struct ImageSize {
//     pub width: u32,
//     pub height: u32
// }

// #[derive(Copy, Clone, Pod, Zeroable)]
// #[repr(C)]
// struct EstimateNeighborhoodMapParameters {
//     pub epsilon: f32
// }

// async fn get_device_and_queue() -> Result<(Device, Queue), SeaThruError> {
//     let instance = wgpu::Instance::default();

//     let adapter = instance
//         .request_adapter(&wgpu::RequestAdapterOptions::default())
//         .await;

//     let adapter = match adapter {
//         Some(adapter) => Ok(adapter),
//         None => Err(SeaThruError::CannotAcquireGpu)
//     }?;

//     let (device, queue) = adapter
//         .request_device(
//             &wgpu::DeviceDescriptor {
//                 label: None,
//                 required_features: wgpu::Features::empty(),
//                 required_limits: wgpu::Limits::downlevel_defaults(),
//                 memory_hints: wgpu::MemoryHints::MemoryUsage
//             },
//             None
//         )
//         .await?;

//     Ok((device, queue))
// }

// fn column_depth_segmentation(
//     image_size_buffer: &Buffer,
//     parameters_buffer: &Buffer,
//     depth_buffer: &Buffer,
//     neighborhood_map_gpu_buffer: &Buffer,
//     image_width: usize,
//     _image_height: usize,
//     shader_module: &ShaderModule,
//     encoder: &mut CommandEncoder,
//     device: &Device) {
//         let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//             label: None,
//             layout: None,
//             module: shader_module,
//             entry_point: Some("column_depth_segmentation"),
//             compilation_options: Default::default(),
//             cache: None
//         });

//         let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
//         let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
//             label: None,
//             layout: &bind_group_layout,
//             entries: &[
//                 wgpu::BindGroupEntry {
//                     binding: 0,
//                     resource: image_size_buffer.as_entire_binding()
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 1,
//                     resource: parameters_buffer.as_entire_binding()
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 2,
//                     resource: depth_buffer.as_entire_binding()
//                 },
//                 wgpu::BindGroupEntry {
//                     binding: 3,
//                     resource: neighborhood_map_gpu_buffer.as_entire_binding()
//                 }
//             ]
//         });

//         let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
//             label: None,
//             timestamp_writes: None
//         });
//         cpass.set_pipeline(&compute_pipeline);
//         cpass.set_bind_group(0, &bind_group, &[]);
//         cpass.insert_debug_marker("column_depth_segmentation");
//         cpass.dispatch_workgroups((image_width as f32 / 64f32).ceil() as u32, 1, 1);
// }

// // fn merge_column_depth_segmentation(
// //     image_size_buffer: &Buffer,
// //     parameters_buffer: &Buffer,
// //     depth_buffer: &Buffer,
// //     neighborhood_map_gpu_buffer: &Buffer,
// //     _image_width: usize,
// //     image_height: usize,
// //     shader_module: &ShaderModule,
// //     encoder: &mut CommandEncoder,
// //     device: &Device) {
// //     let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
// //         label: None,
// //         layout: None,
// //         module: shader_module,
// //         entry_point: Some("merge_column_depth_segmentation"),
// //         compilation_options: Default::default(),
// //         cache: None
// //     });

// //     let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
// //     let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
// //         label: None,
// //         layout: &bind_group_layout,
// //         entries: &[
// //             wgpu::BindGroupEntry {
// //                 binding: 0,
// //                 resource: image_size_buffer.as_entire_binding()
// //             },
// //             wgpu::BindGroupEntry {
// //                 binding: 1,
// //                 resource: parameters_buffer.as_entire_binding()
// //             },
// //             wgpu::BindGroupEntry {
// //                 binding: 2,
// //                 resource: depth_buffer.as_entire_binding()
// //             },
// //             wgpu::BindGroupEntry {
// //                 binding: 3,
// //                 resource: neighborhood_map_gpu_buffer.as_entire_binding()
// //             }
// //         ]
// //     });

// //     let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
// //         label: None,
// //         timestamp_writes: None
// //     });
// //     cpass.set_pipeline(&compute_pipeline);
// //     cpass.set_bind_group(0, &bind_group, &[]);
// //     cpass.insert_debug_marker("merge_column_depth_segmentation");
// //     cpass.dispatch_workgroups(1, (image_height as f32 / 64f32).ceil() as u32, 1);
// // }

// // fn merge_rows_depth_segmentation(
// //     image_size_buffer: &Buffer,
// //     parameters_buffer: &Buffer,
// //     depth_buffer: &Buffer,
// //     neighborhood_map_gpu_buffer: &Buffer,
// //     image_width: usize,
// //     _image_height: usize,
// //     shader_module: &ShaderModule,
// //     encoder: &mut CommandEncoder,
// //     device: &Device) {
// //     let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
// //         label: None,
// //         layout: None,
// //         module: shader_module,
// //         entry_point: Some("merge_rows_depth_segmentation"),
// //         compilation_options: Default::default(),
// //         cache: None
// //     });

// //     let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
// //     let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
// //         label: None,
// //         layout: &bind_group_layout,
// //         entries: &[
// //             wgpu::BindGroupEntry {
// //                 binding: 0,
// //                 resource: image_size_buffer.as_entire_binding()
// //             },
// //             wgpu::BindGroupEntry {
// //                 binding: 1,
// //                 resource: parameters_buffer.as_entire_binding()
// //             },
// //             wgpu::BindGroupEntry {
// //                 binding: 2,
// //                 resource: depth_buffer.as_entire_binding()
// //             },
// //             wgpu::BindGroupEntry {
// //                 binding: 3,
// //                 resource: neighborhood_map_gpu_buffer.as_entire_binding()
// //             }
// //         ]
// //     });

// //     let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
// //         label: None,
// //         timestamp_writes: None
// //     });
// //     cpass.set_pipeline(&compute_pipeline);
// //     cpass.set_bind_group(0, &bind_group, &[]);
// //     cpass.insert_debug_marker("merge_rows_depth_segmentation");
// //     cpass.dispatch_workgroups((image_width as f32 / 64f32).ceil() as u32, 1, 1);
// // }

// async fn depth_segmentation_gpu(depths: &Array2<f32>, epsilon: f32) -> Result<Array2<u32>, SeaThruError> {
//     let (height, width) = depths.dim();

//     let image_size = ImageSize {
//         width: width as u32,
//         height: height as u32
//     };
//     let image_size_bytes = bytemuck::bytes_of(&image_size);

//     let parameters = EstimateNeighborhoodMapParameters {
//         epsilon
//     };
//     let parameters_bytes = bytemuck::bytes_of(&parameters);

//     let neighborhood_map_size = (size_of::<u32>() * height * width) as u64;

//     let (device, queue) = get_device_and_queue().await?;
//     let shader_module = device.create_shader_module(wgpu::include_wgsl!("seathru.wgsl"));

//     let image_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//         label: Some("image_size_buffer"),
//         contents: image_size_bytes,
//         usage: wgpu::BufferUsages::STORAGE
//             | wgpu::BufferUsages::COPY_DST
//     });

//     let parameters_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//         label: Some("parameters_buffer"),
//         contents: parameters_bytes,
//         usage: wgpu::BufferUsages::STORAGE
//             | wgpu::BufferUsages::COPY_DST
//     });

//     let depth_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//         label: Some("depth_buffer"),
//         contents: bytemuck::cast_slice(depths.as_slice().context("Depths should have data.")?),
//         usage: wgpu::BufferUsages::STORAGE
//             | wgpu::BufferUsages::COPY_DST
//     });

//     let neighborhood_map_gpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("neighborhood_map_gpu_buffer"),
//         size: neighborhood_map_size,
//         usage: wgpu::BufferUsages::STORAGE
//             | wgpu::BufferUsages::COPY_SRC,
//         mapped_at_creation: false
//     });

//     let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
//         label: None
//     });

//     column_depth_segmentation(
//         &image_size_buffer,
//         &parameters_buffer,
//         &depth_buffer,
//         &neighborhood_map_gpu_buffer,
//         width,
//         height,
//         &shader_module,
//         &mut encoder,
//         &device);

//     // merge_column_depth_segmentation(
//     //     &image_size_buffer,
//     //     &parameters_buffer,
//     //     &depth_buffer,
//     //     &neighborhood_map_gpu_buffer,
//     //     width,
//     //     height,
//     //     &shader_module,
//     //     &mut encoder,
//     //     &device);

//     // merge_rows_depth_segmentation(
//     //     &image_size_buffer,
//     //     &parameters_buffer,
//     //     &depth_buffer,
//     //     &neighborhood_map_gpu_buffer,
//     //     width,
//     //     height,
//     //     &shader_module,
//     //     &mut encoder,
//     //     &device);

//     let neighborhood_map_cpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
//         label: Some("neighborhood_map_cpu_buffer"),
//         size: neighborhood_map_size,
//         usage: wgpu::BufferUsages::COPY_DST
//             | wgpu::BufferUsages::MAP_READ,
//         mapped_at_creation: false
//     });

//     encoder.copy_buffer_to_buffer(&neighborhood_map_gpu_buffer, 0, &neighborhood_map_cpu_buffer, 0, neighborhood_map_size);

//     queue.submit(Some(encoder.finish()));

//     let neighborhood_map_slice = neighborhood_map_cpu_buffer.slice(..);

//     let (sender, receiver) = flume::bounded(1);
//     neighborhood_map_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

//     device.poll(wgpu::Maintain::wait()).panic_on_timeout();

//     if let Ok(Ok(())) = receiver.recv_async().await {
//         let neighborhood_map_bytes = neighborhood_map_slice.get_mapped_range();
//         let neighborhood_map_array: Vec<u32> = bytemuck::cast_slice(&neighborhood_map_bytes).to_vec();

//         return Ok(Array2::from_shape_vec((height, width), neighborhood_map_array)?);
//     }

//     Err(SeaThruError::UnknownError)
// }

// fn cleanup_depth_segmentations(depth_segmentations: &Array2<u32>) -> Array2<u32> {
//     let range: HashSet<u32> = depth_segmentations.iter().map(|x| *x).collect();
//     let map: HashMap<u32, u32> = range.iter().enumerate().map(|(idx, x)| (*x, idx as u32)).collect();

//     depth_segmentations.mapv(|x| map[&x])
// }

// pub async fn estimate_neighborhood_map(depths: &Array2<f32>, epsilon: f32) -> Result<Array2<u32>, SeaThruError> {
//     let gpu_depth_segmentations = depth_segmentation_gpu(depths, epsilon).await?;

//     Ok(cleanup_depth_segmentations(&gpu_depth_segmentations))
// }

// // #[cfg(test)]
// // mod tests {
// //     use std::fs::File;

// //     use ndarray::Array2;
// //     use ndarray_npy::{NpzReader, NpzWriter};

// //     use super::estimate_neighborhood_map;

// //     #[tokio::test]
// //     async fn estimate_neighborhood_map_test() {
// //         let mut npz = NpzReader::new(File::open("./data/seathru/D3/D3/depth/depthT_S04923.npz").unwrap()).unwrap();
// //         let depths: Array2<f32> = npz.by_name("depths").unwrap();

// //         let neighborhood_map = estimate_neighborhood_map(&depths, 0.21).await.unwrap();
        
// //         let mut npz_writer = NpzWriter::new(File::create("../pySeaThruDev/neighborhood_map.npz").unwrap());
// //         npz_writer.add_array("neighborhood_map", &neighborhood_map).unwrap();
// //     }
// // }