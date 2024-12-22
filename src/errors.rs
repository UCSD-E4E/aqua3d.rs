use thiserror::Error;

#[derive(Error, Debug)]
pub enum Aqua3dError {
    #[error("A GPU could not be aquired.")]
    CannotAcquireGpu,
    #[error(transparent)]
    WgpuRequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error(transparent)]
    AnyhowError(#[from] anyhow::Error),
    #[error(transparent)]
    NdArrayShapeError(#[from] ndarray::ShapeError),
    #[error("An unknown error occurred")]
    UnknownError
}