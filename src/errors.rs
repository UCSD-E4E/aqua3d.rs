use ndarray_stats::errors::MinMaxError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Aqua3dError {
    #[error(transparent)]
    NdArrayMinMaxError(#[from] MinMaxError)
}