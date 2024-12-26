// Implementation of
// https://doi.org/10.1145/3605573.3605594

struct Parameters {
    count: u32,
    dim: u32,
    epsilon: f32,
    min_points: u32
}

@group(0)
@binding(0)
var<storage, read> parameters: Parameters;

@group(0)
@binding(1)
var<storage, read> X: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> y_pred: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn dbscan_preprocessing() {
    let epsilon = parameters.epsilon;
    let a = X[0];
}

@compute @workgroup_size(1, 1, 1)
fn dbscan_main() {
    let epsilon = parameters.epsilon;
    let a = X[0];
    
    y_pred[0] = u32(parameters.count);
}