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
var<storage, read_write> core_points: array<u32>;

@group(0)
@binding(3)
var<storage, read_write> y_pred: array<u32>;

fn calculate_distance(x1_idx: u32, x2_idx: u32) -> f32 {
    var sum: f32 = 0;

    var i: u32;
    for (i = u32(0); i < parameters.dim; i += u32(1)) {
        let idx1 = x1_idx * parameters.dim + i;
        let idx2 = x2_idx * parameters.dim + i;

        let diff = X[idx1] - X[idx2];

        sum += (diff * diff);
    }

    return sqrt(sum);
}

@compute @workgroup_size(64, 1, 1)
fn dbscan_preprocessing(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if (parameters.min_points > 2 && global_id.x < parameters.count) {
        var neighbor_count: u32 = 0;

        var i: u32;
        for (i = u32(0); i < parameters.count; i += u32(1)) {
            if (i != global_id.x && neighbor_count < parameters.min_points) {
                let dist = calculate_distance(global_id.x, i);

                if (dist <= parameters.epsilon) {
                    neighbor_count += u32(1);
                }
            }
        }

        core_points[global_id.x] = neighbor_count;
    }
}

@compute @workgroup_size(1, 1, 1)
fn dbscan_main() {
    let epsilon = parameters.epsilon;
    let a = X[0];
    let b = core_points[0];

    y_pred[0] = u32(parameters.count);
}