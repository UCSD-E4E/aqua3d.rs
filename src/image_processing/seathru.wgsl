// =============================== Start Copied from DBScan - DO NOT EDIT ======================================
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
var<storage, read_write> X: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> core_points: array<u32>;

@group(0)
@binding(3)
var<storage, read_write> y_pred: array<atomic<u32>>;

fn get_X_value(idx: u32, dim: u32) -> f32 { 
    let target_idx = idx * parameters.dim + dim;

    return X[target_idx];
}

fn is_core_point(idx: u32) -> bool {
    return core_points[idx] != 0;
}

fn union_trees(x_idx: u32, y_idx: u32) {
    var x_root = find_root(x_idx);
    
    if (x_idx > y_idx) {
        var y_root = find_root(y_idx);
        var repeat = false;

        loop {
            repeat = false;

            if (x_root != y_root) {
                var ret: u32;
                if (x_root < y_root) {
                    ret = atomicCompareExchangeWeak(&y_pred[y_root], y_root, x_root).old_value;
                    if (ret != y_root) {
                        y_root = ret;
                        repeat = true;
                    }
                }
                else {
                    ret = atomicCompareExchangeWeak(&y_pred[x_root], x_root, y_root).old_value;
                    if (ret != x_root) {
                        x_root = ret;
                        repeat = true;
                    }
                }
            }

            if (!repeat) {
                break;
            }
        }
    }
}

fn find_root(idx: u32) -> u32 {
    var curr = y_pred[idx];
    if (curr != idx) {
        var next = idx;
        var prev = idx;
        
        while (curr > y_pred[curr]) {
            next = y_pred[curr];
            y_pred[prev] = next;
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

fn is_member_of_any_cluster(x_idx: u32) -> bool {
    return y_pred[x_idx] != x_idx;
}

@compute @workgroup_size(64, 1, 1)
fn dbscan_preprocessing(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if (global_id.x < parameters.count) {
        var neighbor_count: u32 = 0;

        if (parameters.min_points > 2 ) {
            var i: u32;
            for (i = u32(0); i < parameters.count; i += u32(1)) {
                if (i != global_id.x && neighbor_count < parameters.min_points) {
                    let dist = calculate_distance(global_id.x, i);

                    if (dist <= parameters.epsilon) {
                        neighbor_count += u32(1);
                    }
                }
            }
        }
        else {
            // When min_points == 2, all points are core points if they are within epsilon distance.
            neighbor_count = parameters.min_points;
        }

        if (neighbor_count >= parameters.min_points) {
            core_points[global_id.x] = neighbor_count;
        }
        else {
            core_points[global_id.x] = u32(0);
        }

        y_pred[global_id.x] = global_id.x; // Initialize the y_pred.
    }
}

@compute @workgroup_size(64, 1, 1)
fn dbscan_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    var y: u32;
    for (y = u32(0); y < parameters.count; y += u32(1)) {
        if (global_id.x < parameters.count &&
            global_id.x != global_id.y &&
            is_core_point(global_id.x) &&
            calculate_distance(global_id.x, y) < parameters.epsilon) {

                if (is_core_point(y) || !is_member_of_any_cluster(y)) {
                    union_trees(global_id.x, y);
                }
        }
    }
}


// =============================== End Copied from DBScan - DO NOT EDIT ========================================

fn calculate_distance(x_idx: u32, y_idx: u32) -> f32 {
    const inf: f32 = 0x1.fffffep+127f;

    let diff1 = u32(get_X_value(x_idx, u32(0))) - u32(get_X_value(y_idx, u32(0)));
    let diff2 = u32(get_X_value(x_idx, u32(1))) - u32(get_X_value(y_idx, u32(1)));
    if (sqrt(f32((diff1 * diff1) + (diff2 * diff2))) > 1) {
        return inf;
    }
    
    let diff = get_X_value(x_idx, u32(2)) - get_X_value(y_idx, u32(2));
    return sqrt(diff * diff);
}

struct NeighborhoodmapParameters {
    height: u32,
    width: u32
}

@group(0)
@binding(4)
var<storage, read> neighborhoodmap_parameters: NeighborhoodmapParameters;

@group(0)
@binding(5)
var<storage, read> depth_map: array<f32>;

@group(0)
@binding(6)
var<storage, read_write> neighborhood_map: array<u32>;

fn get_depth_value(x: u32, y: u32) -> f32 {
    let idx = y * neighborhoodmap_parameters.width + x;

    return depth_map[idx];
}

fn set_X_value(idx: u32, dim: u32, value: f32) { 
    let target_idx = idx * parameters.dim + dim;
    X[target_idx] = value;
}

fn set_neighborhood_value(x: u32, y: u32, value: u32) {
    let idx = y * neighborhoodmap_parameters.width + x;
    neighborhood_map[idx] = value;
}

@compute @workgroup_size(16, 16, 1)
fn seathru_estimate_neighborhoodmap_preprocessing(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if (global_id.x < neighborhoodmap_parameters.width && global_id.y < neighborhoodmap_parameters.height) {
        let X_idx = global_id.y * neighborhoodmap_parameters.width + global_id.x;

        set_X_value(X_idx, u32(0), f32(global_id.x));
        set_X_value(X_idx, u32(1), f32(global_id.y));
        set_X_value(X_idx, u32(2), get_depth_value(global_id.x, global_id.y));
    }
}

@compute @workgroup_size(16, 16, 1)
fn seathru_estimate_neighborhoodmap_postprocessing(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if (global_id.x < neighborhoodmap_parameters.width && global_id.y < neighborhoodmap_parameters.height) {
        let idx = global_id.y * neighborhoodmap_parameters.width + global_id.x;

        set_neighborhood_value(global_id.x, global_id.y, y_pred[idx]);
    }
}
