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
var<storage, read_write> tree: array<atomic<u32>>;

@group(0)
@binding(4)
var<storage, read_write> y_pred: array<u32>;

fn calculate_distance(x_idx: u32, y_idx: u32) -> f32 {
    var sum: f32 = 0;

    var i: u32;
    for (i = u32(0); i < parameters.dim; i += u32(1)) {
        let idx1 = x_idx * parameters.dim + i;
        let idx2 = y_idx * parameters.dim + i;

        let diff = X[idx1] - X[idx2];

        sum += (diff * diff);
    }

    return sqrt(sum);
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
                    ret = atomicCompareExchangeWeak(&tree[y_root], y_root, x_root).old_value;
                    if (ret != y_root) {
                        y_root = ret;
                        repeat = true;
                    }
                }
                else {
                    ret = atomicCompareExchangeWeak(&tree[x_root], x_root, y_root).old_value;
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
    var curr = tree[idx];
    if (curr != idx) {
        var next = idx;
        var prev = idx;
        
        while (curr > tree[curr]) {
            next = tree[curr];
            tree[prev] = next;
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

fn is_member_of_any_cluster(x_idx: u32) -> bool {
    return tree[x_idx] != x_idx;
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

        tree[global_id.x] = global_id.x; // Initialize the tree.
    }
}

@compute @workgroup_size(16, 16, 1)
fn dbscan_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if (global_id.x < parameters.count &&
        global_id.y < parameters.count && 
        global_id.x != global_id.y &&
        is_core_point(global_id.x) &&
        calculate_distance(global_id.x, global_id.y) < parameters.epsilon) {

            if (is_core_point(global_id.y) || !is_member_of_any_cluster(global_id.y)) {
                union_trees(global_id.x, global_id.y);
            }
    }
}

@compute @workgroup_size(64, 1, 1)
fn dbscan_postprocessing(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    y_pred[global_id.x] = tree[global_id.x];
}