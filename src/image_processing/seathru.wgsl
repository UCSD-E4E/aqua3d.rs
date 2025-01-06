// Implementation of
// connected components: https://doi.org/10.1145/3208040.3208041

struct Parameters {
    epsilon: f32,
    height: u32,
    width: u32
}

@group(0)
@binding(0)
var<storage, read> parameters: Parameters;

@group(0)
@binding(1)
var<storage, read> depths: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> neighborhood_map: array<atomic<u32>>;

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
                    ret = atomicCompareExchangeWeak(&neighborhood_map[y_root], y_root, x_root).old_value;
                    if (ret != y_root) {
                        y_root = ret;
                        repeat = true;
                    }
                }
                else {
                    ret = atomicCompareExchangeWeak(&neighborhood_map[x_root], x_root, y_root).old_value;
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
    var curr = neighborhood_map[idx];
    if (curr != idx) {
        var next = idx;
        var prev = idx;
        
        while (curr > neighborhood_map[curr]) {
            next = neighborhood_map[curr];
            neighborhood_map[prev] = next;
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

fn get_position(index: u32) -> vec2<u32> {
    var position: vec2<u32>;
    position.x = index % parameters.width;
    position.y = index / parameters.width;

    return position;
}

fn get_depth(position: vec2<u32>) -> f32 {
    let idx = get_index(position);

    return depths[idx];
}

fn set_neighborhood(position: vec2<u32>, value: u32) {
    let idx = get_index(position);

    neighborhood_map[idx] = value;
}

fn get_index(position: vec2<u32>) -> u32 {
    return position.x + position.y * parameters.width;
}

fn calculate_distance(a: vec2<u32>, b: vec2<u32>) -> f32 {
    const inf: f32 = 0x1.fffffep+127f;

    let diff1 = f32(a.x) - f32(b.x);
    let diff2 = f32(a.y) - f32(b.y);
    let xy_dist = sqrt(diff1 * diff1 + diff2 * diff2);
    if (xy_dist > 1) {
        return inf;
    }

    let diff = get_depth(a) - get_depth(b);
    return sqrt(diff * diff);
}

fn check_distance(a: vec2<u32>, b: vec2<u32>) -> bool {
    return calculate_distance(a, b) < parameters.epsilon;
}

@compute @workgroup_size(16, 16, 1)
fn seathru_estimate_neighborhoodmap_preprocessing(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if (global_id.x < parameters.width && global_id.y < parameters.height) {
        let idx = get_index(global_id.xy);
        set_neighborhood(global_id.xy, idx);

        if (global_id.y > 0) { // possibly connected up
            var up = global_id.xy;
            up.y -= 1u;
            let up_idx = get_index(up);

            if (check_distance(global_id.xy, up)) {
                set_neighborhood(global_id.xy, up_idx);
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn seathru_estimate_neighborhoodmap_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if (global_id.x < parameters.width && global_id.y < parameters.height) {
        let idx = get_index(global_id.xy);

        // Up
        if (global_id.y > 0) {
            var up = global_id.xy;
            up.y -= 1u;
            let up_idx = get_index(up);

            if (check_distance(global_id.xy, up)) {
                union_trees(idx, up_idx);
            }
        }

        // Down
        if (global_id.y < parameters.height - 1) {
            var down = global_id.xy;
            down.y += 1u;
            let down_idx = get_index(down);

            if (check_distance(global_id.xy, down)) {
                union_trees(idx, down_idx);
            }
        }

        // Left
        if (global_id.x > 0) {
            var left = global_id.xy;
            left.x -= 1u;
            let left_idx = get_index(left);

            if (check_distance(global_id.xy, left)) {
                union_trees(idx, left_idx);
            }
        }

        // Right
        if (global_id.x < parameters.width - 1) {
            var right = global_id.xy;
            right.x += 1u;
            let right_idx = get_index(right);

            if (check_distance(global_id.xy, right)) {
                union_trees(idx, right_idx);
            }
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn seathru_estimate_neighborhoodmap_postprocessing(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let count = parameters.width * parameters.height;
    let incr = num_workgroups.x * 256u;

    var v: u32;
    for (v = global_id.x; v < count; v += incr) {
        var next = neighborhood_map[v];
        var vstat = neighborhood_map[v];
        let old = vstat;

        while (vstat > next) {
            vstat = next;
            next = neighborhood_map[vstat];
        }

        if (old != vstat) {
            neighborhood_map[v] = vstat;
        }
    }
//   for (int v = global_id.x; v < nodes; v += incr) {
//     int next, vstat = nstat[v];
//     const int old = vstat;
//     while (vstat > (next = nstat[vstat])) {
//       vstat = next;
//     }
//     if (old != vstat) nstat[v] = vstat;
//   }
}