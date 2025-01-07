struct ImageSize {
    width: u32,
    height: u32
}

struct EstimateNeighborhoodMapParameters {
    epsilon: f32
}

@group(0)
@binding(0)
var<storage, read> image_size: ImageSize;

@group(0)
@binding(1)
var<storage, read> estimate_neighborhood_map_parameters: EstimateNeighborhoodMapParameters;

@group(0)
@binding(2)
var<storage, read> depths: array<f32>;

@group(0)
@binding(3)
var<storage, read_write> neighborhood_map: array<u32>;

fn check_x_y(x: i32, y: i32) -> bool {
    return x >= 0 && x < i32(image_size.width) && y >= 0 && y < i32(image_size.height);
}

fn get_image_index(x: u32, y: u32) -> u32 {
    return y * image_size.width + x;
}

fn get_depth_value(x: u32, y: u32) -> f32 {
    let idx = get_image_index(x, y);

    return depths[idx];
}

fn get_neighborhood(x: u32, y: u32) -> u32 {
    let idx = get_image_index(x, y);

    return neighborhood_map[idx];
}

fn set_neighborhood(x: u32, y: u32, value: u32) {
    let idx = get_image_index(x, y);

    neighborhood_map[idx] = value;
}

@compute @workgroup_size(16, 16, 1)
fn estimate_neighborhood_map(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) workgroups: vec3<u32>
) {
    const workgroup_size: i32 = 16;
    const local_group_count: i32 = 4;
    const local_group_size: i32 = 64;

    let epsilon = estimate_neighborhood_map_parameters.epsilon;

    let group_id = vec3<i32>(floor(vec3<f32>(global_id) / f32(workgroup_size)));
    let group_start = group_id * workgroup_size;

    let local_group_id = vec3<i32>(ceil(vec3<f32>(local_id) / f32(local_group_count)));

    var direction = vec3<i32>(local_id % 2);
    let start = group_id * local_group_count * local_group_size + local_group_id * local_group_size + (local_group_size / 2) - 1 + direction;
    direction *= 2;
    direction -= 1;

    var x = start.x;
    var y = start.y;

    if (check_x_y(x, y))
    {
        set_neighborhood(u32(x), u32(y), get_image_index(u32(x), u32(y)));
    }

    while (check_x_y(x, y)) {
        while (y >= 0 && y < i32(image_size.height)) {       
            let depth_value = get_depth_value(u32(x), u32(y));
            let neighborhood_value = get_neighborhood(u32(x), u32(y));

            if (check_x_y(x + direction.x, y)) {
                var next_neighborhood_value: u32;
                if (abs(depth_value - get_depth_value(u32(x + direction.x), u32(y))) <= epsilon) {
                    next_neighborhood_value = neighborhood_value;
                }
                else {
                    next_neighborhood_value = get_image_index(u32(x + direction.x), u32(y));
                }

                set_neighborhood(u32(x + direction.x), u32(y), next_neighborhood_value);
            }
            
            if (check_x_y(x, y + direction.y)) {
                var next_neighborhood_value: u32;
                if (abs(depth_value - get_depth_value(u32(x), u32(y + direction.y))) <= epsilon) {
                    next_neighborhood_value = neighborhood_value;
                }
                else {
                    next_neighborhood_value = get_image_index(u32(x), u32(y + direction.y));
                }

                set_neighborhood(u32(x), u32(y + direction.y), next_neighborhood_value);
            }

            y += direction.y;
        }

        y = start.y;
        x += direction.x;
    }
}