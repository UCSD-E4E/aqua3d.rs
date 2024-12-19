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

@compute @workgroup_size(64, 1, 1)
fn column_depth_segmentation(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    const workgroup_size: i32 = 64;

    let epsilon = estimate_neighborhood_map_parameters.epsilon;

    let x = global_id.x;
    var y: i32 = 0;

    if (check_x_y(i32(x), i32(y)))
    {
        set_neighborhood(x, u32(y), get_image_index(x, u32(y)));

        y = 1;
        while (y >= 0 && y < i32(image_size.height)) {
            if (abs(get_depth_value(x, u32(y - 1)) - get_depth_value(x, u32(y))) <= epsilon) {
                set_neighborhood(x, u32(y), get_neighborhood(x, u32(y - 1)));
            }
            else {
                set_neighborhood(x, u32(y), get_image_index(x, u32(y)));
            }

            y += 1;
        }
    }
}

@compute @workgroup_size(1, 64, 1)
fn merge_colume_depth_segmentation(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    const workgroup_size: i32 = 64;

    let epsilon = estimate_neighborhood_map_parameters.epsilon;
    
    var x: i32 = 0;
    var y: i32 = i32(global_id.y);

    if (check_x_y(x, y)) {
        x = 1;
        while (x >= 0 && x < i32(image_size.width)) {
            if (abs(get_depth_value(u32(x - 1), u32(y)) - get_depth_value(u32(x), u32(y))) <= epsilon) {
                let neighborhood = get_neighborhood(u32(x - 1), u32(y));
                set_neighborhood(u32(x), u32(y), neighborhood);

                // move up
                // move down
            }

            x += 1;
            y = i32(global_id.y);
        }
    }

    let a = neighborhood_map[0];
}
