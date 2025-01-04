// Implementation of
// https://doi.org/10.1145/3605573.3605594

struct Parameters {
    uint count;
    uint dim;
    float epsilon;
    uint min_points;
}

[vk::binding(0, 0)]
ParameterBlock<Parameters> parameters;

[vk::binding(1, 0)]
StructuredBuffer<float> X;

[vk::binding(2, 0)]
RWStructuredBuffer<uint> core_points;

[vk::binding(3, 0)]
RWStructuredBuffer<uint> tree;

[vk::binding(4, 0)]
RWStructuredBuffer<uint> y_pred: array<u32>;

float calculate_distance(uint x_idx, uint y_idx) {
    float sum = 0;

    for (uint i = 0; i < parameters.dim; i += 1) {
        let idx1 = x_idx * parameters.dim + i;
        let idx2 = y_idx * parameters.dim + i;

        let diff = X[idx1] - X[idx2];

        sum += (diff * diff);
    }

    return sqrt(sum);
}

bool is_core_point(uint idx) {
    return core_points[idx] != 0;
}

void union_trees(uint x_idx, uint y_idx) {
    var x_root = find_root(x_idx);
    
    if (x_idx > y_idx) {
        var y_root = find_root(y_idx);
        bool repeat = false;

        do {
            repeat = false;

            if (x_root != y_root) {
                uint ret;
                if (x_root < y_root) {
                    InterlockedCompareExchange(tree[y_root], y_root, x_root, ret);
                    if (ret != y_root) {
                        y_root = ret;
                        repeat = true;
                    }
                }
                else {
                    InterlockedCompareExchange(tree[x_root], x_root, y_root, ret);
                    if (ret != x_root) {
                        x_root = ret;
                        repeat = true;
                    }
                }
            }
        } while (repeat);
    }
}

[ForceInline]
uint find_root(const uint idx)
{
  int curr = tree[idx];
  if (curr != idx) {
    
    int next, prev = idx;
    while (curr > (next = tree[curr])) {
      tree[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

bool is_member_of_any_cluster(uint x_idx) {
    return tree[x_idx] != x_idx;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void dbscan_preprocessing(
    uint3 global_id : SV_DispatchThreadID
) {
    if (global_id.x < parameters.count) {
        uint neighbor_count = 0;

        if (parameters.min_points > 2 ) {
            for (uint i = 0; i < parameters.count; i += 1) {
                if (i != global_id.x && neighbor_count < parameters.min_points) {
                    let dist = calculate_distance(global_id.x, i);

                    if (dist <= parameters.epsilon) {
                        neighbor_count += 1;
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
            core_points[global_id.x] = 0;
        }

        tree[global_id.x] = global_id.x; // Initialize the tree.
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void dbscan_main(
    uint3 global_id : SV_DispatchThreadID
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

[shader("compute")]
[numthreads(64, 1, 1)]
void dbscan_postprocessing(
    uint3 global_id : SV_DispatchThreadID
) {
    y_pred[global_id.x] = tree[global_id.x];
}