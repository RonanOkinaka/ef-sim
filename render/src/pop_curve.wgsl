//! Compute shader responsible for popping points from curves.

/// Data required to pop a point.
@group(0) @binding(0) var<storage, read> commands: PopCommandBuffer;

/// Vertex buffer.
@group(0) @binding(1) var<storage, read_write> vertices: VertexBuffer;

/// Index buffer.
@group(0) @binding(2) var<storage, read_write> indices: IndexBuffer;

/// Shader state.
@group(0) @binding(3) var<storage, read_write> state: ComputeState;


@compute
@workgroup_size($$WORKGROUP_SIZE_X$$)
fn pop_curve_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    if (global_invocation_id.x >= commands.size) {
        return;
    }

    pop_vertex(commands.data[global_invocation_id.x].curve_index);
}

/// Pop a point off the back of a curve.
fn pop_vertex(curve_index: u32) {
    var curve = state.curves.data[curve_index];

    // If we have no points, do nothing
    if (curve.tail_index < 0) {
        return;
    }

    // Otherwise, turn the segment invisible and decrement the curve's size
    vertices.data[curve.tail_index    ].alpha = 0.0;
    vertices.data[curve.tail_index + 1].alpha = 0.0;
    curve.num_points -= 1;

    // Traverse the linked list
    let next_index = vertices.data[curve.tail_index].next;
    if (curve.tail_index == curve.head_index) {
        curve.head_index = -1;
        atomicSub(&state.curves.size, 1);
    } else {
        let next_vertex = vertices.data[next_index];

        // This really does store the segment index
        push_free_index(vertices.data[next_index + 1].next);

        var ray = vec2<f32>(0.0, 0.0);
        if (next_vertex.next >= 0) {
            // If we have 2+ vertices after us, update the central ray
            // TODO: Factor this into its own function? (Partially shared with push_curve.wgsl)
            let end = vertices.data[next_vertex.next].pos;

            let ray_between = end - next_vertex.pos;
            let mag = length(ray_between);

            if (mag > 0.001) {
                ray = vec2<f32>(-ray_between.y, ray_between.x) / mag;
                vertices.data[next_index    ].ray =  ray;
                vertices.data[next_index + 1].ray = -ray;
            }
        } else {
            // If we have just one, mark its ray invalid
            vertices.data[next_index    ].ray =  ray;
            vertices.data[next_index + 1].ray = -ray;
        }
    }
    push_free_vertex(curve.tail_index);
    curve.tail_index = next_index;

    state.curves.data[curve_index] = curve;
}

fn push_free_vertex(index: i32) {
    let slot = atomicAdd(&state.vert_free.size, 1);

    // If we get interrupted here, we're in trouble
    state.vert_free.data[slot] = index;
    // But, WGPU should synchronize the dispatches to prevent it.
}

fn  push_free_index(index: i32) {
    // Must be careful that another segment doesn't
    // get drawn by this index set
    indices.data[index    ] = 0;
    indices.data[index + 1] = 0;
    indices.data[index + 2] = 0;
    indices.data[index + 3] = 0;
    indices.data[index + 4] = 0;
    indices.data[index + 5] = 0;

    let slot = atomicAdd(&state.indx_free.size, 1);
    state.indx_free.data[slot] = index;
}
