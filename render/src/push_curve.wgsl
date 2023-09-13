//! Compute shader responsible for pushing points to curves.

/// Temporary way for CPU to communicate with this shader.
struct PushCommand {
    pos: vec2<f32>,
    curve_index: u32,
    _pad: u32,
}


/// Data required to push a point.
@group(0) @binding(0) var<uniform> commands: array<PushCommand, $$COMMAND_BUF_SIZE$$>;

/// Vertex buffer.
@group(0) @binding(1) var<storage, read_write> vertices: VertexBuffer;

/// Index buffer.
@group(0) @binding(2) var<storage, read_write> indices: IndexBuffer;

/// Array of active curves.
@group(0) @binding(3) var<storage, read_write> curves: array<Curve>;


@compute
@workgroup_size(1) // TODO: Update to better value!
fn push_curve_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let cmd = commands[global_invocation_id.x];

    // Push the incoming point to this curve
    push_vertex(cmd.pos, cmd.curve_index);
}

/// Given a new point and a curve, calculate and write the values of:
///  - Two vertices, placing them adjacent to each other
///  - Six indices (one quad)
///  - The midpoint for a curve that has 2+ points
fn push_vertex(pos: vec2<f32>, curve_index: u32) {
    var curve = curves[curve_index];
    let index = atomicAdd(&vertices.size, 2);

    var new_ray = vec2<f32>(0.0, 0.0);
    if (curve.tail_index >= 0) {
        // If we have 2+ points, calculate the new ray based on the previous point
        new_ray = calculate_new_rays(curve.head_index, pos);

        // Write the indices out
        push_quad_indices(curve.head_index, index);

        // Push the linked list
        vertices.data[curve.head_index].next = index;
    }
    else {
        // If this is our first point, indicate that
        curve.tail_index = index;
    }

    // In all cases, write out the new point
    vertices.data[index    ] = Vertex(pos,  new_ray, 1.0, -1);
    vertices.data[index + 1] = Vertex(pos, -new_ray, 1.0, -1);

    curve.head_index = index;
    curves[curve_index] = curve;
}

/// Place the 6 indices (one quad) into the index buffer.
fn push_quad_indices(i: i32, j: i32) {
    let index = atomicAdd(&indices.size, 6);

    // 0, 1, 2
    indices.data[index    ] = i;
    indices.data[index + 1] = i + 1;
    indices.data[index + 2] = j;

    // 1, 3, 2
    indices.data[index + 3] = i + 1;
    indices.data[index + 4] = j + 1;
    indices.data[index + 5] = j;
}

/// Compute the rays for the current- and mid-point of the segment list.
fn calculate_new_rays(prev_index: i32, new_pos: vec2<f32>) -> vec2<f32> {
    let prev_vertex = vertices.data[prev_index];

    let ray_between = new_pos - prev_vertex.pos;
    let mag = length(ray_between);

    if (mag < 0.001) {
        // Basically no difference, just ignore it
        return prev_vertex.ray;
    }

    let new_ray = vec2<f32>(-ray_between.y, ray_between.x) / mag;

    if (prev_vertex.ray.x == 0.0 && prev_vertex.ray.y == 0.0) {
        // If the previous ray is invalid, use the same for both
        vertices.data[prev_index    ].ray =  new_ray;
        vertices.data[prev_index + 1].ray = -new_ray;
        return new_ray;
    }

    // Otherwise, calculate the new midpoint ray
    // Note: I know this is difficult to read, just trust me
    // on its correctness lol
    let u = normalize(prev_vertex.ray);
    let v = new_ray;
    let w = vec2<f32>(u.y, -u.x);

    var det = dot(v, w);

    // If it's really small, round it to the previous ray
    if (det * sign(det) < 0.1) {
        // This creates a "poisoned" segment with a small visual artifact
        // on both ends, but it's not hugely important to fix
        return prev_vertex.ray;
    }

    let mul = dot(v, v - u) / det;
    let mid_ray = (w * mul + u);
    vertices.data[prev_index    ].ray =  mid_ray;
    vertices.data[prev_index + 1].ray = -mid_ray;

    return new_ray;
}
