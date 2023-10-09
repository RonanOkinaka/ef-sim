//! Shared functionality for pushing a point to a curve.

/// Given a new point and a curve, calculate and write the values of:
///  - Two vertices, placing them adjacent to each other
///  - Six indices (one quad)
///  - The midpoint for a curve that has 2+ points
/// Returns the number of points now in the curve.
fn push_vertex(pos: vec2<f32>, curve_index: u32) -> vec2<i32> {
    var curve = state.curves.data[curve_index];
    let vertex_index = allocate_vertices();

    var index_index = -1;
    var new_ray = vec2<f32>(0.0, 0.0);
    if (curve.tail_index >= 0) {
        // If we have 2+ points, calculate the new ray based on the previous point
        let prev_vertex = vertices.data[curve.head_index];

        let ray_between = pos - prev_vertex.pos;
        let mag = length(ray_between);

        if (mag < 0.001) {
            // Basically no difference, just ignore it
            new_ray = prev_vertex.ray;
        } else {
            new_ray = vec2<f32>(-ray_between.y, ray_between.x) / mag;

            if (prev_vertex.ray.x == 0.0 && prev_vertex.ray.y == 0.0) {
                // If the previous ray is invalid, use the same for both
                vertices.data[curve.head_index    ].ray =  new_ray;
                vertices.data[curve.head_index + 1].ray = -new_ray;
            } else {
                // Otherwise, calculate the new midpoint ray
                // Note: I know this is difficult to read, just trust me
                // on its correctness lol
                let u = normalize(prev_vertex.ray);
                let v = new_ray;

                let diff = dot(u, v);
                if (abs(diff) > 0.8) {
                    if (diff > 0.0) {
                        // Going straight, use the old ray
                        new_ray = prev_vertex.ray;
                    } else {
                        // Turned exactly around, kill the curve
                        return vec2<i32>(curve.num_points, 0);
                    }
                } else {
                    let w = vec2<f32>(u.y, -u.x);
                    let mul = dot(v, v - u) / dot(v, w);

                    let mid_ray = (w * mul + u);
                    vertices.data[curve.head_index    ].ray =  mid_ray;
                    vertices.data[curve.head_index + 1].ray = -mid_ray;
                }
            }
        }

        // Write the indices out
        index_index = push_quad_indices(curve.head_index, vertex_index);

        // Push the linked list
        vertices.data[curve.head_index].next = vertex_index;
    }
    else {
        // If this is our first point, indicate that
        curve.tail_index = vertex_index;
    }

    // In all cases, write out the new point
    vertices.data[vertex_index    ] = Vertex(pos,  new_ray, 1.0, -1);
    // The vertex above stores the <next> node in the linked list (-1 == nothing)
    // The one below stores the index into the index buffer
    vertices.data[vertex_index + 1] = Vertex(pos, -new_ray, 1.0, index_index);

    curve.head_index = vertex_index;
    curve.num_points += 1;
    state.curves.data[curve_index] = curve;

    return vec2<i32>(curve.num_points, 1);
}

/// Allocate space for two vertices.
fn allocate_vertices() -> i32 {
    let size = atomicSub(&state.vert_free.size, 1);
    if (size <= 0) {
        atomicAdd(&state.vert_free.size, 1);
        return atomicAdd(&vertices.size, 2);
    }

    return state.vert_free.data[size - 1];
}

/// Allocate space for six indices.
fn allocate_indices() -> i32 {
    let size = atomicSub(&state.indx_free.size, 1);
    if (size <= 0) {
        atomicAdd(&state.indx_free.size, 1);
        return atomicAdd(&state.draw.num_indices, 6);
    }

    return state.indx_free.data[size - 1];
}

/// Place the 6 indices (one quad) into the index buffer.
fn push_quad_indices(i: i32, j: i32) -> i32 {
    let index = allocate_indices();

    // 0, 1, 2
    indices.data[index    ] = i;
    indices.data[index + 1] = i + 1;
    indices.data[index + 2] = j;

    // 1, 3, 2
    indices.data[index + 3] = i + 1;
    indices.data[index + 4] = j + 1;
    indices.data[index + 5] = j;

    return index;
}
