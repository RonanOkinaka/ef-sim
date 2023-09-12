//! Compute shader responsible for popping points from curves.

/// Temporary way for CPU to communicate with this shader.
struct PopCommand {
    // Must be stride 16...
    curve_index: vec4<u32>,
}

/// Data required to pop a point.
@group(0) @binding(0) var<uniform> commands: array<PopCommand, 16384>;

/// Vertex buffer.
@group(0) @binding(1) var<storage, read_write> vertices: VertexBuffer;

/// Array of active curves.
@group(0) @binding(2) var<storage, read_write> curves: array<Curve>;


@compute
@workgroup_size(1) // TODO: Update to better value!
fn pop_curve_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    pop_vertex(commands[global_invocation_id.x].curve_index.x);
}

/// Pop a point off the back of a curve.
fn pop_vertex(curve_index: u32) {
    var curve = curves[curve_index];

    // If we have no points, do nothing
    if (curve.tail_index < 0) {
        return;
    }

    // Otherwise, turn the segment invisible
    vertices.data[curve.tail_index    ].alpha = 0.0;
    vertices.data[curve.tail_index + 1].alpha = 0.0;

    // Traverse the linked list
    let next_index = vertices.data[curve.tail_index].next;
    if (curve.tail_index == curve.head_index) {
        curve.head_index = -1;
    } else {
        let next_vertex = vertices.data[next_index];

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
    curve.tail_index = next_index;

    curves[curve_index] = curve;
}
