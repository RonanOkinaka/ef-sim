//! Shader to calculate particle IVPs and output to curve buffers.

/// Stores charge data.
struct Charge {
    pos: vec2<f32>,
    _pad: vec2<u32>,
}

struct ChargeBuffer {
    size: vec4<i32>,
    data: array<Charge, $$CHARGE_BUF_LOGICAL_SIZE$$>,
}


/// Shader state.
@group(0) @binding(0) var<storage, read_write> state: ComputeState;

/// Charge buffer.
@group(0) @binding(1) var<uniform> charges: ChargeBuffer;

/// Pop command buf.
@group(0) @binding(2) var<storage, read_write> pop_commands: PopCommandBuffer;

/// Vertex buffer.
@group(0) @binding(3) var<storage, read_write> vertices: VertexBuffer;

/// Index buffer.
@group(0) @binding(4) var<storage, read_write> indices: IndexBuffer;


@compute
@workgroup_size($$WORKGROUP_SIZE_X$$)
fn particle_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let curve_index = global_invocation_id.x;
    if (curve_index >= state.curves.size) {
        return;
    }

    let head_index = state.curves.data[curve_index].head_index;

    // TODO: If we have no head, create a new point near an existing particle
    if (head_index < 0) {
        return;
    }

    // TODO: Move head position into curve data? Can potentially benefit here and in push_vertex
    var pos = vertices.data[head_index].pos;

    // Calculate our next value
    pos += vec2<f32>(0.01, 0.0);

    // We'll push here instead of dispatching another shader for it
    push_vertex(pos, curve_index);

    // Pops will stay separate, however, mostly for synchronization reasons
}

$$INCLUDE_PUSH_COMMON_WGSL$$
