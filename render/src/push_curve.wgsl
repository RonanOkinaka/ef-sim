//! Compute shader responsible for pushing points to curves.

/// Data required for pushing to a curve.
struct PushCommand {
    pos: vec2<f32>,
    curve_index: u32,
    _pad: u32,
}

struct PushCommandBuffer {
    size: vec2<u32>, // Only .x is used, rest is padding
    data: array<PushCommand, $$PUSH_BUF_LOGICAL_SIZE$$>,
}


/// Data required to push a point.
@group(0) @binding(0) var<storage, read> commands: PushCommandBuffer;

/// Vertex buffer.
@group(0) @binding(1) var<storage, read_write> vertices: VertexBuffer;

/// Index buffer.
@group(0) @binding(2) var<storage, read_write> indices: IndexBuffer;

/// Shader state.
@group(0) @binding(3) var<storage, read_write> state: ComputeState;


@compute
@workgroup_size($$WORKGROUP_SIZE_X$$)
fn push_curve_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    if (global_invocation_id.x >= commands.size.x) {
        return;
    }

    let cmd = commands.data[global_invocation_id.x];

    // Push the incoming point to this curve
    push_vertex(cmd.pos, cmd.curve_index);
}

// Points to "push_common.wgsl"
$$INCLUDE_PUSH_COMMON_WGSL$$
