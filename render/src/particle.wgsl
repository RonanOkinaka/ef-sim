//! Shader to calculate particle IVPs and output to curve buffers.

/// Stores charge data.
struct Charge {
    pos: vec2<f32>,
    charge: f32,
    _pad: u32,
}

struct ChargeBuffer {
    size: vec4<u32>,
    data: array<Charge, $$CHARGE_BUF_LOGICAL_SIZE$$>,
}

struct Params {
    rand_value: u32,
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

/// Helper data from CPU.
@group(0) @binding(5) var<uniform> params: Params;


@compute
@workgroup_size($$WORKGROUP_SIZE_X$$)
fn particle_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let curve_index = global_invocation_id.x;
    if (curve_index >= state.curves.size) {
        return;
    }

    let head_index = state.curves.data[curve_index].head_index;
    var alive = bool(state.curves.data[curve_index].alive);
    var num_points = state.curves.data[curve_index].num_points;

    var pos = vec2<f32>(0.0, 0.0);

    if (head_index < 0) {
        // TODO: We can do much better than this
        let rand = (params.rand_value ^ curve_index);
        let charge_index = rand % charges.size.x;

        // The trig functions seem to struggle with large values, so reduce them
        let angle = f32(rand % 2097152u);

        let offset = $$CHARGE_COLLISION_RADIUS$$ * vec2<f32>(cos(angle), sin(angle));
        pos = charges.data[charge_index].pos + offset;

        // Reset the curve
        if (charges.data[charge_index].charge > 0.0) {
            alive = true;
            state.curves.data[curve_index] = Curve(-1, -1, 0, 1);
        } else {
            alive = false;
        }
    }
    // Calculate our next value
    else if (alive) {
        // TODO: Move head position into curve data? Can potentially benefit here and in push_vertex
        pos = vertices.data[head_index].pos;
        var ds = vec2<f32>(0.0, 0.0);
        for (var i = 0u; i < charges.size.x; i += 1u) {
            let ray = pos - charges.data[i].pos;
            var mag = length(ray);

            if (mag + 0.05 < $$CHARGE_COLLISION_RADIUS$$) {
                alive = false;
                break;
            }

            mag *= mag;
            ds += charges.data[i].charge * ray / mag;
        }

        // Don't want to get stuck in a sink (e.g. exactly between two equal charges)
        if (length(ds) > 0.001) {
            pos += normalize(ds) * 0.01;
        } else {
            alive = false;
        }
    }

    if (alive) {
        // We'll push here instead of dispatching another shader for it
        let ret = push_vertex(pos, curve_index);
        num_points = ret.x;
        alive = bool(ret.y);
    }

    var should_pop = (num_points >= $$MAX_POINTS_PER_CURVE$$); // TODO: This should be dynamic
    if (!alive) {
        state.curves.data[curve_index].alive = 0;
        should_pop |= bool(num_points);
    }

    // Pops will stay separate, however, mostly for synchronization reasons
    if (should_pop) {
        let pop_index = atomicAdd(&pop_commands.size, 1u);
        pop_commands.data[pop_index] = PopCommand(curve_index);

        let num_workgroups_x = (pop_index + $$WORKGROUP_SIZE_X$$u) / ($$WORKGROUP_SIZE_X$$u);
        atomicMax(&pop_commands.dispatch.workgroups_x, num_workgroups_x);
    }
}

$$INCLUDE_PUSH_COMMON_WGSL$$
